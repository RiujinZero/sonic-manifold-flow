"""Optional CUDA-graph-captured block-causal stewardship traffic
bookkeeping.

Not a precision feature (unlike gator/mx_precision.py) -- runs at
whatever dtype the caller already uses and exists purely to eliminate
the per-call CPU dispatch overhead of the reshape/sum/cumsum/permute
sequence inside ``_dense_stewardship_context`` that computes each
cell's cumulative block-causal traffic (see gator/models/language.py),
the same overhead-elimination mechanism already shipped for node
compute, readout (gator/mx_precision.py), receiver memory
(gator/cuda_graph_receiver_memory.py), and broadcast-mix cell coupling
(gator/cuda_graph_broadcast_mix.py). Read those docstrings first; the
cache-threading pattern here is closest to the receiver-memory one
(a running numerator/write-sum carried relay to relay).

Unlike node compute/readout/broadcast-mix, this block has no trainable
parameters at all -- pure tensor arithmetic over ``node_locations``/
``writer_mass``/``write`` and the running cache. No value-copy bridge
is needed: with nothing to route a live parameter's gradient through,
the inputs flow directly into ``make_graphed_callables``' returned
callable, same as any ordinary graphed module.

Scope: only the ``traffic_scale=1.0`` case, i.e. ``ownership is None``
in ``_dense_stewardship_context`` -- broadcast_mix's actual call site.
The ``ownership is not None`` (stewardship transport) branch uses a
different ``traffic_scale`` and is out of scope here (dead code for
``cell_transport_mode='broadcast_mix'``, this production config).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class _StewardshipTrafficBlock(nn.Module):
    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        node_locations: torch.Tensor,
        writer_mass: torch.Tensor,
        write: torch.Tensor,
        previous_numerator: torch.Tensor,
        previous_write: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, cells, tokens, nodes = node_locations.shape
        block_size = self.block_size
        blocks = tokens // block_size
        accumulator_dtype = previous_numerator.dtype

        token_numerator = node_locations * writer_mass.unsqueeze(-1)
        block_numerator = token_numerator.reshape(
            batch, cells, blocks, block_size, nodes
        ).sum(dim=3, dtype=accumulator_dtype).permute(0, 2, 1, 3)
        block_write = write.reshape(batch, blocks, block_size).sum(
            dim=2, dtype=accumulator_dtype
        )
        cumulative_numerator = (
            block_numerator.cumsum(dim=1) + previous_numerator.unsqueeze(1)
        )
        cumulative_write = (
            block_write.cumsum(dim=1) + previous_write.unsqueeze(1)
        ).clamp_min(1e-6)
        traffic_blocks = cumulative_numerator / cumulative_write[:, :, None, None]
        traffic = traffic_blocks[:, :, None].expand(
            -1, -1, block_size, -1, -1
        ).reshape(batch, tokens, cells, nodes).permute(0, 2, 1, 3)
        next_numerator = previous_numerator + block_numerator.sum(dim=1)
        next_write = previous_write + block_write.sum(dim=1)
        return traffic, next_numerator, next_write


_STEWARDSHIP_TRAFFIC_CUDAGRAPH_CACHE: dict[tuple, tuple[_StewardshipTrafficBlock, Callable]] = {}


def cuda_graph_stewardship_traffic(
    node_locations: torch.Tensor,
    writer_mass: torch.Tensor,
    write: torch.Tensor,
    previous_numerator: torch.Tensor,
    previous_write: torch.Tensor,
    *,
    block_size: int,
    states_dtype: torch.dtype,
    graph_slot: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA-graph-captured block-causal stewardship traffic bookkeeping.

    Returns ``(traffic, next_numerator, next_write)`` matching
    ``_dense_stewardship_context``'s ``traffic_scale=1.0`` (``ownership
    is None``) computation exactly, just faster. ``traffic`` is cast to
    ``states_dtype``; ``next_numerator``/``next_write`` stay in the
    accumulator dtype (fp32 for bf16/fp16 states, matching
    ``GatorARWRelayBlockCache``'s convention) -- feed them back in
    unchanged as ``previous_numerator``/``previous_write`` for the next
    call sharing this cache chain.

    ``previous_numerator``/``previous_write`` must always be real
    tensors, never a Python ``None`` -- CUDA-graph capture needs a
    fixed input structure; the caller constructs zeros for the very
    first relay/block, same convention as
    ``cuda_graph_linear_scan_receiver_context``'s ``cache_forward_state``.

    ``graph_slot`` -- **required to be distinct per call-site when this
    function is called more than once before a shared backward() fires**
    (this model's real usage: once per relay). See
    ``gator.mx_precision.te_fp8_cudagraph_node_linear``'s docstring for
    the direct-test evidence this caveat is based on.

    Shape-cached per ``(batch, cells, tokens, nodes, block_size, dtypes,
    device, graph_slot)`` -- the first call for a given shape pays
    graph-capture warmup cost; later calls with the same shape reuse the
    captured graph. Batch size and context length must stay fixed
    across calls sharing a cache entry, same requirement as any
    CUDA-graph-captured code in this codebase.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("cuda_graph_stewardship_traffic requires a CUDA device")
    if node_locations.ndim != 4:
        raise ValueError(
            "cuda_graph_stewardship_traffic node_locations must be "
            "[batch,cells,tokens,nodes]"
        )
    batch, cells, tokens, nodes = node_locations.shape
    if tokens % block_size:
        raise ValueError("traffic token length must be divisible by block_size")

    key = (
        batch, cells, tokens, nodes, block_size,
        node_locations.dtype, writer_mass.dtype, write.dtype,
        previous_numerator.dtype, node_locations.device, graph_slot,
    )
    cached = _STEWARDSHIP_TRAFFIC_CUDAGRAPH_CACHE.get(key)
    if cached is None:
        raw_module = _StewardshipTrafficBlock(block_size).to(device=node_locations.device)
        sample_locations = torch.zeros(
            batch, cells, tokens, nodes, device=node_locations.device,
            dtype=node_locations.dtype, requires_grad=True,
        )
        sample_writer_mass = torch.zeros(
            batch, cells, tokens, device=node_locations.device,
            dtype=writer_mass.dtype, requires_grad=True,
        )
        sample_write = torch.zeros(
            batch, tokens, device=node_locations.device, dtype=write.dtype,
            requires_grad=True,
        )
        sample_previous_numerator = torch.zeros(
            batch, cells, nodes, device=node_locations.device,
            dtype=previous_numerator.dtype, requires_grad=True,
        )
        sample_previous_write = torch.zeros(
            batch, device=node_locations.device, dtype=previous_write.dtype,
            requires_grad=True,
        )
        # make_graphed_callables rejects capture from inside an active
        # autocast context -- real training always wraps forward in
        # torch.autocast(bf16), which would otherwise crash capture on
        # the first real training step. Suppress it for just this
        # one-time capture; replay is a raw kernel replay, not affected.
        with torch.autocast(device_type="cuda", enabled=False):
            graphed_module = torch.cuda.make_graphed_callables(
                raw_module,
                (
                    sample_locations, sample_writer_mass, sample_write,
                    sample_previous_numerator, sample_previous_write,
                ),
                num_warmup_iters=5,
            )
        cached = (raw_module, graphed_module)
        _STEWARDSHIP_TRAFFIC_CUDAGRAPH_CACHE[key] = cached
    raw_module, graphed_module = cached

    traffic, next_numerator, next_write = graphed_module(
        node_locations, writer_mass, write, previous_numerator, previous_write
    )
    # .clone() required on all three outputs, not optional -- see the
    # identical comment in cuda_graph_linear_scan_receiver_context for
    # why (make_graphed_callables' output is a live view into the CUDA
    # graph's static output buffer, overwritten by the next call to this
    # same graphed callable). Bites hardest for next_numerator/next_write
    # specifically, since they're designed to be held and fed into the
    # following relay's call.
    return (
        traffic.clone().to(states_dtype),
        next_numerator.clone(),
        next_write.clone(),
    )
