"""Optional CUDA-graph-captured linear_scan receiver memory.

Not a precision feature (unlike gator/mx_precision.py) -- this runs at
whatever dtype the caller already uses (bf16 in practice) and exists
purely to eliminate the per-call CPU dispatch overhead of
``linear_scan_receiver_context``'s scan (a Python-unrolled loop over
``tokens // chunk_size`` chunks -- see gator/models/memory.py and
gator_linear/core.py's ``decayed_scan``), the same overhead-elimination
mechanism already shipped for node compute and readout in
gator/mx_precision.py (``te_fp8_cudagraph_node_linear``/
``te_fp8_cudagraph_readout_linear`` -- read those docstrings first, the
value-copy/gradient-copy bridge pattern here is identical).

Scope: wraps just ``linear_scan_receiver_context`` itself, not the
surrounding ``_dense_stewardship_context`` orchestration (traffic
accumulation, cache-tuple bookkeeping) -- same "graph the expensive
kernel, leave the orchestration eager" split already used for node
compute/readout inside ``_dense_relay_transform``.

Requires ``block_causal`` mode (this backend's cache-carrying path,
``block_size`` not ``None``) -- the non-block-causal path doesn't take a
cache at all and isn't the shape this is built for. ``cache_forward_state``
is always a real zero tensor for the first call, never ``None`` --
CUDA-graph capture needs a fixed input structure; the caller is
responsible for constructing zeros on the very first relay/block rather
than passing ``None`` (unlike the underlying ``linear_scan_receiver_context``,
which accepts ``None`` and allocates zeros internally -- that branch
itself isn't graphable, since None-vs-tensor is a structural change).

Same DelayedScaling-cold-start-shaped caveat class doesn't apply here
(this is bf16, no quantization scale to warm up) -- but the general
"isolated-layer speedup shrinks when embedded among other un-graphed
model layers" caveat from the FP8 work applies identically, and hasn't
been re-measured in the full model yet at time of writing.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .models.memory import LinearScanReceiverCache, linear_scan_receiver_context


class _ReceiverMemoryBlock(nn.Module):
    """Holds the persistent ``relative_node_bias`` buffer CUDA-graph replay
    reads from -- see ``_BridgeReceiverMemoryBias`` for why a bridge is
    needed even though the model's real ``relative_node_bias`` is already
    a persistent ``nn.Parameter`` (it's still not the SAME memory address
    as this block's own copy, and gets a dtype cast on every real call
    that may or may not be a no-op depending on whether the model trains
    its master weights in a different dtype).
    """

    def __init__(
        self,
        cells: int,
        nodes: int,
        *,
        block_size: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.relative_node_bias = nn.Parameter(torch.zeros(cells, nodes, 4))
        self.block_size = block_size
        self.chunk_size = chunk_size

    def forward(
        self,
        states: torch.Tensor,
        node_locations: torch.Tensor,
        writer_mass: torch.Tensor,
        cache_forward_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, next_cache = linear_scan_receiver_context(
            states,
            node_locations,
            writer_mass,
            self.relative_node_bias,
            chunk_size=self.chunk_size,
            block_size=self.block_size,
            cache=LinearScanReceiverCache(cache_forward_state),
            return_cache=True,
        )
        return context, next_cache.forward_state


class _BridgeReceiverMemoryBias(torch.autograd.Function):
    """Route the live ``relative_node_bias``'s value/gradient through
    ``_ReceiverMemoryBlock``. Same mechanism as
    ``gator.mx_precision._BridgeFP8CellWeights`` -- see its docstring for
    the full reasoning (value-copy-in on forward, real dependency chain
    via passing the other inputs through unchanged so autograd guarantees
    this function's backward runs after the graphed module's own backward
    populates ``.grad``, gradient-copy-out on backward).
    """

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx, states, node_locations, writer_mass, cache_forward_state, bias, raw_module
    ):
        with torch.no_grad():
            raw_module.relative_node_bias.copy_(bias)
            raw_module.relative_node_bias.grad = None
        ctx.raw_module = raw_module
        return states, node_locations, writer_mass, cache_forward_state

    @staticmethod
    def backward(ctx, grad_states, grad_node_locations, grad_writer_mass, grad_cache):  # type: ignore[no-untyped-def]
        grad_bias = ctx.raw_module.relative_node_bias.grad.clone()
        return grad_states, grad_node_locations, grad_writer_mass, grad_cache, grad_bias, None


_RECEIVER_MEMORY_CUDAGRAPH_CACHE: dict[tuple, tuple[_ReceiverMemoryBlock, Callable]] = {}


def cuda_graph_linear_scan_receiver_context(
    states: torch.Tensor,
    node_locations: torch.Tensor,
    writer_mass: torch.Tensor,
    relative_node_bias: torch.Tensor,
    *,
    block_size: int,
    cache_forward_state: torch.Tensor,
    chunk_size: int = 64,
    graph_slot: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA-graph-captured ``linear_scan_receiver_context``, block-causal only.

    ``graph_slot`` -- **required to be distinct per call-site when this
    function is called more than once before a shared backward() fires**
    (this model's real usage: once per relay, sharing one combined
    backward across all relays). Found via direct test: a single graphed
    instance shared across two forward calls silently breaks the wrapped
    module's parameter gradient on the second-processed backward (the
    bridge's ``.grad`` read comes back ``None`` -- a crash here, but a
    silently wrong gradient in variants without the crash), a *different
    and more severe* problem than the output-buffer-aliasing bug already
    fixed with ``.clone()`` in this function. Verified fix: give each
    call-site its own dedicated graphed instance -- ``graph_slot`` is
    folded into the shape-cache key for exactly this. The caller (this
    model's per-relay loop) must pass a stable, distinct slot per
    call-site (the relay index) -- this function can't detect a missing
    or colliding slot itself.

    Returns ``(context, next_cache_forward_state)`` -- same shapes as
    ``linear_scan_receiver_context(..., block_size=block_size, cache=
    LinearScanReceiverCache(cache_forward_state), return_cache=True)``,
    just faster (dispatch-overhead elimination, not a precision change).

    Shape-cached per ``(batch, cells, tokens, hidden, nodes, block_size,
    chunk_size, device)`` -- first call for a given shape pays graph-
    capture warmup cost; later calls with the same shape reuse the
    captured graph. Batch size and context length must stay fixed across
    calls sharing a cache entry, same requirement as
    ``gator.mx_precision.te_fp8_cudagraph_node_linear``.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "cuda_graph_linear_scan_receiver_context requires a CUDA device"
        )
    if states.ndim != 4:
        raise ValueError("states must have [batch,cells,tokens,hidden] shape")
    batch, cells, tokens, hidden = states.shape
    nodes = node_locations.shape[-1]

    key = (
        batch, cells, tokens, hidden, nodes, block_size, chunk_size,
        states.device, graph_slot,
    )
    cached = _RECEIVER_MEMORY_CUDAGRAPH_CACHE.get(key)
    if cached is None:
        raw_module = _ReceiverMemoryBlock(
            cells, nodes, block_size=block_size, chunk_size=chunk_size
        ).to(device=states.device, dtype=torch.bfloat16)
        sample_states = torch.zeros(
            batch, cells, tokens, hidden,
            device=states.device, dtype=torch.bfloat16, requires_grad=True,
        )
        sample_locations = torch.zeros(
            batch, cells, tokens, nodes,
            device=states.device, dtype=torch.bfloat16, requires_grad=True,
        )
        sample_writer_mass = torch.zeros(
            batch, cells, tokens,
            device=states.device, dtype=torch.bfloat16, requires_grad=True,
        )
        rows = batch * (hidden + 1)
        channels = cells * nodes
        sample_cache = torch.zeros(
            rows, channels,
            device=states.device, dtype=torch.bfloat16, requires_grad=True,
        )
        # make_graphed_callables rejects capture from inside an active
        # autocast context ("does not support the autocast caching") --
        # real training always wraps the forward pass in
        # torch.autocast(bf16), which would otherwise crash capture on
        # the first real training step. Suppress it for just this
        # one-time capture; replay is a raw kernel replay, not affected.
        with torch.autocast(device_type="cuda", enabled=False):
            graphed_module = torch.cuda.make_graphed_callables(
                raw_module,
                (sample_states, sample_locations, sample_writer_mass, sample_cache),
                num_warmup_iters=5,
            )
        cached = (raw_module, graphed_module)
        _RECEIVER_MEMORY_CUDAGRAPH_CACHE[key] = cached
    raw_module, graphed_module = cached

    states_bf16 = states.to(torch.bfloat16)
    locations_bf16 = node_locations.to(torch.bfloat16)
    writer_mass_bf16 = writer_mass.to(torch.bfloat16)
    cache_bf16 = cache_forward_state.to(torch.bfloat16)
    bias_bf16 = relative_node_bias.to(torch.bfloat16)

    bridged_states, bridged_locations, bridged_writer_mass, bridged_cache = (
        _BridgeReceiverMemoryBias.apply(
            states_bf16, locations_bf16, writer_mass_bf16, cache_bf16,
            bias_bf16, raw_module,
        )
    )
    # .clone() required on BOTH outputs, not optional: make_graphed_callables'
    # outputs are live views into the CUDA graph's static output buffers,
    # silently overwritten by the NEXT call to this same graphed callable.
    # This bites hardest for next_cache_forward_state specifically, since
    # that tensor is *designed* to be held onto and fed into the following
    # relay's call -- without cloning, that next call's own graph replay
    # would corrupt the very cache value it was just handed, silently
    # producing wrong (not crashing) block-causal receiver memory.
    context, next_cache_forward_state = graphed_module(
        bridged_states, bridged_locations, bridged_writer_mass, bridged_cache
    )
    return (
        context.clone().to(states.dtype),
        next_cache_forward_state.clone().to(states.dtype),
    )
