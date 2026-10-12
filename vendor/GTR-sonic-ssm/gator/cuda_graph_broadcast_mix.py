"""Optional CUDA-graph-captured broadcast-mix cell coupling.

Not a precision feature (unlike gator/mx_precision.py) -- runs at
whatever dtype the caller already uses (bf16 in practice) and exists
purely to eliminate the per-call CPU dispatch overhead of the several
small roll/stack/gate/project/norm ops in GTR-ARW's broadcast_mix
cell-transport ring-topology message passing (see
gator/models/language.py's per-relay loop, the ``if ownership is None:
... continue`` branch) -- the same overhead-elimination mechanism
already shipped for node compute and readout (gator/mx_precision.py's
``te_fp8_cudagraph_node_linear``/``te_fp8_cudagraph_readout_linear``)
and the linear_scan receiver memory
(gator/cuda_graph_receiver_memory.py). Read those docstrings first; the
value-copy/gradient-copy bridge pattern here is identical.

Scope: wraps just the roll/stack/gate/project/norm computation that
turns a relay's ``local`` into the next relay's ``h`` -- not the
upstream ``_dense_relay_transform`` call that produces ``local`` in the
first place (already separately graphed via
``te_fp8_cudagraph_node_linear`` when ``node_compute_precision=fp8_te``
is selected).

Only applies to ``cell_transport_mode='broadcast_mix'`` -- the
``ownership is not None`` stewardship-transport branch is a
structurally different code path (and, per the dense/packed
stewardship context's data-dependent tensor shapes, not CUDA-graph-
capturable the same way) not touched here.

Unlike the receiver-memory bias, the coupling parameters here
(``left_projection``/``right_projection``/``coupling_norm``/
``neighbor_gates``/``self_gates``) are genuine persistent model
parameters used directly at their call site, not values derived fresh
each forward pass -- but a bridge is still required, for the same
reason ``te_fp8_cudagraph_readout_linear`` needs one despite
``token_embedding.weight`` also being a persistent parameter: this
block needs its own bf16 copies so the captured graph replays at bf16
like the rest of training under autocast, rather than at the model's
fp32 master-weight precision (capture itself must run with autocast
disabled, so referencing the real fp32 parameters directly would
silently capture fp32 compute -- wrong dtype, not just slower).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class _BroadcastMixBlock(nn.Module):
    """Holds its own persistent bf16 copies of the coupling parameters
    CUDA-graph replay reads from -- see ``_BridgeBroadcastMixWeights``
    for why a bridge is needed even though the model's real parameters
    are already persistent ``nn.Parameter``/module weights.
    """

    def __init__(self, cells: int, d_model: int, radius: int) -> None:
        super().__init__()
        self.left_projection = nn.Linear(d_model, d_model)
        self.right_projection = nn.Linear(d_model, d_model)
        self.coupling_norm = nn.LayerNorm(d_model)
        self.neighbor_gates = nn.Parameter(torch.zeros(cells, 2))
        self.self_gates = nn.Parameter(torch.ones(cells))
        self.radius = radius

    def forward(self, local: torch.Tensor) -> torch.Tensor:
        left = torch.stack(
            [
                torch.roll(local, distance, 1)
                for distance in range(1, self.radius + 1)
            ]
        ).mean(0)
        right = torch.stack(
            [
                torch.roll(local, -distance, 1)
                for distance in range(1, self.radius + 1)
            ]
        ).mean(0)
        gates = torch.sigmoid(self.neighbor_gates)
        mixed = self.self_gates[None, :, None, None] * local
        mixed = mixed + gates[None, :, 0, None, None] * self.left_projection(left)
        mixed = mixed + gates[None, :, 1, None, None] * self.right_projection(right)
        return self.coupling_norm(mixed)


class _BridgeBroadcastMixWeights(torch.autograd.Function):
    """Route the live coupling parameters' values/gradients through
    ``_BroadcastMixBlock``. Same mechanism as
    ``gator.mx_precision._BridgeFP8CellWeights`` -- see its docstring
    for the full reasoning (value-copy-in on forward, real dependency
    chain via passing ``local`` through unchanged so autograd guarantees
    this function's backward runs after the graphed module's own
    backward populates each parameter's ``.grad``, gradient-copy-out on
    backward).
    """

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        local,
        left_weight,
        left_bias,
        right_weight,
        right_bias,
        norm_weight,
        norm_bias,
        neighbor_gates,
        self_gates,
        raw_module,
    ):
        with torch.no_grad():
            raw_module.left_projection.weight.copy_(left_weight)
            raw_module.left_projection.weight.grad = None
            raw_module.left_projection.bias.copy_(left_bias)
            raw_module.left_projection.bias.grad = None
            raw_module.right_projection.weight.copy_(right_weight)
            raw_module.right_projection.weight.grad = None
            raw_module.right_projection.bias.copy_(right_bias)
            raw_module.right_projection.bias.grad = None
            raw_module.coupling_norm.weight.copy_(norm_weight)
            raw_module.coupling_norm.weight.grad = None
            raw_module.coupling_norm.bias.copy_(norm_bias)
            raw_module.coupling_norm.bias.grad = None
            raw_module.neighbor_gates.copy_(neighbor_gates)
            raw_module.neighbor_gates.grad = None
            raw_module.self_gates.copy_(self_gates)
            raw_module.self_gates.grad = None
        ctx.raw_module = raw_module
        return local

    @staticmethod
    def backward(ctx, grad_local):  # type: ignore[no-untyped-def]
        raw_module = ctx.raw_module
        return (
            grad_local,
            raw_module.left_projection.weight.grad.clone(),
            raw_module.left_projection.bias.grad.clone(),
            raw_module.right_projection.weight.grad.clone(),
            raw_module.right_projection.bias.grad.clone(),
            raw_module.coupling_norm.weight.grad.clone(),
            raw_module.coupling_norm.bias.grad.clone(),
            raw_module.neighbor_gates.grad.clone(),
            raw_module.self_gates.grad.clone(),
            None,
        )


_BROADCAST_MIX_CUDAGRAPH_CACHE: dict[tuple, tuple[_BroadcastMixBlock, Callable]] = {}


def cuda_graph_broadcast_mix(
    local: torch.Tensor,
    *,
    left_projection: nn.Linear,
    right_projection: nn.Linear,
    coupling_norm: nn.LayerNorm,
    neighbor_gates: torch.Tensor,
    self_gates: torch.Tensor,
    radius: int,
    graph_slot: int = 0,
) -> torch.Tensor:
    """CUDA-graph-captured broadcast-mix cell coupling.

    ``local``: ``[batch, cells, tokens, d_model]``, the per-cell output
    of this relay's ``_dense_relay_transform``. Returns the next
    relay's ``h``, same shape.

    ``graph_slot`` -- **required to be distinct per call-site when this
    function is called more than once before a shared backward() fires**
    (this model's real usage: once per relay, sharing one combined
    backward across all relays, exactly the pattern documented in
    ``gator.mx_precision.te_fp8_cudagraph_node_linear`` and
    ``gator.cuda_graph_receiver_memory.cuda_graph_linear_scan_receiver_
    context`` -- see either docstring for the direct-test evidence: a
    single graphed instance shared across two forward calls silently
    corrupts backward's gradient). The caller (this model's per-relay
    loop) must pass a stable, distinct slot per call-site (the relay
    index).

    Shape-cached per ``(batch, cells, tokens, d_model, radius, device,
    graph_slot)`` -- the first call for a given shape pays graph-capture
    warmup cost; later calls with the same shape reuse the captured
    graph. Batch size and context length must stay fixed across calls
    sharing a cache entry, same requirement as any CUDA-graph-captured
    code in this codebase.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("cuda_graph_broadcast_mix requires a CUDA device")
    if local.ndim != 4:
        raise ValueError(
            "cuda_graph_broadcast_mix local must be [batch,cells,tokens,d_model]"
        )
    if radius <= 0:
        raise ValueError("cuda_graph_broadcast_mix radius must be positive")
    batch, cells, tokens, d_model = local.shape

    key = (batch, cells, tokens, d_model, radius, local.device, graph_slot)
    cached = _BROADCAST_MIX_CUDAGRAPH_CACHE.get(key)
    if cached is None:
        raw_module = _BroadcastMixBlock(cells, d_model, radius).to(
            device=local.device, dtype=torch.bfloat16
        )
        sample = torch.zeros(
            batch, cells, tokens, d_model,
            device=local.device, dtype=torch.bfloat16, requires_grad=True,
        )
        # make_graphed_callables rejects capture from inside an active
        # autocast context ("does not support the autocast caching") --
        # real training always wraps forward in torch.autocast(bf16),
        # which would otherwise crash capture on the first real training
        # step. Suppress it for just this one-time capture; replay is a
        # raw kernel replay, not affected.
        with torch.autocast(device_type="cuda", enabled=False):
            graphed_module = torch.cuda.make_graphed_callables(
                raw_module, (sample,), num_warmup_iters=5
            )
        cached = (raw_module, graphed_module)
        _BROADCAST_MIX_CUDAGRAPH_CACHE[key] = cached
    raw_module, graphed_module = cached

    local_bf16 = local.to(torch.bfloat16)
    bridged_local = _BridgeBroadcastMixWeights.apply(
        local_bf16,
        left_projection.weight.to(torch.bfloat16),
        left_projection.bias.to(torch.bfloat16),
        right_projection.weight.to(torch.bfloat16),
        right_projection.bias.to(torch.bfloat16),
        coupling_norm.weight.to(torch.bfloat16),
        coupling_norm.bias.to(torch.bfloat16),
        neighbor_gates.to(torch.bfloat16),
        self_gates.to(torch.bfloat16),
        raw_module,
    )
    # .clone() required, not optional -- make_graphed_callables' output is a
    # live view into the CUDA graph's static output buffer, silently
    # overwritten by the NEXT call to this same graphed callable (e.g. the
    # next relay's broadcast-mix call in the same forward pass, before this
    # relay's activation has been consumed by backward()).
    output = graphed_module(bridged_local).clone()
    return output.to(local.dtype)
