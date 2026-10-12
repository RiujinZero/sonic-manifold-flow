"""Optional CUDA-graph-captured router/edge-logit computation.

**NOT PRODUCTION-SAFE -- NOT WIRED INTO GatorLanguageConfig/CLI.**
Unlike node compute, readout, receiver memory, broadcast-mix, and
stewardship traffic (all shipped and verified), this module has a
confirmed, unresolved gradient-corruption bug and is deliberately kept
disconnected from ``gator/models/language.py`` and
``gator_language_train.py``. Do not wire it back in without fixing
this first -- see ``tests/test_cuda_graph_router.py``'s
``test_multi_relay_shared_weights_gradients_not_corrupted`` (marked
``@unittest.expectedFailure``) for a live regression repro.

**The bug, precisely:** reusing a single cached ``graph_slot`` across
two SEPARATE forward+backward sessions (i.e. two different training
steps, or -- proven directly -- the same shape/slot called, backward()
called, then called again with fresh data and a fresh backward()) with
this module's real forward computation (``cell_linear``'s
``grouped_bmm`` backend WITH padding -- i.e.
``router_inputs``/``edges`` not already a multiple of
``pad_to_multiple``, production's actual shape -- feeding into
``F.softmax``) produces a captured backward graph whose second-and-
later replays compute a WRONG gradient for the bridged
``router_weight``/``router_bias`` (confirmed: forward output stays
correct every time; only the backward diverges, by 60-140% relative
norm error in direct measurement, not simple bf16 rounding). Node
compute/readout/receiver-memory/broadcast-mix/stewardship-traffic are
each unaffected -- confirmed directly for broadcast-mix and
stewardship-traffic via the identical two-session repro pattern (0.0
relative error) -- because none of them combine padding with softmax;
this looks like a genuine, narrow ``make_graphed_callables`` /
``cell_linear`` grouped-bmm-padding / softmax interaction, not a
mistake in this file's bridge pattern (the bridge pattern itself was
independently verified correct via ``num_warmup_iters`` sensitivity
tests, weight-identity-aliasing tests -- fixed, see the ``+ 0`` below
-- and explicit ``.grad`` zeroing tests, none of which alone or
together resolved it).

Two real, confirmed-necessary fixes ARE included below even though
they don't fix the core bug, since they fix genuinely separate issues
found along the way and should stay if this is ever revisited:
randn/rand (not zeros) warmup samples, and forcing ``weight_bf16``/
``bias_bf16`` to be genuinely new tensors (``+ 0``) rather than
aliasing the caller's parameter when it's already bf16 -- see their
inline comments for what each one independently fixes.

Not a precision feature (unlike gator/mx_precision.py) -- runs at
whatever dtype the caller already uses (bf16 in practice) and exists
purely to eliminate the per-call CPU dispatch overhead of the router
matmul / adjacency masking / softmax / transition einsum sequence
inside ``_dense_relay_transform`` (see gator/models/language.py), the
same overhead-elimination mechanism already shipped for node compute,
readout, receiver memory, broadcast-mix cell coupling, and stewardship
traffic bookkeeping. Read those docstrings first; the value-copy
bridge pattern for ``router_weight``/``router_bias`` here is the same
one used for broadcast-mix's ``left_projection``/``right_projection``
(genuine persistent shared parameters, not a freshly-derived-per-call
tensor like node compute's folded weight).

Precision note: this graphs the SAME bf16 computation already used for
routing today -- gator/mx_precision.py's module docstring documents an
explicit, separate scope decision that routing never goes through a
lower-precision format. This module doesn't touch that decision; it
targets dispatch overhead only, same as the other CUDA-graph modules
in this investigation.

Scope: wraps the ``router_input -> edge_logits -> transition -> q_next``
computation, independent of (not sequentially dependent on)
``_dense_relay_transform``'s node-compute sub-path -- ``q_next`` and
``destination`` are combined only in that function's final ``local =
...`` line, which stays outside this module.

``temperature`` is a scheduled/annealed Python float supplied by the
training loop (``route_temperature_start`` -> ``route_temperature_end``
interpolation in gator_language_train.py), not a fixed hyperparameter
-- baking it in as a Python constant at capture time would silently
freeze it at its first-call value for the rest of training, a real
correctness bug much subtler than a crash. It is threaded through as a
genuine 0-dim tensor input instead, exactly like
``h``/``context``/``occupancy``/``q``/``dense_address_edges``, so each
replay picks up the current step's scheduled value via
``make_graphed_callables``' input-copy-in mechanism.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F



class _RouterEdgeBlock(nn.Module):
    """Holds its own persistent bf16 copies of ``router_weight``/
    ``router_bias`` CUDA-graph replay reads from -- see
    ``_BridgeRouterEdgeWeights`` for why a bridge is needed even though
    the model's real router parameters are already persistent. The
    ``adjacency`` buffer is copied once at construction (fixed routing
    topology, never updated by the optimizer, so no bridge needed for
    it -- referenced directly like any ordinary non-trainable buffer).
    """

    def __init__(
        self,
        cells: int,
        edges: int,
        router_inputs: int,
        n: int,
        adjacency: torch.Tensor,
        backend: str,
        pad_to_multiple: int,
    ) -> None:
        super().__init__()
        self.router_weight = nn.Parameter(torch.empty(cells, edges, router_inputs))
        self.router_bias = nn.Parameter(torch.zeros(cells, edges))
        self.register_buffer("adjacency", adjacency.clone())
        self.n = n
        self.backend = backend
        self.pad_to_multiple = pad_to_multiple

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        occupancy: torch.Tensor,
        q: torch.Tensor,
        dense_address_edges: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        # Deferred: gator.models.__init__ imports gator.models.language,
        # which imports this module -- a module-level import of
        # cell_linear here would be a real circular import whenever this
        # module is the entry point (e.g. a test importing
        # gator.cuda_graph_router before anything has imported
        # gator.models). Importing lazily here breaks the cycle since by
        # the time forward() actually runs, gator.models is fully loaded.
        from .models.dense_kernels import cell_linear

        n = self.n
        batch, c, total_length = h.shape[0], h.shape[1], h.shape[2]
        router_input = torch.cat([h, context, occupancy], -1)
        edge_logits = cell_linear(
            router_input,
            self.router_weight,
            self.router_bias,
            backend=self.backend,
            pad_to_multiple=self.pad_to_multiple,
        )
        edge_logits = (edge_logits + dense_address_edges[None]).view(
            batch, c, total_length, n, n
        )
        adjacency = (
            self.adjacency[None, None, None]
            if self.adjacency.ndim == 2
            else self.adjacency[None, :, None]
        )
        transition = F.softmax(
            edge_logits.masked_fill(~adjacency, -1e9) / temperature,
            -1,
        )
        return torch.einsum("bcts,bctsd->bctd", q, transition)


class _BridgeRouterEdgeWeights(torch.autograd.Function):
    """Route the live router weight/bias's value/gradient through
    ``_RouterEdgeBlock``. Same mechanism as
    ``gator.cuda_graph_broadcast_mix._BridgeBroadcastMixWeights`` --
    see its docstring for the full reasoning (value-copy-in on forward,
    real dependency chain via passing every real input through
    unchanged so autograd guarantees this function's backward runs
    after the graphed module's own backward populates each parameter's
    ``.grad``, gradient-copy-out on backward). ``temperature`` is
    passed through too even though it never requires grad, for the
    same structural reason.
    """

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        h,
        context,
        occupancy,
        q,
        dense_address_edges,
        temperature,
        weight,
        bias,
        raw_module,
    ):
        with torch.no_grad():
            raw_module.router_weight.copy_(weight)
            if raw_module.router_weight.grad is not None:
                raw_module.router_weight.grad.zero_()
            raw_module.router_bias.copy_(bias)
            if raw_module.router_bias.grad is not None:
                raw_module.router_bias.grad.zero_()
        ctx.raw_module = raw_module
        return h, context, occupancy, q, dense_address_edges, temperature

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]
        ctx, grad_h, grad_context, grad_occupancy, grad_q, grad_edges, grad_temperature
    ):
        raw_module = ctx.raw_module
        grad_weight = raw_module.router_weight.grad.clone()
        grad_bias = raw_module.router_bias.grad.clone()
        return (
            grad_h,
            grad_context,
            grad_occupancy,
            grad_q,
            grad_edges,
            grad_temperature,
            grad_weight,
            grad_bias,
            None,
        )


_ROUTER_CUDAGRAPH_CACHE: dict[tuple, tuple[_RouterEdgeBlock, Callable]] = {}


def cuda_graph_router_edges(
    h: torch.Tensor,
    context: torch.Tensor,
    occupancy: torch.Tensor,
    q: torch.Tensor,
    dense_address_edges: torch.Tensor,
    temperature: float,
    *,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    adjacency: torch.Tensor,
    n: int,
    backend: str,
    pad_to_multiple: int,
    graph_slot: int = 0,
) -> torch.Tensor:
    """CUDA-graph-captured router/edge-logit computation.

    ``h``/``context``/``occupancy``: ``[batch, cells, tokens, *]``.
    ``q``: ``[batch, cells, tokens, n]``. ``dense_address_edges``:
    ``[cells, tokens, n*n]``. Returns ``q_next``: ``[batch, cells,
    tokens, n]`` -- identical shape/semantics to the eager
    ``_dense_relay_transform`` router sub-path, just faster.

    ``graph_slot`` -- **required to be distinct per call-site when this
    function is called more than once before a shared backward() fires**
    (this model's real usage: once per relay). See
    ``gator.mx_precision.te_fp8_cudagraph_node_linear``'s docstring for
    the direct-test evidence this caveat is based on.

    Shape-cached per ``(batch, cells, tokens, edges, router_inputs, n,
    device, graph_slot)`` -- the first call for a given shape pays
    graph-capture warmup cost; later calls with the same shape reuse
    the captured graph.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("cuda_graph_router_edges requires a CUDA device")
    if h.ndim != 4:
        raise ValueError("cuda_graph_router_edges h must be [batch,cells,tokens,d_model]")
    batch, cells, tokens, _ = h.shape
    edges = router_weight.shape[1]
    router_inputs = router_weight.shape[2]

    key = (batch, cells, tokens, edges, router_inputs, n, h.device, graph_slot)
    cached = _ROUTER_CUDAGRAPH_CACHE.get(key)
    if cached is None:
        raw_module = _RouterEdgeBlock(
            cells, edges, router_inputs, n, adjacency, backend, pad_to_multiple,
        ).to(device=h.device, dtype=torch.bfloat16)
        # randn, not zeros: an all-zero warmup/capture input makes
        # edge_logits uniform across the softmax dimension (a degenerate
        # all-tied case) whenever this router's router_inputs isn't
        # already a multiple of pad_to_multiple (production's real
        # shape) -- combined with cell_linear's grouped_bmm padding
        # path, capturing against that degenerate input produces a
        # captured backward graph that silently computes WRONG
        # gradients once real (non-degenerate) values flow through on
        # later replays, while forward output stays numerically fine.
        # Verified directly via a minimal repro: identical code with
        # zero-valued samples corrupts router_weight.grad by ~65-140%
        # relative error on the second use of a cached graph_slot;
        # switching every sample to randn/rand (still shape/dtype-only,
        # values are never semantically meaningful for warmup) makes
        # the error exactly 0. Not needed for node compute/readout/
        # receiver-memory/broadcast-mix/stewardship-traffic -- none of
        # those combine padding with softmax, the specific combination
        # that triggers this.
        sample_h = torch.randn(
            batch, cells, tokens, h.shape[-1],
            device=h.device, dtype=torch.bfloat16, requires_grad=True,
        )
        sample_context = torch.randn(
            batch, cells, tokens, context.shape[-1],
            device=h.device, dtype=torch.bfloat16, requires_grad=True,
        )
        sample_occupancy = torch.randn(
            batch, cells, tokens, occupancy.shape[-1],
            device=h.device, dtype=torch.bfloat16, requires_grad=True,
        )
        sample_q = torch.rand(
            batch, cells, tokens, n,
            device=h.device, dtype=torch.bfloat16, requires_grad=True,
        )
        sample_edges = torch.randn(
            cells, tokens, edges,
            device=h.device, dtype=torch.bfloat16, requires_grad=True,
        )
        # requires_grad=True even though nothing reads its .grad --
        # make_graphed_callables only sets up its per-call
        # copy-new-value-into-the-static-buffer mechanism for inputs
        # that require grad; a requires_grad=False sample gets baked in
        # as a capture-time constant instead, silently freezing
        # temperature at whatever value warmup/capture happened to use.
        # Verified directly: without this, a later call at a different
        # temperature produced the SAME output as the first call.
        sample_temperature = torch.tensor(
            0.35, device=h.device, dtype=torch.float32, requires_grad=True
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
                    sample_h, sample_context, sample_occupancy, sample_q,
                    sample_edges, sample_temperature,
                ),
                num_warmup_iters=5,
            )
        cached = (raw_module, graphed_module)
        _ROUTER_CUDAGRAPH_CACHE[key] = cached
    raw_module, graphed_module = cached

    h_bf16 = h.to(torch.bfloat16)
    context_bf16 = context.to(torch.bfloat16)
    occupancy_bf16 = occupancy.to(torch.bfloat16)
    q_bf16 = q.to(torch.bfloat16)
    edges_bf16 = dense_address_edges.to(torch.bfloat16)
    # requires_grad=True for the same reason as sample_temperature above
    # -- required for make_graphed_callables to copy this call's real
    # value into the graph's static input buffer before replay.
    temperature_tensor = torch.tensor(
        float(temperature), device=h.device, dtype=torch.float32, requires_grad=True
    )
    # `+ 0`, not just `.to(bfloat16)`: when router_weight/router_bias
    # are ALREADY bf16 (production's real case, since master weights
    # here aren't kept in a separate higher-precision dtype the way
    # some training setups do), `.to(bfloat16)` is a no-op returning
    # the SAME tensor object, not a copy -- meaning the exact same
    # leaf parameter gets passed into _BridgeRouterEdgeWeights.apply()
    # across every call sharing this graph_slot. Confirmed via direct
    # instrumentation this was one real contributor to the gradient
    # corruption documented in this module's docstring (fixing it
    # alone was not sufficient to fully resolve the bug, but it is a
    # genuine, independently-necessary fix).
    weight_bf16 = router_weight.to(torch.bfloat16) + 0
    bias_bf16 = router_bias.to(torch.bfloat16) + 0

    (
        bridged_h, bridged_context, bridged_occupancy, bridged_q,
        bridged_edges, bridged_temperature,
    ) = _BridgeRouterEdgeWeights.apply(
        h_bf16, context_bf16, occupancy_bf16, q_bf16, edges_bf16, temperature_tensor,
        weight_bf16, bias_bf16, raw_module,
    )
    # .clone() required, not optional -- make_graphed_callables' output is a
    # live view into the CUDA graph's static output buffer, silently
    # overwritten by the NEXT call to this same graphed callable (e.g. the
    # next relay's router call in the same forward pass, before this
    # relay's activation has been consumed by backward()).
    q_next = graphed_module(
        bridged_h, bridged_context, bridged_occupancy, bridged_q,
        bridged_edges, bridged_temperature,
    ).clone()
    return q_next.to(q.dtype)
