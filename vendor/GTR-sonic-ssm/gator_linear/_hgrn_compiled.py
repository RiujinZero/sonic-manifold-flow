"""torch.compile-friendly reimplementation of flash-linear-attention's
``chunk_hgrn``.

``fla.ops.hgrn.chunk.chunk_hgrn`` is decorated ``@torch.compiler.disable``,
which forces a Dynamo graph break on every call. In GTR-ARW's receiver
memory this kernel is invoked up to ~32x per optimizer update (2 relays x
16 gradient-accumulation steps), and the repeated break/re-enter tax ate
the kernel's isolated ~5.5x speedup in a real 300-step production A/B --
see Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md.

This module exposes the exact same two Triton kernel launches (forward:
``chunk_hgrn_fwd_kernel_h`` + ``chunk_hgrn_fwd_kernel_o``; backward:
``chunk_hgrn_bwd_kernel_h`` + ``chunk_hgrn_bwd_kernel_o``, read directly
from FLA's own ``ChunkHGRNFunction`` source) wrapped as
``torch.library.triton_op``s with an explicit ``register_autograd``
backward, so torch.compile can capture the whole thing as one opaque
graph node instead of breaking the graph.

Gradient semantics intentionally match stock ``chunk_hgrn`` exactly,
including its existing limitation that no gradient flows back through
``initial_state`` (FLA's own backward returns ``None`` for that slot) --
this module is a drop-in replacement for the kernel calls, not a
behavior change.

One correctness-critical difference from a naive port: FLA's
``ChunkHGRNFunction.forward``/``.backward`` are wrapped in an
``@input_guard`` decorator that calls ``.contiguous()`` on every tensor
argument before the kernel ever runs. ``register_autograd`` has no such
guard. In particular, autograd may hand backward an EXPANDED, stride-0
gradient (e.g. from `y.sum().backward()`, which conceptually means
"every element is 1" without materializing memory) -- the backward
kernel does flat pointer arithmetic assuming real per-element strides,
so a stride-0 `do` silently produces NaN. Confirmed via direct
before/after test (Runbooks/colab_cells/_hgrn_compile_friendly.py):
identical kernel-calling code produces bit-for-bit correct gradients
when called directly with a materialized `do`, and NaN when invoked
through the real register_autograd dispatch path with an unmaterialized
one. Fix is the explicit `do.contiguous()` call in `_backward` below.
"""

from typing import Optional

import torch
from torch import Tensor

try:
    import triton

    from fla.ops.hgrn.chunk import (
        chunk_hgrn_bwd_kernel_h,
        chunk_hgrn_bwd_kernel_o,
        chunk_hgrn_fwd_kernel_h,
        chunk_hgrn_fwd_kernel_o,
    )

    _IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - exercised only without fla/triton
    triton = None  # type: ignore[assignment]
    _IMPORT_ERROR = error


if triton is not None:

    @torch.library.triton_op("gator::chunk_hgrn_fwd", mutates_args={})
    def _chunk_hgrn_fwd_op(
        x: Tensor,
        g: Tensor,
        initial_state: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        B, T, D = x.shape
        BT, BD = 128, min(64, triton.next_power_of_2(D))
        num_warps = 8 if BD == 64 else 4

        gc = torch.empty_like(g, dtype=torch.float)
        o = torch.empty_like(x, dtype=torch.float)

        def grid_h(meta):
            return (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)

        torch.library.wrap_triton(chunk_hgrn_fwd_kernel_h)[grid_h](
            x, g, gc, o, initial_state,
            T=T, D=D, BT=BT,
            USE_INITIAL_STATE=initial_state is not None,
        )

        def grid_o(meta):
            return (triton.cdiv(D, meta["BD"]), B)

        torch.library.wrap_triton(chunk_hgrn_fwd_kernel_o)[grid_o](
            gc, o,
            o.stride(-3), o.stride(-2), o.stride(-1),
            T=T, D=D, BT=BT, BD=BD,
            num_warps=num_warps,
        )
        final_state = o[:, -1].clone()
        o = o.to(x.dtype)
        return o, final_state

    @_chunk_hgrn_fwd_op.register_fake
    def _(x, g, initial_state):
        B, T, D = x.shape
        return torch.empty_like(x), torch.empty(B, D, device=x.device, dtype=x.dtype)

    @torch.library.triton_op("gator::chunk_hgrn_bwd", mutates_args={})
    def _chunk_hgrn_bwd_op(
        g: Tensor,
        o: Tensor,
        do: Tensor,
        initial_state: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        B, T, D = do.shape
        BT, BD = 128, min(64, triton.next_power_of_2(D))
        num_warps = 8 if BD == 64 else 4

        gc = torch.empty_like(g, dtype=torch.float)
        dx = torch.empty_like(o, dtype=torch.float)

        def grid_h(meta):
            return (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)

        torch.library.wrap_triton(chunk_hgrn_bwd_kernel_h)[grid_h](
            g, gc, dx, do, T=T, D=D, BT=BT,
        )

        dg = torch.empty_like(g, dtype=torch.float)

        def grid_o(meta):
            return (triton.cdiv(D, meta["BD"]), B)

        torch.library.wrap_triton(chunk_hgrn_bwd_kernel_o)[grid_o](
            g, gc, o, dx, dg,
            o.stride(-3), o.stride(-2), o.stride(-1),
            T=T, D=D, BT=BT, BD=BD,
            num_warps=num_warps,
        )
        if initial_state is not None:
            dg = dg.clone()
            dg[:, 0] = (initial_state * dx[:, 0] * g[:, 0].float().exp()).to(dg.dtype)
        return dx.to(o.dtype), dg

    @_chunk_hgrn_bwd_op.register_fake
    def _(g, o, do, initial_state):
        return torch.empty_like(o), torch.empty_like(g)

    def _setup_context(ctx, inputs, output):
        x, g, initial_state = inputs
        o, final_state = output
        ctx.save_for_backward(g, o, initial_state)

    def _backward(ctx, do, dfinal_state):
        # dfinal_state is intentionally unused: stock chunk_hgrn's own
        # backward ignores it too (returns None for the initial_state
        # gradient slot), so this stays an exact drop-in match rather
        # than a behavior change.
        g, o, initial_state = ctx.saved_tensors
        do = do.contiguous()
        dx, dg = _chunk_hgrn_bwd_op(g, o, do, initial_state)
        return dx, dg, None

    torch.library.register_autograd(
        "gator::chunk_hgrn_fwd", _backward, setup_context=_setup_context,
    )


def compiled_chunk_hgrn(
    x: Tensor,
    g: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    """Drop-in replacement for ``fla.ops.hgrn.chunk_hgrn`` with the same
    signature, compile-friendly (no ``@torch.compiler.disable`` graph
    break) via ``torch.library.triton_op`` + ``register_autograd``.
    """
    if triton is None:
        raise RuntimeError(
            "compiled_chunk_hgrn requires flash-linear-attention and triton; "
            f"import failed with: {_IMPORT_ERROR!r}"
        )
    x = x.contiguous()
    g = g.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    o, final_state = _chunk_hgrn_fwd_op(x, g, initial_state)
    if not output_final_state:
        final_state = None
    return o, final_state
