"""Phase 2: specialized fixed-decay HGRN kernel.

gator_linear._hgrn_compiled.compiled_chunk_hgrn (Phase 1.5) is a
compile-friendly port of FLA's *general* chunk_hgrn, whose gate `g` is
an arbitrary [B, T, D] tensor. This module's use of it is always the
much narrower case where the gate is a single [D]-shaped `log_decay`
broadcast constant across every batch and time position (`decay` is
learned but time-invariant) -- see chunk_hgrn_decayed_scan's docstring
in gator_linear/core.py for the RetNet/HGRN equivalence this exploits.

The general kernel doesn't know that, so it pays real memory traffic
for it that a specialized kernel doesn't have to:

  - `g` is materialized as a full [B, T, D] tensor by the caller
    (`decay.log()[None, None, :].expand(...).contiguous()`) purely to
    satisfy chunk_hgrn's API -- every position holds an identical
    per-channel value.
  - The forward kernel writes, and a second forward kernel reads, a
    full [B, T, D] `gc` (cumulative log-decay within each chunk) --
    but for a constant gate this has the closed form
    `gc[b, t, d] = log_decay[d] * local_offset(t)`, needing no memory
    traffic at all.
  - The backward kernel produces a full [B, T, D] `dg`, which the
    caller's autograd then reduces (via the `.expand()` backward) down
    to the [D]-shaped gradient decay actually needs -- materializing
    and reducing a tensor that could instead be accumulated directly.

This module forks the four kernels (read from FLA's
fla.ops.hgrn.chunk source) to take `log_decay: [D]` instead of
`g: [B, T, D]`, drop `gc` entirely (recomputed analytically), and
accumulate the decay gradient directly via a same-dtype float32
[D]-shaped buffer with `tl.atomic_add` across the batch-parallel
kernel_o grid, instead of ever materializing a [B, T, D] `dg`.

Same compile-friendliness approach as Phase 1.5
(torch.library.triton_op + register_autograd, `do.contiguous()` guard
against autograd's expanded stride-0 gradients) and the same
correctness bar: verified to match stock chunk_hgrn (called with a
broadcast-expanded gate) exactly, including the `initial_state`
position-0 gradient correction FLA's own ChunkHGRNFunction.backward
applies post-hoc.
"""

from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    from fla.ops.utils.op import exp
    from fla.utils import autotune_cache_kwargs

    _IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - exercised only without fla/triton
    triton = None  # type: ignore[assignment]
    _IMPORT_ERROR = error


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({'BD': BD}, num_warps=num_warps)
            for BD in [32, 64, 128]
            for num_warps in [1, 2, 4, 8]
        ],
        key=['D'],
        **autotune_cache_kwargs,
    )
    @triton.jit(do_not_specialize=['T'])
    def _fixed_decay_fwd_kernel_h(
        x, log_decay, o, h0,
        T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
    ):
        i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        o_d = i_d * BD + tl.arange(0, BD)
        mask = o_d < D

        p_x = x + i_b * T * D + i_t * BT * D + o_d
        p_o = o + i_b * T * D + i_t * BT * D + o_d

        b_decay = exp(tl.load(log_decay + o_d, mask=mask, other=0.0).to(tl.float32))

        b_h = tl.zeros([BD], dtype=tl.float32)
        if USE_INITIAL_STATE:
            if i_t == 0:
                b_h += tl.load(h0 + i_b * D + o_d, mask=mask, other=0).to(tl.float32)
        for i in range(0, BT):
            mask_t = mask & ((i_t * BT + i) < T)
            b_x = tl.load(p_x, mask=mask_t, other=0).to(tl.float32)
            b_h = b_decay * b_h + b_x
            tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask_t)
            p_x += D
            p_o += D

    @triton.jit(do_not_specialize=['T'])
    def _fixed_decay_fwd_kernel_o(
        log_decay, o,
        s_b, s_t, s_d,
        T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    ):
        i_d, i_b = tl.program_id(0), tl.program_id(1)
        o_d = i_d * BD + tl.arange(0, BD)
        mask = o_d < D
        b_g = tl.load(log_decay + o_d, mask=mask, other=0.0).to(tl.float32)
        local_pos = (tl.arange(0, BT) + 1).to(tl.float32)

        for i_t in range(1, tl.cdiv(T, BT)):
            o_t = (i_t * BT).to(tl.int64) + tl.arange(0, BT)
            m_g = (o_t[:, None] < T) & mask[None, :]
            p_o = o + i_b * s_b + o_t[:, None] * s_t + o_d[None, :] * s_d

            b_h0 = tl.load(o + i_b * T * D + i_t * BT * D - D + o_d, mask=mask, other=0).to(tl.float32)
            b_gc = b_g[None, :] * local_pos[:, None]
            b_o = tl.load(p_o, mask=m_g, other=0.0).to(tl.float32)
            b_o = b_o + exp(b_gc) * b_h0[None, :]
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_g)

    @triton.autotune(
        configs=[
            triton.Config({'BD': BD}, num_warps=num_warps)
            for BD in [32, 64, 128]
            for num_warps in [1, 2, 4, 8]
        ],
        key=['D'],
        **autotune_cache_kwargs,
    )
    @triton.jit(do_not_specialize=['T'])
    def _fixed_decay_bwd_kernel_h(
        log_decay, dx, do,
        T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    ):
        i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        o_d = i_d * BD + tl.arange(0, BD)
        mask = o_d < D
        BC = min(BT, T - i_t * BT)

        p_dx = dx + (i_b * T + i_t * BT + BC - 1) * D + o_d
        p_do = do + (i_b * T + i_t * BT + BC - 1) * D + o_d

        b_decay = exp(tl.load(log_decay + o_d, mask=mask, other=0.0).to(tl.float32))
        b_dh = tl.zeros([BD], dtype=tl.float32)
        for _ in range(BC - 1, -1, -1):
            b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
            b_dh = b_dh + b_do
            b_dx = b_dh
            b_dh = b_dh * b_decay
            tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
            p_dx -= D
            p_do -= D

    @triton.jit(do_not_specialize=['T'])
    def _fixed_decay_bwd_kernel_o(
        log_decay, o, dx, d_log_decay,
        s_b, s_t, s_d,
        T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    ):
        i_d, i_b = tl.program_id(0), tl.program_id(1)
        o_d = i_d * BD + tl.arange(0, BD)
        mask = o_d < D
        b_g = tl.load(log_decay + o_d, mask=mask, other=0.0).to(tl.float32)
        b_decay = exp(b_g)
        # Backward's `gc` is a SUFFIX cumulative sum (built walking t
        # from the block's last position down to its first, storing
        # BEFORE accumulating each step -- see chunk_hgrn_bwd_kernel_h),
        # unlike forward's prefix gc. For a full (non-final) block of
        # length BT this closes to log_decay * (BT - local_offset),
        # counting down from BT to 1 -- NOT local_offset + 1 like
        # forward's. Only non-final blocks reach here with a nonzero
        # b_ht carry-in (the final block's is masked to 0), and every
        # non-final block has BC == BT, so this is exact.
        local_pos = (BT - tl.arange(0, BT)).to(tl.float32)

        b_dg_acc = tl.zeros([BD], dtype=tl.float32)

        for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
            o_t = (i_t * BT).to(tl.int64) + tl.arange(0, BT)
            o_to = (i_t * BT - 1).to(tl.int64) + tl.arange(0, BT)
            m_g = (o_t[:, None] < T) & mask[None, :]
            m_go = ((o_to >= 0) & (o_to < T))[:, None] & mask[None, :]
            p_o = o + i_b * s_b + o_to[:, None] * s_t + o_d[None, :] * s_d
            p_dx = dx + i_b * s_b + o_t[:, None] * s_t + o_d[None, :] * s_d

            mask_t = mask & ((i_t + 1) * BT < T)
            b_ht = tl.load(dx + i_b * T * D + (i_t + 1) * BT * D + o_d, mask=mask_t, other=0).to(tl.float32)

            b_gc = b_g[None, :] * local_pos[:, None]
            b_o = tl.load(p_o, mask=m_go, other=0.0).to(tl.float32)
            b_dx = tl.load(p_dx, mask=m_g, other=0.0).to(tl.float32)

            b_dx = b_dx + exp(b_gc) * b_ht[None, :]
            b_dg = tl.where(m_g, b_o * b_dx * b_decay[None, :], 0.0)
            tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=m_g)

            b_dg_acc += tl.sum(b_dg, axis=0)

        tl.atomic_add(d_log_decay + o_d, b_dg_acc, mask=mask)

    @torch.library.triton_op("gator::fixed_decay_hgrn_fwd", mutates_args={})
    def _fixed_decay_fwd_op(
        x: Tensor,
        log_decay: Tensor,
        initial_state: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        B, T, D = x.shape
        BT, BD = 128, min(64, triton.next_power_of_2(D))
        num_warps = 8 if BD == 64 else 4

        o = torch.empty_like(x, dtype=torch.float)

        def grid_h(meta):
            return (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)

        torch.library.wrap_triton(_fixed_decay_fwd_kernel_h)[grid_h](
            x, log_decay, o, initial_state,
            T=T, D=D, BT=BT,
            USE_INITIAL_STATE=initial_state is not None,
        )

        def grid_o(meta):
            return (triton.cdiv(D, meta["BD"]), B)

        torch.library.wrap_triton(_fixed_decay_fwd_kernel_o)[grid_o](
            log_decay, o,
            o.stride(-3), o.stride(-2), o.stride(-1),
            T=T, D=D, BT=BT, BD=BD,
            num_warps=num_warps,
        )
        final_state = o[:, -1].clone()
        o = o.to(x.dtype)
        return o, final_state

    @_fixed_decay_fwd_op.register_fake
    def _(x, log_decay, initial_state):
        B, T, D = x.shape
        return torch.empty_like(x), torch.empty(B, D, device=x.device, dtype=x.dtype)

    @torch.library.triton_op("gator::fixed_decay_hgrn_bwd", mutates_args={})
    def _fixed_decay_bwd_op(
        log_decay: Tensor,
        o: Tensor,
        do: Tensor,
        initial_state: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        B, T, D = do.shape
        BT, BD = 128, min(64, triton.next_power_of_2(D))
        num_warps = 8 if BD == 64 else 4

        dx = torch.empty_like(o, dtype=torch.float)

        def grid_h(meta):
            return (triton.cdiv(D, meta["BD"]), triton.cdiv(T, meta["BT"]), B)

        torch.library.wrap_triton(_fixed_decay_bwd_kernel_h)[grid_h](
            log_decay, dx, do, T=T, D=D, BT=BT,
        )

        d_log_decay = torch.zeros(D, device=do.device, dtype=torch.float)

        def grid_o(meta):
            return (triton.cdiv(D, meta["BD"]), B)

        torch.library.wrap_triton(_fixed_decay_bwd_kernel_o)[grid_o](
            log_decay, o, dx, d_log_decay,
            o.stride(-3), o.stride(-2), o.stride(-1),
            T=T, D=D, BT=BT, BD=BD,
            num_warps=num_warps,
        )
        if initial_state is not None:
            # The kernel's position-0 `b_o` load is masked to 0 (there's
            # no o[-1] before the sequence start), so its contribution
            # to d_log_decay at t=0 is missing the initial_state term --
            # same fixup FLA's own ChunkHGRNFunction.backward applies to
            # dg[:, 0] post-hoc, just pre-reduced here (summed over the
            # batch dim) since d_log_decay is already [D]-shaped.
            correction = (
                initial_state.float() * dx[:, 0].float() * log_decay.float().exp()
            ).sum(dim=0)
            d_log_decay = d_log_decay + correction
        return dx.to(o.dtype), d_log_decay.to(log_decay.dtype)

    @_fixed_decay_bwd_op.register_fake
    def _(log_decay, o, do, initial_state):
        return torch.empty_like(o), torch.empty_like(log_decay)

    def _setup_context(ctx, inputs, output):
        x, log_decay, initial_state = inputs
        o, final_state = output
        ctx.save_for_backward(log_decay, o, initial_state)

    def _backward(ctx, do, dfinal_state):
        # dfinal_state intentionally unused -- matches stock chunk_hgrn's
        # own backward (see gator_linear/_hgrn_compiled.py's _backward).
        log_decay, o, initial_state = ctx.saved_tensors
        do = do.contiguous()
        dx, d_log_decay = _fixed_decay_bwd_op(log_decay, o, do, initial_state)
        return dx, d_log_decay, None

    torch.library.register_autograd(
        "gator::fixed_decay_hgrn_fwd", _backward, setup_context=_setup_context,
    )


def fixed_decay_chunk_hgrn(
    x: Tensor,
    log_decay: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    """Specialized HGRN forward for a [D]-shaped, time/batch-invariant
    `log_decay` -- no [B, T, D] gate materialization, no [B, T, D] `gc`
    or `dg` intermediate/output. `log_decay` gets a properly-reduced
    [D]-shaped gradient directly (no caller-side .expand() needed).
    """
    if triton is None:
        raise RuntimeError(
            "fixed_decay_chunk_hgrn requires flash-linear-attention and triton; "
            f"import failed with: {_IMPORT_ERROR!r}"
        )
    if log_decay.ndim != 1 or log_decay.shape[0] != x.shape[-1]:
        raise ValueError("log_decay must have shape [channels]")
    x = x.contiguous()
    log_decay = log_decay.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    o, final_state = _fixed_decay_fwd_op(x, log_decay, initial_state)
    if not output_final_state:
        final_state = None
    return o, final_state
