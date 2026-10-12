"""Custom Triton kernel for the N=4 weighted-node-mixing pattern that
appears at both gator/models/language.py's _dense_relay_transform
(with a fused residual) and gator/models/memory.py's receiver-context
finalization (without one):

    y[b,c,t,:] = [h[b,c,t,:] +] sum_{n=0}^{3} q[b,c,t,n] * dest[b,c,t,n,:]

Profiling the real compiled production model (Runbooks/colab_cells/
_recompiled_kernel_profile.py) found this pattern's reduction
operations collectively responsible for ~40% of total GPU time, far
more than receiver-memory's own chunk_hgrn kernel (~6%) -- see
Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md.

Written as `q.unsqueeze(-1) * destination).sum(3)`, this materializes
a full [B, C, T, 4, D] elementwise product (~172M elements at
production shape: B=2, C=8, T=2048, D=1312) purely to reduce it back
down over an axis of size 4 -- a generic elementwise-multiply-then-
reduce path when N is fixed and tiny. This module writes the exact
operation instead: one streaming kernel that loads 4 destination
vectors, 4 routing scalars, and (optionally) a residual, does 4 FMAs
per output element, and stores -- no intermediate tensor, no generic
reduction, and (in the residual-fused case) no separate residual-add
kernel/tensor round-trip either.

Same compile-friendliness approach as gator_linear/_hgrn_compiled.py
and gator_linear/_hgrn_fixed_decay.py: torch.library.triton_op +
register_autograd, with the same do.contiguous() defensive guard
against autograd's expanded stride-0 gradients (confirmed necessary
there; applied here preemptively rather than rediscovered the hard
way).

Backward, worked out by hand for y = h + sum_n q[n] * dest[n]:
    dh          = dy
    dDest[n, d] = q[n] * dy[d]
    dq[n]       = sum_d dy[d] * dest[n, d]
only dq is a real reduction (over D, not over the tiny N=4 axis).
"""

from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - exercised only without triton
    triton = None  # type: ignore[assignment]
    _IMPORT_ERROR = error


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_D": BLOCK_D}, num_warps=num_warps)
            for BLOCK_D in [128, 256, 512]
            for num_warps in [4, 8]
        ],
        key=["D"],
    )
    @triton.jit
    def _weighted_nodes4_fwd_kernel(
        q_ptr, dest_ptr, h_ptr, out_ptr,
        D: tl.constexpr, BLOCK_D: tl.constexpr, USE_RESIDUAL: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        d = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d < D

        q_base = row * 4
        q0 = tl.load(q_ptr + q_base + 0).to(tl.float32)
        q1 = tl.load(q_ptr + q_base + 1).to(tl.float32)
        q2 = tl.load(q_ptr + q_base + 2).to(tl.float32)
        q3 = tl.load(q_ptr + q_base + 3).to(tl.float32)

        dest_base = row * 4 * D
        d0 = tl.load(dest_ptr + dest_base + 0 * D + d, mask=mask, other=0.0).to(tl.float32)
        d1 = tl.load(dest_ptr + dest_base + 1 * D + d, mask=mask, other=0.0).to(tl.float32)
        d2 = tl.load(dest_ptr + dest_base + 2 * D + d, mask=mask, other=0.0).to(tl.float32)
        d3 = tl.load(dest_ptr + dest_base + 3 * D + d, mask=mask, other=0.0).to(tl.float32)

        y = q0 * d0 + q1 * d1 + q2 * d2 + q3 * d3
        if USE_RESIDUAL:
            h = tl.load(h_ptr + row * D + d, mask=mask, other=0.0).to(tl.float32)
            y = y + h

        tl.store(out_ptr + row * D + d, y.to(out_ptr.dtype.element_ty), mask=mask)

    # Deliberately NOT @triton.autotune'd, unlike the forward kernel:
    # this kernel does tl.atomic_add into a caller-allocated dq buffer,
    # and triton.autotune's benchmarking re-runs a kernel multiple
    # times per candidate config purely to time it. Since dq isn't
    # reset between those internal timing repeats, every extra
    # benchmark run accumulates again on top of the last -- confirmed
    # directly: dq came out ~2500x too large, a near-constant multiple
    # across every element, matching autotune's internal repeat count.
    # Fixed block size instead (same pattern FLA's own chunk_hgrn uses
    # for its kernel_o variants, which also carry accumulation
    # semantics -- only its kernel_h variants, which have no atomics,
    # are autotuned).
    @triton.jit
    def _weighted_nodes4_bwd_kernel(
        q_ptr, dest_ptr, dy_ptr, d_dest_ptr, dq_ptr,
        D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        d = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d < D

        q_base = row * 4
        q0 = tl.load(q_ptr + q_base + 0).to(tl.float32)
        q1 = tl.load(q_ptr + q_base + 1).to(tl.float32)
        q2 = tl.load(q_ptr + q_base + 2).to(tl.float32)
        q3 = tl.load(q_ptr + q_base + 3).to(tl.float32)

        dest_base = row * 4 * D
        d0 = tl.load(dest_ptr + dest_base + 0 * D + d, mask=mask, other=0.0).to(tl.float32)
        d1 = tl.load(dest_ptr + dest_base + 1 * D + d, mask=mask, other=0.0).to(tl.float32)
        d2 = tl.load(dest_ptr + dest_base + 2 * D + d, mask=mask, other=0.0).to(tl.float32)
        d3 = tl.load(dest_ptr + dest_base + 3 * D + d, mask=mask, other=0.0).to(tl.float32)

        dy = tl.load(dy_ptr + row * D + d, mask=mask, other=0.0).to(tl.float32)

        dtype = d_dest_ptr.dtype.element_ty
        tl.store(d_dest_ptr + dest_base + 0 * D + d, (q0 * dy).to(dtype), mask=mask)
        tl.store(d_dest_ptr + dest_base + 1 * D + d, (q1 * dy).to(dtype), mask=mask)
        tl.store(d_dest_ptr + dest_base + 2 * D + d, (q2 * dy).to(dtype), mask=mask)
        tl.store(d_dest_ptr + dest_base + 3 * D + d, (q3 * dy).to(dtype), mask=mask)

        dq0 = tl.sum(tl.where(mask, dy * d0, 0.0), axis=0)
        dq1 = tl.sum(tl.where(mask, dy * d1, 0.0), axis=0)
        dq2 = tl.sum(tl.where(mask, dy * d2, 0.0), axis=0)
        dq3 = tl.sum(tl.where(mask, dy * d3, 0.0), axis=0)
        tl.atomic_add(dq_ptr + q_base + 0, dq0)
        tl.atomic_add(dq_ptr + q_base + 1, dq1)
        tl.atomic_add(dq_ptr + q_base + 2, dq2)
        tl.atomic_add(dq_ptr + q_base + 3, dq3)

    @torch.library.triton_op("gator::weighted_nodes4_fwd", mutates_args={})
    def _weighted_nodes4_fwd_op(
        q: Tensor,
        destination: Tensor,
        h: Optional[Tensor],
    ) -> Tensor:
        M, N = q.shape
        D = destination.shape[-1]
        out = torch.empty(M, D, device=q.device, dtype=destination.dtype)

        def grid(meta):
            return (M, triton.cdiv(D, meta["BLOCK_D"]))

        h_arg = h if h is not None else out  # unused (masked by USE_RESIDUAL) but must be a real pointer
        torch.library.wrap_triton(_weighted_nodes4_fwd_kernel)[grid](
            q, destination, h_arg, out,
            D=D, USE_RESIDUAL=h is not None,
        )
        return out

    @_weighted_nodes4_fwd_op.register_fake
    def _(q, destination, h):
        M, N = q.shape
        D = destination.shape[-1]
        return torch.empty(M, D, device=q.device, dtype=destination.dtype)

    @torch.library.triton_op("gator::weighted_nodes4_bwd", mutates_args={})
    def _weighted_nodes4_bwd_op(
        q: Tensor,
        destination: Tensor,
        dy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        M, N = q.shape
        D = destination.shape[-1]
        d_dest = torch.empty_like(destination)
        dq = torch.zeros(M, N, device=q.device, dtype=torch.float32)

        BLOCK_D = 256
        grid = (M, triton.cdiv(D, BLOCK_D))
        torch.library.wrap_triton(_weighted_nodes4_bwd_kernel)[grid](
            q, destination, dy, d_dest, dq, D=D, BLOCK_D=BLOCK_D, num_warps=8,
        )
        return d_dest, dq.to(q.dtype)

    @_weighted_nodes4_bwd_op.register_fake
    def _(q, destination, dy):
        return torch.empty_like(destination), torch.empty_like(q)

    def _setup_context(ctx, inputs, output):
        q, destination, h = inputs
        ctx.save_for_backward(q, destination)
        ctx.use_residual = h is not None

    def _backward(ctx, dy):
        q, destination = ctx.saved_tensors
        dy = dy.contiguous()
        d_dest, dq = _weighted_nodes4_bwd_op(q, destination, dy)
        dh = dy if ctx.use_residual else None
        return dq, d_dest, dh

    torch.library.register_autograd(
        "gator::weighted_nodes4_fwd", _backward, setup_context=_setup_context,
    )


def weighted_nodes4(
    q: Tensor,
    destination: Tensor,
    h: Optional[Tensor] = None,
) -> Tensor:
    """``[h +] sum_n q[..., n] * destination[..., n, :]`` for a fixed
    N=4 last-but-one axis, as one streaming Triton kernel (fused
    residual add when ``h`` is given) instead of
    ``(q.unsqueeze(-1) * destination).sum(-2)`` [+ h], which
    materializes a full ``[..., 4, D]`` elementwise product purely to
    reduce it away over an axis of size 4.

    Args:
        q: ``[..., 4]``.
        destination: ``[..., 4, D]`` (same leading dims as ``q``).
        h: optional ``[..., D]`` residual, fused into the same kernel.

    Returns:
        ``[..., D]``.
    """
    if triton is None:
        raise RuntimeError(
            "weighted_nodes4 requires triton; import failed with: "
            f"{_IMPORT_ERROR!r}"
        )
    if q.shape[-1] != 4:
        raise ValueError("weighted_nodes4 requires a fixed last dim of 4 on q")
    if destination.shape[:-1] != q.shape:
        raise ValueError("destination must have shape q.shape + [D]")
    if h is not None and h.shape != destination.shape[:-2] + destination.shape[-1:]:
        raise ValueError("h must have shape destination.shape[:-2] + [D]")

    leading_shape = q.shape[:-1]
    D = destination.shape[-1]
    M = q.numel() // 4

    q_flat = q.reshape(M, 4).contiguous()
    dest_flat = destination.reshape(M, 4, D).contiguous()
    h_flat = h.reshape(M, D).contiguous() if h is not None else None

    out = _weighted_nodes4_fwd_op(q_flat, dest_flat, h_flat)
    return out.reshape(*leading_shape, D)


# --- tanh-fused variant -----------------------------------------------
#
# gator/models/language.py's actual call site is
# `destination = torch.tanh(node_affine); local = cell_norm(h +
# (q.unsqueeze(-1) * destination).sum(3))` -- `destination` exists
# ONLY to feed that one line. weighted_nodes4() above still takes
# `destination` as an already-materialized [..., 4, D] tensor, so
# neither it nor the plain baseline it was benchmarked against
# actually pay (or save) the cost of that intermediate -- the real
# question is whether fusing `tanh` INTO this kernel, operating
# directly on `node_affine` and never writing `destination` at all,
# beats a baseline that also does the tanh (a fair comparison, unlike
# the first benchmark). Forward here writes only the final [..., D]
# output; backward recomputes tanh from the saved `node_affine`
# (already saved regardless, since it's a real op input) rather than
# saving the [..., 4, D] tanh output as a separate tensor.
if triton is not None:

    @triton.jit
    def _stable_tanh(x):
        # tanh(x) = sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|)),
        # avoids exp() overflow for large |x| that a naive
        # (exp(2x)-1)/(exp(2x)+1) would hit.
        ax = tl.abs(x)
        t = tl.math.exp(-2.0 * ax)
        r = (1.0 - t) / (1.0 + t)
        return tl.where(x >= 0, r, -r)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_D": BLOCK_D}, num_warps=num_warps)
            for BLOCK_D in [128, 256, 512]
            for num_warps in [4, 8]
        ],
        key=["D"],
    )
    @triton.jit
    def _weighted_nodes4_tanh_fwd_kernel(
        q_ptr, affine_ptr, h_ptr, out_ptr,
        D: tl.constexpr, BLOCK_D: tl.constexpr, USE_RESIDUAL: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        d = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d < D

        q_base = row * 4
        q0 = tl.load(q_ptr + q_base + 0).to(tl.float32)
        q1 = tl.load(q_ptr + q_base + 1).to(tl.float32)
        q2 = tl.load(q_ptr + q_base + 2).to(tl.float32)
        q3 = tl.load(q_ptr + q_base + 3).to(tl.float32)

        affine_base = row * 4 * D
        a0 = tl.load(affine_ptr + affine_base + 0 * D + d, mask=mask, other=0.0).to(tl.float32)
        a1 = tl.load(affine_ptr + affine_base + 1 * D + d, mask=mask, other=0.0).to(tl.float32)
        a2 = tl.load(affine_ptr + affine_base + 2 * D + d, mask=mask, other=0.0).to(tl.float32)
        a3 = tl.load(affine_ptr + affine_base + 3 * D + d, mask=mask, other=0.0).to(tl.float32)

        y = (
            q0 * _stable_tanh(a0) + q1 * _stable_tanh(a1)
            + q2 * _stable_tanh(a2) + q3 * _stable_tanh(a3)
        )
        if USE_RESIDUAL:
            h = tl.load(h_ptr + row * D + d, mask=mask, other=0.0).to(tl.float32)
            y = y + h

        tl.store(out_ptr + row * D + d, y.to(out_ptr.dtype.element_ty), mask=mask)

    # Not autotuned -- same atomic_add-vs-autotune hazard as
    # _weighted_nodes4_bwd_kernel above.
    @triton.jit
    def _weighted_nodes4_tanh_bwd_kernel(
        q_ptr, affine_ptr, dy_ptr, d_affine_ptr, dq_ptr,
        D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        d = block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d < D

        q_base = row * 4
        q0 = tl.load(q_ptr + q_base + 0).to(tl.float32)
        q1 = tl.load(q_ptr + q_base + 1).to(tl.float32)
        q2 = tl.load(q_ptr + q_base + 2).to(tl.float32)
        q3 = tl.load(q_ptr + q_base + 3).to(tl.float32)

        affine_base = row * 4 * D
        a0 = tl.load(affine_ptr + affine_base + 0 * D + d, mask=mask, other=0.0).to(tl.float32)
        a1 = tl.load(affine_ptr + affine_base + 1 * D + d, mask=mask, other=0.0).to(tl.float32)
        a2 = tl.load(affine_ptr + affine_base + 2 * D + d, mask=mask, other=0.0).to(tl.float32)
        a3 = tl.load(affine_ptr + affine_base + 3 * D + d, mask=mask, other=0.0).to(tl.float32)
        t0, t1, t2, t3 = _stable_tanh(a0), _stable_tanh(a1), _stable_tanh(a2), _stable_tanh(a3)

        dy = tl.load(dy_ptr + row * D + d, mask=mask, other=0.0).to(tl.float32)

        dtype = d_affine_ptr.dtype.element_ty
        tl.store(d_affine_ptr + affine_base + 0 * D + d, (dy * q0 * (1.0 - t0 * t0)).to(dtype), mask=mask)
        tl.store(d_affine_ptr + affine_base + 1 * D + d, (dy * q1 * (1.0 - t1 * t1)).to(dtype), mask=mask)
        tl.store(d_affine_ptr + affine_base + 2 * D + d, (dy * q2 * (1.0 - t2 * t2)).to(dtype), mask=mask)
        tl.store(d_affine_ptr + affine_base + 3 * D + d, (dy * q3 * (1.0 - t3 * t3)).to(dtype), mask=mask)

        dq0 = tl.sum(tl.where(mask, dy * t0, 0.0), axis=0)
        dq1 = tl.sum(tl.where(mask, dy * t1, 0.0), axis=0)
        dq2 = tl.sum(tl.where(mask, dy * t2, 0.0), axis=0)
        dq3 = tl.sum(tl.where(mask, dy * t3, 0.0), axis=0)
        tl.atomic_add(dq_ptr + q_base + 0, dq0)
        tl.atomic_add(dq_ptr + q_base + 1, dq1)
        tl.atomic_add(dq_ptr + q_base + 2, dq2)
        tl.atomic_add(dq_ptr + q_base + 3, dq3)

    @torch.library.triton_op("gator::weighted_nodes4_tanh_fwd", mutates_args={})
    def _weighted_nodes4_tanh_fwd_op(
        q: Tensor,
        node_affine: Tensor,
        h: Optional[Tensor],
    ) -> Tensor:
        M, N = q.shape
        D = node_affine.shape[-1]
        out = torch.empty(M, D, device=q.device, dtype=node_affine.dtype)

        def grid(meta):
            return (M, triton.cdiv(D, meta["BLOCK_D"]))

        h_arg = h if h is not None else out
        torch.library.wrap_triton(_weighted_nodes4_tanh_fwd_kernel)[grid](
            q, node_affine, h_arg, out,
            D=D, USE_RESIDUAL=h is not None,
        )
        return out

    @_weighted_nodes4_tanh_fwd_op.register_fake
    def _(q, node_affine, h):
        M, N = q.shape
        D = node_affine.shape[-1]
        return torch.empty(M, D, device=q.device, dtype=node_affine.dtype)

    @torch.library.triton_op("gator::weighted_nodes4_tanh_bwd", mutates_args={})
    def _weighted_nodes4_tanh_bwd_op(
        q: Tensor,
        node_affine: Tensor,
        dy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        M, N = q.shape
        D = node_affine.shape[-1]
        d_affine = torch.empty_like(node_affine)
        dq = torch.zeros(M, N, device=q.device, dtype=torch.float32)

        BLOCK_D = 256
        grid = (M, triton.cdiv(D, BLOCK_D))
        torch.library.wrap_triton(_weighted_nodes4_tanh_bwd_kernel)[grid](
            q, node_affine, dy, d_affine, dq, D=D, BLOCK_D=BLOCK_D, num_warps=8,
        )
        return d_affine, dq.to(q.dtype)

    @_weighted_nodes4_tanh_bwd_op.register_fake
    def _(q, node_affine, dy):
        return torch.empty_like(node_affine), torch.empty_like(q)

    def _setup_context_tanh(ctx, inputs, output):
        q, node_affine, h = inputs
        ctx.save_for_backward(q, node_affine)
        ctx.use_residual = h is not None

    def _backward_tanh(ctx, dy):
        q, node_affine = ctx.saved_tensors
        dy = dy.contiguous()
        d_affine, dq = _weighted_nodes4_tanh_bwd_op(q, node_affine, dy)
        dh = dy if ctx.use_residual else None
        return dq, d_affine, dh

    torch.library.register_autograd(
        "gator::weighted_nodes4_tanh_fwd", _backward_tanh, setup_context=_setup_context_tanh,
    )


def weighted_nodes4_tanh(
    q: Tensor,
    node_affine: Tensor,
    h: Optional[Tensor] = None,
) -> Tensor:
    """Same as ``weighted_nodes4`` but fuses the ``tanh`` in too:
    ``[h +] sum_n q[..., n] * tanh(node_affine[..., n, :])``, taking
    the PRE-tanh affine output directly so the ``tanh`` result (a full
    ``[..., 4, D]`` tensor) is never materialized at all -- backward
    recomputes it from the saved ``node_affine`` instead.

    Args:
        q: ``[..., 4]``.
        node_affine: ``[..., 4, D]`` (same leading dims as ``q``),
            PRE-tanh.
        h: optional ``[..., D]`` residual, fused into the same kernel.

    Returns:
        ``[..., D]``.
    """
    if triton is None:
        raise RuntimeError(
            "weighted_nodes4_tanh requires triton; import failed with: "
            f"{_IMPORT_ERROR!r}"
        )
    if q.shape[-1] != 4:
        raise ValueError("weighted_nodes4_tanh requires a fixed last dim of 4 on q")
    if node_affine.shape[:-1] != q.shape:
        raise ValueError("node_affine must have shape q.shape + [D]")
    if h is not None and h.shape != node_affine.shape[:-2] + node_affine.shape[-1:]:
        raise ValueError("h must have shape node_affine.shape[:-2] + [D]")

    leading_shape = q.shape[:-1]
    D = node_affine.shape[-1]
    M = q.numel() // 4

    q_flat = q.reshape(M, 4).contiguous()
    affine_flat = node_affine.reshape(M, 4, D).contiguous()
    h_flat = h.reshape(M, D).contiguous() if h is not None else None

    out = _weighted_nodes4_tanh_fwd_op(q_flat, affine_flat, h_flat)
    return out.reshape(*leading_shape, D)
