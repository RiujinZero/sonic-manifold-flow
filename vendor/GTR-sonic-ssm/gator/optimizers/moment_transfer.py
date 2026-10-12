"""Moment transfer across a directional-basis refresh (Procoim-S design spec
section 9.4).

When a group's directional-Adam basis rotates, this projects the old
coefficient moments onto the new basis by their geometric overlap, instead
of resetting to zero — a direction that barely changed keeps its momentum;
a genuinely new, non-overlapping direction naturally gets zero first moment
and floors to the variance floor, with no special-casing needed (see
:func:`transfer_moments`'s docstring for why).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["overlap_matrix", "transfer_moments"]


def overlap_matrix(new_basis: Tensor, old_basis: Tensor) -> Tensor:
    """``R[i, j] = <new_basis[i], old_basis[j]>`` for row-vector bases
    ``new_basis: [new_k, dim]``, ``old_basis: [old_k, dim]`` — the
    per-direction cosine overlap when both are unit-norm rows (design spec
    section 9.4's ``R = U_new^T U_old``, expressed for row-major bases)."""

    if new_basis.ndim != 2 or old_basis.ndim != 2:
        raise ValueError("both bases must be 2D [k, dim] tensors")
    if new_basis.shape[1] != old_basis.shape[1]:
        raise ValueError("both bases must share the same direction dimension")
    return new_basis @ old_basis.T


def transfer_moments(
    new_basis: Tensor,
    old_basis: Tensor,
    old_first_moment: Tensor,
    old_second_moment: Tensor,
    *,
    variance_floor: float,
) -> tuple[Tensor, Tensor]:
    """Project Adam moments from ``old_basis`` onto ``new_basis`` by overlap.

    ``m_new = R @ m_old``; ``v_new_j = sum_k R_jk^2 * v_old_k + variance_floor``
    (design spec section 9.4). A new direction orthogonal to every old one
    has ``R`` row ~0, so it lands at ``m_new ~= 0`` and ``v_new ~=
    variance_floor`` automatically — exactly the spec's stated behavior for
    "new non-overlapping directions" — without any special-casing.
    """

    if variance_floor < 0:
        raise ValueError("variance_floor must be nonnegative")
    if old_first_moment.shape != (old_basis.shape[0],):
        raise ValueError("old_first_moment must have one entry per old-basis direction")
    if old_second_moment.shape != (old_basis.shape[0],):
        raise ValueError("old_second_moment must have one entry per old-basis direction")

    overlap = overlap_matrix(new_basis, old_basis)
    new_first_moment = overlap @ old_first_moment
    new_second_moment = overlap.pow(2) @ old_second_moment + variance_floor
    return new_first_moment, new_second_moment
