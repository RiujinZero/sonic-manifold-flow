"""Stable local quadratic energies and their analytic frozen-gate flows."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def positive_rate(
    raw_rate: Tensor,
    rate_min: float = 1e-4,
    rate_max: float = 32.0,
) -> Tensor:
    """Map unconstrained parameters to bounded positive contraction rates."""
    if rate_min <= 0:
        raise ValueError("rate_min must be positive")
    if rate_max < rate_min:
        raise ValueError("rate_max must be greater than or equal to rate_min")
    return (F.softplus(raw_rate) + rate_min).clamp(min=rate_min, max=rate_max)


def _validate_local_shapes(
    z0: Tensor,
    mu: Tensor,
    raw_rate: Tensor,
    bias: Tensor,
) -> None:
    if z0.ndim != 2:
        raise ValueError("z0 must have shape [batch, d]")
    if mu.ndim != 3:
        raise ValueError("mu must have shape [batch, k, d]")
    if mu.shape != raw_rate.shape:
        raise ValueError("mu and raw_rate must have identical shapes")
    if bias.shape != mu.shape[:2]:
        raise ValueError("bias must have shape [batch, k]")
    if z0.shape[0] != mu.shape[0] or z0.shape[1] != mu.shape[2]:
        raise ValueError("z0, mu, and raw_rate batch/latent dimensions must agree")
    if mu.shape[1] == 0:
        raise ValueError("at least one candidate basin is required")


def basin_energy(
    z0: Tensor,
    mu: Tensor,
    raw_rate: Tensor,
    bias: Tensor,
    *,
    rate_min: float = 1e-4,
    rate_max: float = 32.0,
) -> Tensor:
    """Evaluate anisotropic local energy for an already gathered candidate set."""
    _validate_local_shapes(z0, mu, raw_rate, bias)
    rate = positive_rate(raw_rate, rate_min=rate_min, rate_max=rate_max)
    delta = z0[:, None, :] - mu
    return 0.5 * (rate * delta.square()).sum(dim=-1) - bias


def analytic_topk_flow(
    z0: Tensor,
    mu: Tensor,
    raw_rate: Tensor,
    bias: Tensor,
    *,
    t: float | Tensor = 1.0,
    temperature: float = 0.1,
    rate_min: float = 1e-4,
    rate_max: float = 32.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one exact frozen-gate soft local flow segment.

    Candidate selection must happen before this function. The soft gate is
    computed at ``z0`` and held fixed for the whole segment, making the affine
    vector field analytically solvable.
    """
    _validate_local_shapes(z0, mu, raw_rate, bias)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if isinstance(t, Tensor):
        if t.ndim == 0:
            flow_time = t
        elif t.shape == (z0.shape[0],):
            flow_time = t[:, None]
        elif t.shape == (z0.shape[0], 1):
            flow_time = t
        else:
            raise ValueError("tensor t must be scalar, [batch], or [batch, 1]")
        if torch.any(flow_time < 0):
            raise ValueError("t must be non-negative")
    else:
        if t < 0:
            raise ValueError("t must be non-negative")
        flow_time = t

    rate = positive_rate(raw_rate, rate_min=rate_min, rate_max=rate_max)
    delta = z0[:, None, :] - mu
    energy = 0.5 * (rate * delta.square()).sum(dim=-1) - bias
    weight = torch.softmax(-energy / temperature, dim=-1)

    weighted_rate = weight[..., None] * rate
    effective_rate = weighted_rate.sum(dim=1).clamp_min(1e-6)
    effective_center = (weighted_rate * mu).sum(dim=1) / effective_rate

    zt = effective_center + (z0 - effective_center) * torch.exp(
        -effective_rate * flow_time
    )
    return zt, energy, weight
