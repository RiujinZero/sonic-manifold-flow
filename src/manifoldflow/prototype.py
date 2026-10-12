"""Simple prototype-memory transforms used to falsify analytic-flow value."""

from __future__ import annotations

import torch
from torch import Tensor

from manifoldflow.index import ExactIndex


def _validate_strength(strength: float) -> None:
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")


def contract_toward(latent: Tensor, target: Tensor, *, strength: float) -> Tensor:
    """Interpolate toward an already selected prototype target."""
    _validate_strength(strength)
    if target.shape != latent.shape:
        raise ValueError("target must match latent shape")
    return latent + strength * (target - latent)


@torch.no_grad()
def nearest_prototype_target(
    latent: Tensor,
    centers: Tensor,
    *,
    index: ExactIndex | None = None,
) -> Tensor:
    """Select the nearest Euclidean prototype for each latent."""
    index = index if index is not None else ExactIndex(centers)
    ids = index.query(latent, 1).squeeze(1)
    return centers.to(device=latent.device, dtype=latent.dtype)[ids]


@torch.no_grad()
def rbf_prototype_target(
    latent: Tensor,
    centers: Tensor,
    *,
    top_k: int = 4,
    temperature: float = 0.1,
    index: ExactIndex | None = None,
) -> Tensor:
    """Return a Euclidean RBF mixture of nearby prototypes."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    index = index if index is not None else ExactIndex(centers)
    if not 1 <= top_k <= index.size:
        raise ValueError("top_k must be in [1, prototype count]")
    ids = index.query(latent, top_k)
    local = centers.to(device=latent.device, dtype=latent.dtype)[ids]
    squared_distance = (latent[:, None, :] - local).square().sum(dim=-1)
    weight = torch.softmax(-0.5 * squared_distance / temperature, dim=-1)
    return (weight[..., None] * local).sum(dim=1)


@torch.no_grad()
def nearest_prototype_contraction(
    latent: Tensor,
    centers: Tensor,
    *,
    strength: float,
    index: ExactIndex | None = None,
) -> Tensor:
    """Move toward the nearest Euclidean prototype by a fixed fraction."""
    _validate_strength(strength)
    target = nearest_prototype_target(latent, centers, index=index)
    return contract_toward(latent, target, strength=strength)


@torch.no_grad()
def nearest_prototype_class_scores(
    latent: Tensor,
    centers: Tensor,
    class_count: Tensor,
    *,
    index: ExactIndex | None = None,
) -> Tensor:
    """Return each latent's nearest prototype's raw per-class vote counts.

    The vector-quantization (VQ) matched baseline: classify by the class
    histogram of the single nearest center, at whatever vector budget
    `centers` represents.
    """
    if class_count.ndim != 2 or class_count.shape[0] != centers.shape[0]:
        raise ValueError("class_count must have shape [centers, classes]")
    index = index if index is not None else ExactIndex(centers)
    ids = index.query(latent, 1).squeeze(1)
    return class_count.to(device=latent.device)[ids].float()


@torch.no_grad()
def rbf_prototype_class_scores(
    latent: Tensor,
    centers: Tensor,
    class_count: Tensor,
    *,
    top_k: int = 4,
    temperature: float = 0.1,
    index: ExactIndex | None = None,
) -> Tensor:
    """Return an RBF-weighted mixture of nearby prototypes' class scores."""
    if class_count.ndim != 2 or class_count.shape[0] != centers.shape[0]:
        raise ValueError("class_count must have shape [centers, classes]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    index = index if index is not None else ExactIndex(centers)
    if not 1 <= top_k <= index.size:
        raise ValueError("top_k must be in [1, prototype count]")
    ids = index.query(latent, top_k)
    local_centers = centers.to(device=latent.device, dtype=latent.dtype)[ids]
    squared_distance = (latent[:, None, :] - local_centers).square().sum(dim=-1)
    weight = torch.softmax(-0.5 * squared_distance / temperature, dim=-1)
    counts = class_count.to(latent.device)[ids].float()
    probability = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1)
    return (weight[..., None] * probability).sum(dim=1)


@torch.no_grad()
def rbf_prototype_contraction(
    latent: Tensor,
    centers: Tensor,
    *,
    strength: float,
    top_k: int = 4,
    temperature: float = 0.1,
    index: ExactIndex | None = None,
) -> Tensor:
    """Move toward a Euclidean RBF mixture of nearby prototypes."""
    _validate_strength(strength)
    target = rbf_prototype_target(
        latent,
        centers,
        top_k=top_k,
        temperature=temperature,
        index=index,
    )
    return contract_toward(latent, target, strength=strength)
