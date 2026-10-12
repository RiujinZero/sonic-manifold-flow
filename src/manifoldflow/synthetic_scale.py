"""Synthetic bank/query generation for systems-scale ANN benchmarking.

Real Fashion-MNIST-derived banks do not reach the 100k-1M basin range the
design's roadmap requires for scale validation (Fashion-MNIST has 60,000
training images). This module generates a bounded Gaussian-mixture latent
space of the requested size so index build time, recall, latency, and memory
can be measured before that many basins exist from a real stream. It is a
systems-scale stand-in, not a correctness or learning-quality measurement.
"""

from __future__ import annotations

import torch
from torch import Tensor

from manifoldflow.bank import BasinState


@torch.no_grad()
def synthetic_gaussian_mixture_bank(
    basins: int,
    dimension: int,
    *,
    clusters: int | None = None,
    cluster_spread: float = 3.0,
    within_cluster_std: float = 0.6,
    seed: int = 17,
) -> BasinState:
    """Build a basin bank whose centers sit in a bounded Gaussian-mixture space.

    Centers are drawn around a fixed number of coarse cluster means rather
    than uniformly over a high-dimensional cube. Uniform random centers make
    approximate search trivially easy at any scale and would not exercise the
    local density the design's routing is meant to handle.
    """
    if basins <= 0 or dimension <= 0:
        raise ValueError("basins and dimension must be positive")
    if clusters is not None and clusters <= 0:
        raise ValueError("clusters must be positive")
    if cluster_spread <= 0 or within_cluster_std <= 0:
        raise ValueError("cluster_spread and within_cluster_std must be positive")
    generator = torch.Generator().manual_seed(seed)
    cluster_count = clusters if clusters is not None else max(1, round(basins**0.5))
    cluster_means = (
        torch.randn(cluster_count, dimension, generator=generator) * cluster_spread
    )
    assignment = torch.randint(0, cluster_count, (basins,), generator=generator)
    noise = torch.randn(basins, dimension, generator=generator) * within_cluster_std
    centers = cluster_means[assignment] + noise
    count = torch.randint(2, 50, (basins,), generator=generator)
    spread = torch.rand(basins, dimension, generator=generator) * 0.5 + 0.1
    m2 = spread * (count - 1).clamp_min(1).to(spread.dtype)[:, None]
    return BasinState.from_statistics(centers, count, m2)


@torch.no_grad()
def synthetic_query_latents(
    bank: BasinState,
    queries: int,
    *,
    near_fraction: float = 0.8,
    near_std: float = 0.5,
    far_spread: float = 3.0,
    seed: int = 19,
) -> Tensor:
    """Draw query latents, mostly near existing basins with some off-manifold."""
    if queries <= 0:
        raise ValueError("queries must be positive")
    if not 0.0 <= near_fraction <= 1.0:
        raise ValueError("near_fraction must be in [0, 1]")
    if near_std <= 0 or far_spread <= 0:
        raise ValueError("near_std and far_spread must be positive")
    generator = torch.Generator().manual_seed(seed)
    near_count = round(queries * near_fraction)
    far_count = queries - near_count
    dimension = bank.mu.shape[1]
    parts = []
    if near_count:
        basin_ids = torch.randint(
            0, bank.mu.shape[0], (near_count,), generator=generator
        )
        parts.append(
            bank.mu[basin_ids]
            + torch.randn(near_count, dimension, generator=generator) * near_std
        )
    if far_count:
        far = torch.randn(far_count, dimension, generator=generator) * far_spread
        parts.append(far)
    return torch.cat(parts, dim=0)
