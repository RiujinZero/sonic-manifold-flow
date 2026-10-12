"""Mini-batch K-means and exact sufficient-statistic basin initialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from manifoldflow.bank import BasinState


@dataclass(frozen=True)
class NearestAssignment:
    """Nearest-center IDs and squared distances for a latent batch."""

    ids: Tensor
    squared_distance: Tensor


@dataclass(frozen=True)
class MiniBatchKMeansResult:
    """Centers learned by the approximate mini-batch optimization stage."""

    centers: Tensor
    update_count: Tensor
    steps: int


@dataclass(frozen=True)
class InitializedBasinBank:
    """A populated bank plus task values and initialization diagnostics."""

    state: BasinState
    class_count: Tensor
    assignment: Tensor
    inertia: float
    requested_basins: int
    mini_batch_steps: int
    refinement_steps: int
    variance_scale: float

    @property
    def active_basins(self) -> int:
        return self.state.mu.shape[0]

    @property
    def class_probability(self) -> Tensor:
        denominator = self.class_count.sum(dim=1, keepdim=True).clamp_min(1)
        return self.class_count / denominator

    @torch.no_grad()
    def predict_nearest_value(
        self,
        latent: Tensor,
        *,
        query_chunk_size: int = 1024,
        center_chunk_size: int = 4096,
    ) -> Tensor:
        """Predict from the class distribution of the nearest basin."""
        assignment = nearest_center_assignment(
            latent,
            self.state.mu.to(device=latent.device, dtype=latent.dtype),
            query_chunk_size=query_chunk_size,
            center_chunk_size=center_chunk_size,
        )
        value = self.class_probability.to(latent.device)
        return value[assignment.ids].argmax(dim=1)

    def serializable(self) -> dict[str, Tensor | float | int]:
        """Return CPU state suitable for ``torch.save`` and later indexing."""
        return {
            "mu": self.state.mu.detach().cpu(),
            "raw_rate": self.state.raw_rate.detach().cpu(),
            "bias": self.state.bias.detach().cpu(),
            "count": self.state.count.detach().cpu(),
            "m2": self.state.m2.detach().cpu(),
            "usage": self.state.usage.detach().cpu(),
            "version": self.state.version,
            "class_count": self.class_count.detach().cpu(),
            "inertia": self.inertia,
            "requested_basins": self.requested_basins,
            "mini_batch_steps": self.mini_batch_steps,
            "refinement_steps": self.refinement_steps,
            "variance_scale": self.variance_scale,
        }


@torch.no_grad()
def nearest_center_assignment(
    latent: Tensor,
    centers: Tensor,
    *,
    query_chunk_size: int = 1024,
    center_chunk_size: int = 4096,
) -> NearestAssignment:
    """Assign exact Euclidean winners while bounding the distance workspace."""
    _validate_latent_centers(latent, centers)
    if query_chunk_size <= 0 or center_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if latent.device != centers.device or latent.dtype != centers.dtype:
        raise ValueError("latent and centers must share device and dtype")

    id_parts = []
    distance_parts = []
    for query_start in range(0, latent.shape[0], query_chunk_size):
        query = latent[query_start : query_start + query_chunk_size]
        best_distance = torch.full(
            (query.shape[0],),
            torch.inf,
            dtype=latent.dtype,
            device=latent.device,
        )
        best_id = torch.zeros(query.shape[0], dtype=torch.long, device=latent.device)
        for center_start in range(0, centers.shape[0], center_chunk_size):
            local_centers = centers[center_start : center_start + center_chunk_size]
            local_distance, local_id = torch.cdist(query, local_centers).min(dim=1)
            replace = local_distance < best_distance
            best_distance = torch.where(replace, local_distance, best_distance)
            best_id = torch.where(replace, local_id + center_start, best_id)
        id_parts.append(best_id)
        distance_parts.append(best_distance.square())
    return NearestAssignment(
        ids=torch.cat(id_parts), squared_distance=torch.cat(distance_parts)
    )


@torch.no_grad()
def mini_batch_kmeans(
    latent: Tensor,
    *,
    clusters: int,
    batch_size: int = 512,
    steps: int = 100,
    seed: int = 17,
    center_chunk_size: int = 4096,
) -> MiniBatchKMeansResult:
    """Fit centers with random-sample initialization and online batch means."""
    if latent.ndim != 2 or latent.shape[0] == 0:
        raise ValueError("latent must have shape [samples, d] with samples > 0")
    if not 1 <= clusters <= latent.shape[0]:
        raise ValueError("clusters must be in [1, sample count]")
    if batch_size <= 0 or steps <= 0:
        raise ValueError("batch_size and steps must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    generator = torch.Generator(device=latent.device).manual_seed(seed)
    initial_rows = torch.randperm(
        latent.shape[0], generator=generator, device=latent.device
    )[:clusters]
    centers = latent[initial_rows].detach().clone()
    update_count = torch.zeros(clusters, dtype=torch.long, device=latent.device)

    for _ in range(steps):
        if batch_size >= latent.shape[0]:
            rows = torch.randperm(
                latent.shape[0], generator=generator, device=latent.device
            )
        else:
            rows = torch.randint(
                latent.shape[0],
                (batch_size,),
                generator=generator,
                device=latent.device,
            )
        batch = latent[rows]
        assignment = nearest_center_assignment(
            batch,
            centers,
            query_chunk_size=batch.shape[0],
            center_chunk_size=center_chunk_size,
        ).ids
        batch_count = torch.bincount(assignment, minlength=clusters)
        batch_sum = torch.zeros_like(centers)
        batch_sum.index_add_(0, assignment, batch)
        new_count = update_count + batch_count
        active = batch_count > 0
        centers[active] = (
            centers[active] * update_count[active, None].to(centers.dtype)
            + batch_sum[active]
        ) / new_count[active, None].to(centers.dtype)
        update_count = new_count

    return MiniBatchKMeansResult(
        centers=centers,
        update_count=update_count,
        steps=steps,
    )


@torch.no_grad()
def initialize_basin_bank(
    latent: Tensor,
    label: Tensor,
    *,
    basins: int,
    classes: int,
    mini_batch_size: int = 512,
    mini_batch_steps: int = 100,
    refinement_steps: int = 5,
    seed: int = 17,
    query_chunk_size: int = 1024,
    center_chunk_size: int = 4096,
    variance_scale: float | None = None,
    target_median_rate: float = 4.0,
    variance_epsilon: float = 1e-4,
    rate_min: float = 1e-4,
    rate_max: float = 32.0,
    bias_bound: float = 10.0,
) -> InitializedBasinBank:
    """Initialize, refine, and exactly populate a local attractor bank."""
    _validate_labels(latent, label, classes)
    if refinement_steps < 0:
        raise ValueError("refinement_steps must be non-negative")
    if bias_bound <= 0:
        raise ValueError("bias_bound must be positive")
    if not rate_min < target_median_rate <= rate_max:
        raise ValueError("target_median_rate must be in (rate_min, rate_max]")

    approximate = mini_batch_kmeans(
        latent,
        clusters=basins,
        batch_size=mini_batch_size,
        steps=mini_batch_steps,
        seed=seed,
        center_chunk_size=center_chunk_size,
    )
    centers = approximate.centers
    completed_refinements = 0
    for _ in range(refinement_steps):
        assignment = nearest_center_assignment(
            latent,
            centers,
            query_chunk_size=query_chunk_size,
            center_chunk_size=center_chunk_size,
        ).ids
        count, means = _assigned_means(latent, assignment, centers.shape[0])
        occupied = count > 0
        centers = means[occupied]
        completed_refinements += 1

    final_assignment = nearest_center_assignment(
        latent,
        centers,
        query_chunk_size=query_chunk_size,
        center_chunk_size=center_chunk_size,
    )
    count, exact_center = _assigned_means(
        latent, final_assignment.ids, centers.shape[0]
    )
    occupied = count > 0
    if not torch.all(occupied):
        mapping = torch.full(
            (centers.shape[0],), -1, dtype=torch.long, device=latent.device
        )
        mapping[occupied] = torch.arange(occupied.sum(), device=latent.device)
        assignment_ids = mapping[final_assignment.ids]
        count = count[occupied]
        exact_center = exact_center[occupied]
    else:
        assignment_ids = final_assignment.ids

    delta = latent - exact_center[assignment_ids]
    m2 = torch.zeros_like(exact_center)
    m2.index_add_(0, assignment_ids, delta.square())
    occupancy = count.to(latent.dtype) / latent.shape[0]
    bias = occupancy.log().clamp(min=-bias_bound, max=0.0)
    if variance_scale is None:
        denominator = (count - 1).clamp_min(1).to(latent.dtype)[:, None]
        variance = m2 / denominator
        eligible = (count > 1)[:, None].expand_as(variance)
        if torch.any(eligible):
            median_variance = variance[eligible].median()
        else:
            median_variance = torch.zeros((), dtype=latent.dtype, device=latent.device)
        variance_scale = float(
            (target_median_rate * (median_variance + variance_epsilon)).item()
        )
    state = BasinState.from_statistics(
        exact_center,
        count,
        m2,
        bias=bias,
        variance_scale=variance_scale,
        epsilon=variance_epsilon,
        rate_min=rate_min,
        rate_max=rate_max,
    )

    flat_value_id = assignment_ids * classes + label
    class_count = torch.bincount(
        flat_value_id, minlength=state.mu.shape[0] * classes
    ).reshape(state.mu.shape[0], classes)
    inertia = float(delta.square().sum(dim=1).mean().item())
    return InitializedBasinBank(
        state=state,
        class_count=class_count,
        assignment=assignment_ids,
        inertia=inertia,
        requested_basins=basins,
        mini_batch_steps=mini_batch_steps,
        refinement_steps=completed_refinements,
        variance_scale=variance_scale,
    )


def _assigned_means(
    latent: Tensor,
    assignment: Tensor,
    clusters: int,
) -> tuple[Tensor, Tensor]:
    count = torch.bincount(assignment, minlength=clusters)
    total = torch.zeros(
        clusters, latent.shape[1], dtype=latent.dtype, device=latent.device
    )
    total.index_add_(0, assignment, latent)
    means = total / count.clamp_min(1).to(latent.dtype)[:, None]
    return count, means


def _validate_latent_centers(latent: Tensor, centers: Tensor) -> None:
    if latent.ndim != 2 or latent.shape[0] == 0:
        raise ValueError("latent must have shape [samples, d] with samples > 0")
    if centers.ndim != 2 or centers.shape[0] == 0:
        raise ValueError("centers must have shape [basins, d] with basins > 0")
    if latent.shape[1] != centers.shape[1]:
        raise ValueError("latent and centers must share their feature width")
    if not latent.is_floating_point() or not centers.is_floating_point():
        raise ValueError("latent and centers must use floating-point dtypes")


def _validate_labels(latent: Tensor, label: Tensor, classes: int) -> None:
    if label.shape != (latent.shape[0],) or label.dtype != torch.long:
        raise ValueError("label must use torch.long with shape [samples]")
    if classes <= 0:
        raise ValueError("classes must be positive")
    if torch.any(label < 0) or torch.any(label >= classes):
        raise ValueError("label values must be in [0, classes)")
