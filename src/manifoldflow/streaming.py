"""Foreground statistical adaptation with novelty staging and a delta index."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from manifoldflow.bank import BasinState
from manifoldflow.index import ExactMainDeltaIndex


@dataclass(frozen=True)
class NoveltyProposal:
    """Sufficient statistics for one persistence-validated new basin."""

    center: Tensor
    count: int
    m2: Tensor
    class_count: Tensor | None


@dataclass
class _PendingNovelty:
    center: Tensor
    count: int
    m2: Tensor
    last_seen: int
    class_count: Tensor | None


class NoveltyStager:
    """Accumulate nearby novel observations before proposing a basin."""

    def __init__(
        self,
        *,
        dimension: int,
        persistence: int = 3,
        support_radius: float,
        max_pending: int = 256,
        max_age: int = 1024,
        classes: int | None = None,
    ) -> None:
        if dimension <= 0 or persistence < 2:
            raise ValueError("dimension must be positive and persistence must be >= 2")
        if support_radius <= 0 or max_pending <= 0 or max_age <= 0:
            raise ValueError("staging bounds must be positive")
        if classes is not None and classes < 2:
            raise ValueError("classes must be >= 2 when supplied")
        self.dimension = dimension
        self.persistence = persistence
        self.support_radius_squared = support_radius**2
        self.max_pending = max_pending
        self.max_age = max_age
        self.classes = classes
        self._pending: list[_PendingNovelty] = []
        self._step = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _new_class_count(self, observation: Tensor, label: int | None) -> Tensor | None:
        if self.classes is None:
            if label is not None:
                raise ValueError("labels require a class-aware novelty stager")
            return None
        if label is None or not 0 <= label < self.classes:
            raise ValueError("a valid label is required by a class-aware stager")
        result = torch.zeros(self.classes, dtype=torch.long, device=observation.device)
        result[label] = 1
        return result

    @torch.no_grad()
    def observe(
        self, observation: Tensor, *, label: int | None = None
    ) -> NoveltyProposal | None:
        """Stage one novel observation and return a proposal at persistence."""
        if observation.shape != (self.dimension,):
            raise ValueError("observation must have shape [dimension]")
        self._step += 1
        self._pending = [
            item
            for item in self._pending
            if self._step - item.last_seen <= self.max_age
        ]
        class_increment = self._new_class_count(observation, label)
        if self._pending:
            compatible_indices = list(range(len(self._pending)))
            if self.classes is not None:
                assert label is not None
                compatible_indices = [
                    index
                    for index, item in enumerate(self._pending)
                    if item.class_count is not None
                    and int(item.class_count[label].item()) == item.count
                ]
            centers = (
                torch.stack(
                    [self._pending[index].center for index in compatible_indices]
                ).to(device=observation.device, dtype=observation.dtype)
                if compatible_indices
                else observation.new_empty((0, self.dimension))
            )
            if centers.shape[0]:
                distance = (centers - observation).square().sum(dim=-1)
                nearest_local = int(distance.argmin().item())
                nearest = compatible_indices[nearest_local]
            else:
                distance = observation.new_empty(0)
                nearest_local = -1
                nearest = -1
            if (
                centers.shape[0]
                and float(distance[nearest_local].item()) <= self.support_radius_squared
            ):
                item = self._pending[nearest]
                value = observation.to(
                    device=item.center.device, dtype=item.center.dtype
                )
                new_count = item.count + 1
                delta = value - item.center
                new_center = item.center + delta / new_count
                item.m2 += delta * (value - new_center)
                item.center = new_center
                item.count = new_count
                item.last_seen = self._step
                if item.class_count is not None:
                    assert class_increment is not None
                    item.class_count += class_increment.to(item.class_count.device)
                if item.count >= self.persistence:
                    promoted = self._pending.pop(nearest)
                    return NoveltyProposal(
                        center=promoted.center,
                        count=promoted.count,
                        m2=promoted.m2,
                        class_count=promoted.class_count,
                    )
                return None
        if len(self._pending) >= self.max_pending:
            evict = min(
                range(len(self._pending)),
                key=lambda index: (
                    self._pending[index].count,
                    self._pending[index].last_seen,
                ),
            )
            self._pending.pop(evict)
        self._pending.append(
            _PendingNovelty(
                center=observation.detach().clone(),
                count=1,
                m2=torch.zeros_like(observation),
                last_seen=self._step,
                class_count=class_increment,
            )
        )
        return None

    def serializable(self) -> dict[str, object]:
        """Return CPU state sufficient to reconstruct this stager exactly."""
        count = len(self._pending)
        return {
            "step": self._step,
            "pending_center": (
                torch.stack([item.center for item in self._pending]).cpu()
                if count
                else torch.empty(0, self.dimension)
            ),
            "pending_count": torch.tensor(
                [item.count for item in self._pending], dtype=torch.long
            ),
            "pending_m2": (
                torch.stack([item.m2 for item in self._pending]).cpu()
                if count
                else torch.empty(0, self.dimension)
            ),
            "pending_last_seen": torch.tensor(
                [item.last_seen for item in self._pending], dtype=torch.long
            ),
            "pending_class_count": (
                torch.stack([item.class_count for item in self._pending]).cpu()
                if count and self.classes is not None
                else None
            ),
        }

    def load_state(self, payload: Mapping[str, object]) -> None:
        """Restore pending items and the step counter from `serializable()`."""
        self._step = int(payload["step"])
        centers = payload["pending_center"]
        counts = payload["pending_count"]
        m2s = payload["pending_m2"]
        last_seen = payload["pending_last_seen"]
        class_counts = payload["pending_class_count"]
        self._pending = [
            _PendingNovelty(
                center=centers[index].clone(),
                count=int(counts[index].item()),
                m2=m2s[index].clone(),
                last_seen=int(last_seen[index].item()),
                class_count=(
                    class_counts[index].clone() if class_counts is not None else None
                ),
            )
            for index in range(centers.shape[0])
        ]


@dataclass(frozen=True)
class OnlineUpdateResult:
    basin_ids: Tensor
    novel: Tensor
    created: Tensor
    capacity_rejected: Tensor
    squared_distance: Tensor


class OnlineBasinMemory:
    """Sequential foreground updater for a frozen latent representation."""

    def __init__(
        self,
        bank: BasinState,
        *,
        novelty_threshold: float,
        novelty_persistence: int = 3,
        novelty_support_radius: float,
        max_basins: int,
        novelty_max_pending: int = 256,
        novelty_max_age: int = 1024,
        class_count: Tensor | None = None,
        variance_scale: float = 1.0,
        variance_epsilon: float = 1e-4,
        rate_min: float = 1e-4,
        rate_max: float = 32.0,
    ) -> None:
        if novelty_threshold < 0:
            raise ValueError("novelty_threshold must be non-negative")
        if max_basins < bank.mu.shape[0]:
            raise ValueError("max_basins cannot be smaller than the current bank")
        if class_count is not None and (
            class_count.ndim != 2 or class_count.shape[0] != bank.mu.shape[0]
        ):
            raise ValueError("class_count must have shape [basins, classes]")
        self.bank = bank
        self.index = ExactMainDeltaIndex(bank.mu)
        self.novelty_threshold = novelty_threshold
        self.max_basins = max_basins
        self.class_count = (
            class_count.detach().clone() if class_count is not None else None
        )
        classes = None if class_count is None else class_count.shape[1]
        self.stager = NoveltyStager(
            dimension=bank.mu.shape[1],
            persistence=novelty_persistence,
            support_radius=novelty_support_radius,
            max_pending=novelty_max_pending,
            max_age=novelty_max_age,
            classes=classes,
        )
        self.variance_scale = variance_scale
        self.variance_epsilon = variance_epsilon
        self.rate_min = rate_min
        self.rate_max = rate_max

    @torch.no_grad()
    def predict_nearest_value(self, latent: Tensor) -> Tensor:
        """Predict from the current nearest basin's empirical class value."""
        if self.class_count is None:
            raise ValueError("value prediction requires class_count")
        ids = self.index.query(latent, 1).squeeze(1)
        values = self.class_count.to(latent.device)[ids]
        return values.argmax(dim=-1)

    @torch.no_grad()
    def predict_rbf_value(
        self,
        latent: Tensor,
        *,
        top_k: int = 4,
        temperature: float = 0.1,
    ) -> Tensor:
        """Predict from a Euclidean RBF mixture of current basin values."""
        if self.class_count is None:
            raise ValueError("value prediction requires class_count")
        if not 1 <= top_k <= self.index.size:
            raise ValueError("top_k must be in [1, bank size]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        ids = self.index.query(latent, top_k)
        records = self.bank.mu.to(device=latent.device, dtype=latent.dtype)[ids]
        squared_distance = (latent[:, None, :] - records).square().sum(dim=-1)
        weight = torch.softmax(-0.5 * squared_distance / temperature, dim=-1)
        count = self.class_count.to(latent.device)[ids].float()
        probability = count / count.sum(dim=-1, keepdim=True).clamp_min(1)
        mixture = (weight[..., None] * probability).sum(dim=1)
        return mixture.argmax(dim=-1)

    @torch.no_grad()
    def rebuild_main_index(self) -> None:
        """Publish current bank centers into the main index and clear delta."""
        self.index.rebuild(self.bank.mu)

    def serializable(self) -> dict[str, object]:
        """Return CPU state sufficient to reconstruct this memory exactly.

        Bundles the bank, value (class-count) state, and delta-index state
        the design's §5 groups as one versioned unit, plus enough novelty-
        stager and configuration state to resume online updates and produce
        results identical to an uninterrupted run.
        """
        return {
            "bank": {
                "mu": self.bank.mu.detach().cpu(),
                "raw_rate": self.bank.raw_rate.detach().cpu(),
                "bias": self.bank.bias.detach().cpu(),
                "count": self.bank.count.detach().cpu(),
                "m2": self.bank.m2.detach().cpu(),
                "usage": self.bank.usage.detach().cpu(),
                "version": self.bank.version,
            },
            "index": self.index.serializable(),
            "class_count": (
                self.class_count.detach().cpu()
                if self.class_count is not None
                else None
            ),
            "novelty_threshold": self.novelty_threshold,
            "max_basins": self.max_basins,
            "variance_scale": self.variance_scale,
            "variance_epsilon": self.variance_epsilon,
            "rate_min": self.rate_min,
            "rate_max": self.rate_max,
            "novelty_persistence": self.stager.persistence,
            "novelty_support_radius": self.stager.support_radius_squared**0.5,
            "novelty_max_pending": self.stager.max_pending,
            "novelty_max_age": self.stager.max_age,
            "stager": self.stager.serializable(),
        }

    @classmethod
    def from_serialized(cls, payload: Mapping[str, object]) -> OnlineBasinMemory:
        """Reconstruct an online memory exactly as `serializable()` captured it."""
        bank = BasinState.from_serialized(payload["bank"])
        memory = cls(
            bank,
            novelty_threshold=payload["novelty_threshold"],
            novelty_persistence=payload["novelty_persistence"],
            novelty_support_radius=payload["novelty_support_radius"],
            max_basins=payload["max_basins"],
            novelty_max_pending=payload["novelty_max_pending"],
            novelty_max_age=payload["novelty_max_age"],
            class_count=payload["class_count"],
            variance_scale=payload["variance_scale"],
            variance_epsilon=payload["variance_epsilon"],
            rate_min=payload["rate_min"],
            rate_max=payload["rate_max"],
        )
        memory.index = ExactMainDeltaIndex.from_serialized(payload["index"])
        memory.stager.load_state(payload["stager"])
        return memory

    @torch.no_grad()
    def update(
        self,
        observations: Tensor,
        labels: Tensor | None = None,
    ) -> OnlineUpdateResult:
        """Assign or stage each observation in stream order."""
        if observations.ndim != 2 or observations.shape[1] != self.bank.mu.shape[1]:
            raise ValueError("observations must have shape [batch, bank.dimension]")
        if (labels is None) != (self.class_count is None):
            raise ValueError("labels must be supplied exactly for class-aware memory")
        if labels is not None and labels.shape != (observations.shape[0],):
            raise ValueError("labels must have shape [batch]")
        basin_ids = []
        novel_flags = []
        created_flags = []
        rejected_flags = []
        distances = []
        for offset, observation in enumerate(observations):
            query = observation[None, :]
            winner = int(self.index.query(query, 1).item())
            current_center = self.bank.mu[winner].to(
                device=observation.device, dtype=observation.dtype
            )
            squared_distance = float(
                (observation - current_center).square().sum().item()
            )
            is_novel = squared_distance > self.novelty_threshold
            assigned = -1
            created = False
            rejected = False
            label = None if labels is None else int(labels[offset].item())
            if is_novel:
                proposal = self.stager.observe(observation, label=label)
                if proposal is not None:
                    if self.bank.mu.shape[0] >= self.max_basins:
                        rejected = True
                    else:
                        assigned = self.bank.append_statistics(
                            proposal.center,
                            proposal.count,
                            proposal.m2,
                            variance_scale=self.variance_scale,
                            epsilon=self.variance_epsilon,
                            rate_min=self.rate_min,
                            rate_max=self.rate_max,
                        )
                        if self.class_count is not None:
                            assert proposal.class_count is not None
                            self.class_count = torch.cat(
                                (
                                    self.class_count,
                                    proposal.class_count.to(self.class_count.device)[
                                        None, :
                                    ],
                                )
                            )
                        self.index.upsert(
                            torch.tensor([assigned], device=self.bank.mu.device),
                            self.bank.mu[assigned][None, :],
                        )
                        created = True
            else:
                assigned = winner
                self.bank.update_winners(
                    observation[None, :],
                    torch.tensor([winner], device=observation.device),
                    variance_scale=self.variance_scale,
                    epsilon=self.variance_epsilon,
                    rate_min=self.rate_min,
                    rate_max=self.rate_max,
                )
                if self.class_count is not None:
                    assert label is not None
                    if not 0 <= label < self.class_count.shape[1]:
                        raise ValueError("label is outside the class_count width")
                    self.class_count[winner, label] += 1
                self.index.upsert(
                    torch.tensor([winner], device=self.bank.mu.device),
                    self.bank.mu[winner][None, :],
                )
            basin_ids.append(assigned)
            novel_flags.append(is_novel)
            created_flags.append(created)
            rejected_flags.append(rejected)
            distances.append(squared_distance)
        device = observations.device
        return OnlineUpdateResult(
            basin_ids=torch.tensor(basin_ids, dtype=torch.long, device=device),
            novel=torch.tensor(novel_flags, dtype=torch.bool, device=device),
            created=torch.tensor(created_flags, dtype=torch.bool, device=device),
            capacity_rejected=torch.tensor(
                rejected_flags, dtype=torch.bool, device=device
            ),
            squared_distance=torch.tensor(
                distances, dtype=observations.dtype, device=device
            ),
        )


@torch.no_grad()
def calibrate_novelty_threshold(
    latent: Tensor,
    centers: Tensor,
    *,
    quantile: float = 0.99,
    chunk_size: int = 1024,
) -> float:
    """Calibrate a squared Euclidean novelty threshold on known latents."""
    if latent.ndim != 2 or centers.ndim != 2 or latent.shape[1] != centers.shape[1]:
        raise ValueError("latent and centers must have compatible [rows, d] shapes")
    if not 0 < quantile < 1 or chunk_size <= 0:
        raise ValueError("quantile must be in (0, 1) and chunk_size must be positive")
    index = ExactMainDeltaIndex(centers)
    distances = []
    for start in range(0, latent.shape[0], chunk_size):
        sample = latent[start : start + chunk_size]
        ids = index.query(sample, 1).squeeze(1)
        nearest = centers.to(device=sample.device, dtype=sample.dtype)[ids]
        distances.append((sample - nearest).square().sum(dim=-1).cpu())
    return float(torch.quantile(torch.cat(distances), quantile).item())
