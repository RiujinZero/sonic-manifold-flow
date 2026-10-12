"""Stratified replay reservoir for Procoim-S (Phase 2).

Replaces the old anchor-bank/consolidation AND-gate (every retained example
must approve every candidate) with a rotating, stratified sample that feeds
an *aggregate* utility score instead (see ``candidate_utility.py``). The
motivating finding (Papers/GTRL_PROCOIM_TRACE.md §5.2, and the Procoim-S
design spec §2.2/§12) is that a fixed intersection of hard constraints
collapses acceptance; a soft, rotating replay cost does not.

This module only holds and samples items — it does not compute surprise,
reliability, or forgetting-surprise; those are supplied by the caller
(eventually Phase 5's surprise/reliability signals) and stored on each
:class:`ReplayItem`.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import torch

__all__ = [
    "ReplayCategory",
    "ReplayItem",
    "ReplayReservoir",
]


class ReplayCategory(str, Enum):
    """The five strata from the Procoim-S design spec, section 12."""

    CONFIRMED = "confirmed"  # high-surprise, high-reliability confirmed events
    FRAGILE = "fragile"  # skills with high forgetting surprise
    FOUNDATION = "foundation"  # ordinary low-surprise foundational examples
    RECENT = "recent"  # recent examples
    RANDOM = "random"  # random coverage examples


#: Target composition from the design spec (section 12): 35/25/20/10/10.
DEFAULT_CATEGORY_FRACTIONS: dict[ReplayCategory, float] = {
    ReplayCategory.CONFIRMED: 0.35,
    ReplayCategory.FRAGILE: 0.25,
    ReplayCategory.FOUNDATION: 0.20,
    ReplayCategory.RECENT: 0.10,
    ReplayCategory.RANDOM: 0.10,
}


@dataclass
class ReplayItem:
    """One retained example, per Procoim-S design spec section 12.1."""

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    valid_mask: torch.Tensor
    reference_loss: float
    reference_slot_losses: torch.Tensor | None = None
    reference_logits: torch.Tensor | None = None
    surprise: float = 0.0
    reliability: float = 0.0
    forgetting_surprise: float = 0.0
    novelty: float = 0.0
    task_id: str = ""
    context_id: str = ""
    age: int = 0
    uses: int = 0
    regressions: int = 0
    importance: float = 0.0
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class ReplayReservoir:
    """Capacity-bounded, stratified store of :class:`ReplayItem` records.

    Each category (see :class:`ReplayCategory`) gets a capacity slice
    proportional to ``category_fractions`` (default: the spec's 35/25/20/10/10
    composition). Within CONFIRMED/FRAGILE/FOUNDATION/RECENT, an incoming
    item evicts the lowest-``importance`` resident (ties broken by oldest
    ``age``) once its category is full. RANDOM uses classic reservoir
    sampling instead, since its purpose is unbiased coverage rather than
    retaining the "best" examples.
    """

    def __init__(
        self,
        capacity: int = 256,
        *,
        category_fractions: dict[ReplayCategory, float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        fractions = category_fractions or DEFAULT_CATEGORY_FRACTIONS
        total_fraction = sum(fractions.values())
        if total_fraction <= 0 or abs(total_fraction - 1.0) > 1e-6:
            raise ValueError("category_fractions must sum to 1.0")
        self.capacity = capacity
        self._rng = rng or random.Random()
        self._category_capacity: dict[ReplayCategory, int] = {
            category: max(1, round(capacity * fraction))
            for category, fraction in fractions.items()
        }
        self._items: dict[ReplayCategory, list[ReplayItem]] = {
            category: [] for category in fractions
        }
        self._random_seen: dict[ReplayCategory, int] = {
            category: 0 for category in fractions
        }

    def __len__(self) -> int:
        return sum(len(items) for items in self._items.values())

    def items_in(self, category: ReplayCategory) -> tuple[ReplayItem, ...]:
        return tuple(self._items[category])

    def all_items(self) -> tuple[ReplayItem, ...]:
        return tuple(item for items in self._items.values() for item in items)

    def add(self, item: ReplayItem, category: ReplayCategory) -> None:
        """Insert ``item`` into ``category``, evicting if the slice is full."""

        bucket = self._items[category]
        capacity = self._category_capacity[category]

        if category is ReplayCategory.RANDOM:
            self._random_seen[category] += 1
            if len(bucket) < capacity:
                bucket.append(item)
                return
            replace_index = self._rng.randrange(self._random_seen[category])
            if replace_index < capacity:
                bucket[replace_index] = item
            return

        if len(bucket) < capacity:
            bucket.append(item)
            return

        worst_index = min(
            range(len(bucket)),
            key=lambda index: (bucket[index].importance, -bucket[index].age),
        )
        if item.importance >= bucket[worst_index].importance:
            bucket[worst_index] = item

    def tick_ages(self) -> None:
        """Advance every resident item's age by one (call once per step)."""

        for bucket in self._items.values():
            for item in bucket:
                item.age += 1

    def sample_stratified(
        self,
        *,
        confirmed: int = 0,
        fragile: int = 0,
        foundation: int = 0,
        recent: int = 0,
        random_coverage: int = 0,
    ) -> list[ReplayItem]:
        """Draw a fresh sample each call — rotation is inherent, not stored.

        Requests larger than a category's population fall back to sampling
        with replacement so a small reservoir still produces the requested
        batch size (see design spec section 12.2/12.3).
        """

        requested = {
            ReplayCategory.CONFIRMED: confirmed,
            ReplayCategory.FRAGILE: fragile,
            ReplayCategory.FOUNDATION: foundation,
            ReplayCategory.RECENT: recent,
            ReplayCategory.RANDOM: random_coverage,
        }
        sampled: list[ReplayItem] = []
        for category, count in requested.items():
            if count <= 0:
                continue
            bucket = self._items[category]
            if not bucket:
                continue
            if count <= len(bucket):
                sampled.extend(self._rng.sample(bucket, count))
            else:
                sampled.extend(self._rng.choices(bucket, k=count))
        for item in sampled:
            item.uses += 1
        return sampled

    def mark_regression(self, item_id: str) -> None:
        for bucket in self._items.values():
            for item in bucket:
                if item.item_id == item_id:
                    item.regressions += 1
                    return

    def update_reference(self, item_id: str, *, reference_loss: float) -> None:
        """Refresh an item's reference loss (e.g. after consolidation)."""

        for bucket in self._items.values():
            for item in bucket:
                if item.item_id == item_id:
                    item.reference_loss = reference_loss
                    return

    def category_of(self, item_id: str) -> ReplayCategory | None:
        for category, bucket in self._items.items():
            if any(item.item_id == item_id for item in bucket):
                return category
        return None
