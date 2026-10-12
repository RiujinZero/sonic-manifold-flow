"""Learned direction dictionary (Procoim-S design spec section 9).

Tracks per-group direction metadata — category (anchor/recent/repair/
exploration), utility EMA, age, and the contexts where a direction proved
useful — so the search basis can become a *learned* memory of what has
worked, not just a sequence of disposable random spans.

This module manages the bookkeeping and refresh-trigger policy only. It
does not itself regenerate perturbation tensors — that machinery already
exists in ``gator/optimizers/procoim.py``'s seeded rotating-basis generation
(``_rotating_factorized_direction``); wiring this dictionary's classification
into that live generation path is a further integration step, deliberately
not done in this pass (the same scoping choice SGBM's benchmark harness
made in Phase G: build and test the mechanism, don't silently claim an
integration that would need its own careful, separately-reviewed surgery on
shared code).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "DirectionCategory",
    "DirectionRecord",
    "DirectionDictionary",
    "RefreshSignals",
    "should_refresh_basis",
]


class DirectionCategory(str, Enum):
    ANCHOR = "anchor"
    RECENT = "recent"
    REPAIR = "repair"
    EXPLORATION = "exploration"


@dataclass
class DirectionRecord:
    direction_id: str
    category: DirectionCategory
    utility: float = 0.0
    age: int = 0
    contexts: set[str] = field(default_factory=set)
    catastrophic_regressions: int = 0
    trials: int = 0

    def record_trial(
        self,
        *,
        validated_gain: float,
        forwards_used: int,
        context: str | None = None,
        catastrophic: bool = False,
        utility_ema: float,
        eps: float = 1e-8,
    ) -> None:
        """``utility_k <- beta_u * utility_k + (1 - beta_u) * (gain / (forwards + eps))``
        (design spec section 9.2)."""

        if not 0.0 <= utility_ema <= 1.0:
            raise ValueError("utility_ema must be in [0, 1]")
        if forwards_used < 0:
            raise ValueError("forwards_used must be nonnegative")
        per_forward_gain = validated_gain / (forwards_used + eps)
        self.utility = utility_ema * self.utility + (1.0 - utility_ema) * per_forward_gain
        self.trials += 1
        if context is not None:
            self.contexts.add(context)
        if catastrophic:
            self.catastrophic_regressions += 1


class DirectionDictionary:
    """Per-group catalog of :class:`DirectionRecord`, one dictionary instance
    shared across all of Procoim's parameter groups."""

    def __init__(
        self,
        *,
        anchor_fraction: float = 0.50,
        recent_fraction: float = 0.25,
        exploration_fraction: float = 0.25,
        utility_ema: float = 0.95,
    ) -> None:
        fractions = (anchor_fraction, recent_fraction, exploration_fraction)
        if any(value < 0 for value in fractions):
            raise ValueError("fractions must be nonnegative")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("anchor/recent/exploration fractions must sum to 1.0")
        if not 0.0 <= utility_ema <= 1.0:
            raise ValueError("utility_ema must be in [0, 1]")
        self.anchor_fraction = anchor_fraction
        self.recent_fraction = recent_fraction
        self.exploration_fraction = exploration_fraction
        self.utility_ema = utility_ema
        self._records: dict[str, dict[str, DirectionRecord]] = {}

    def register(
        self, group: str, direction_id: str, category: DirectionCategory
    ) -> DirectionRecord:
        bucket = self._records.setdefault(group, {})
        record = bucket.get(direction_id)
        if record is None:
            record = DirectionRecord(direction_id=direction_id, category=category)
            bucket[direction_id] = record
        else:
            record.category = category
        return record

    def get(self, group: str, direction_id: str) -> DirectionRecord | None:
        return self._records.get(group, {}).get(direction_id)

    def record_trial(
        self,
        group: str,
        direction_id: str,
        *,
        validated_gain: float,
        forwards_used: int,
        context: str | None = None,
        catastrophic: bool = False,
    ) -> DirectionRecord:
        record = self.get(group, direction_id)
        if record is None:
            raise KeyError(f"direction {direction_id!r} not registered for group {group!r}")
        record.record_trial(
            validated_gain=validated_gain,
            forwards_used=forwards_used,
            context=context,
            catastrophic=catastrophic,
            utility_ema=self.utility_ema,
        )
        return record

    def tick_age(self, group: str) -> None:
        for record in self._records.get(group, {}).values():
            record.age += 1

    def composition(self, group: str) -> dict[DirectionCategory, tuple[str, ...]]:
        bucket = self._records.get(group, {})
        result: dict[DirectionCategory, list[str]] = {category: [] for category in DirectionCategory}
        for direction_id, record in bucket.items():
            result[record.category].append(direction_id)
        return {category: tuple(ids) for category, ids in result.items()}

    def retire_lowest_utility(self, group: str, *, keep: int) -> tuple[str, ...]:
        """Drop all but the ``keep`` highest-utility directions in ``group``,
        returning the retired IDs."""

        if keep < 0:
            raise ValueError("keep must be nonnegative")
        bucket = self._records.get(group, {})
        ranked = sorted(bucket.values(), key=lambda record: record.utility, reverse=True)
        retired = tuple(record.direction_id for record in ranked[keep:])
        for direction_id in retired:
            del bucket[direction_id]
        return retired


@dataclass(frozen=True)
class RefreshSignals:
    """Inputs to the stall-triggered refresh decision (design spec section 9.3)."""

    consecutive_stalls: int = 0
    stall_threshold: int | None = None
    gain_per_forward: float | None = None
    gain_per_forward_floor: float | None = None
    sustained_surprise: float | None = None
    surprise_threshold: float | None = None
    protected_debt: float | None = None
    protected_debt_threshold: float | None = None
    span_novelty: float | None = None
    span_novelty_floor: float | None = None


def should_refresh_basis(signals: RefreshSignals) -> bool:
    """Section 9.3: refresh on repeated stalls, poor gain-per-forward,
    sustained surprise, high protected debt, or low span novelty — any one
    of these firing is sufficient ("multiple signals agree" describes the
    typical case, not a requirement that all must fire together). Explicitly
    NOT triggered by a forward-count schedule alone (that is the separate,
    already-built forwards-based rotation mode in procoim.py).
    """

    if (
        signals.stall_threshold is not None
        and signals.consecutive_stalls >= signals.stall_threshold
    ):
        return True
    if (
        signals.gain_per_forward is not None
        and signals.gain_per_forward_floor is not None
        and signals.gain_per_forward < signals.gain_per_forward_floor
    ):
        return True
    if (
        signals.sustained_surprise is not None
        and signals.surprise_threshold is not None
        and signals.sustained_surprise >= signals.surprise_threshold
    ):
        return True
    if (
        signals.protected_debt is not None
        and signals.protected_debt_threshold is not None
        and signals.protected_debt >= signals.protected_debt_threshold
    ):
        return True
    if (
        signals.span_novelty is not None
        and signals.span_novelty_floor is not None
        and signals.span_novelty < signals.span_novelty_floor
    ):
        return True
    return False
