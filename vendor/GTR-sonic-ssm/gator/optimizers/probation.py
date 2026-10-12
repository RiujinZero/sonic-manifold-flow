"""Probation and delayed utility tracking for provisional updates (Procoim-S
design spec section 13).

Adapts the spec's ``ProvisionalUpdate`` (coefficients + direction IDs over a
learned direction dictionary — Procoim-S Phase 6, not yet built) to this
repo's actual Phase 1 fast-weight representation: a snapshotted
:class:`~gator.optimizers.fast_weights.FastWeightDelta` rather than a
coefficient vector. Everything else — future-utility EMA sampling,
context-helped/hurt bookkeeping, and the probation outcome policy — follows
the spec directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from gator.optimizers.fast_weights import FastWeightDelta, FastWeightState
from gator.optimizers.procoim_s import ProcoimSConfig

__all__ = ["ProvisionalUpdate", "ProbationOutcome", "ProbationTracker"]


@dataclass
class ProvisionalUpdate:
    update_id: int
    group: str
    delta: FastWeightDelta
    current_gain: float
    replay_cost: float
    surprise: float
    reliability: float
    opened_step: int
    expires_step: int
    future_utility_ema: float = 0.0
    future_trials: int = 0
    contexts_helped: int = 0
    contexts_hurt: int = 0

    def record_future_utility(self, utility: float, *, decay: float) -> None:
        """Fold one delayed-utility sample into the running EMA (spec section 13.1)."""

        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.future_utility_ema = decay * self.future_utility_ema + (1.0 - decay) * utility
        self.future_trials += 1
        if utility > 0:
            self.contexts_helped += 1
        elif utility < 0:
            self.contexts_hurt += 1

    def is_expired(self, current_step: int) -> bool:
        return current_step >= self.expires_step


class ProbationOutcome(str, Enum):
    """The spec's four outcomes (section 13.2), plus PENDING for "not enough
    evidence yet" — a bookkeeping necessity the spec doesn't name explicitly
    but that any real tracker needs before ``minimum_future_trials`` samples
    have accumulated."""

    PENDING = "pending"
    CONSOLIDATE = "consolidate"
    RETAIN = "retain"
    DECAY = "decay"
    ROLLBACK = "rollback"


class ProbationTracker:
    """Holds every open :class:`ProvisionalUpdate` and decides its outcome."""

    def __init__(self, config: ProcoimSConfig) -> None:
        self.config = config
        self._next_id = 0
        self.updates: dict[int, ProvisionalUpdate] = {}

    def add(
        self,
        *,
        group: str,
        delta: FastWeightDelta,
        current_gain: float,
        replay_cost: float,
        surprise: float,
        reliability: float,
        opened_step: int,
    ) -> ProvisionalUpdate:
        update = ProvisionalUpdate(
            update_id=self._next_id,
            group=group,
            delta=delta,
            current_gain=current_gain,
            replay_cost=replay_cost,
            surprise=surprise,
            reliability=reliability,
            opened_step=opened_step,
            expires_step=opened_step + self.config.provisional_expiry_examples,
        )
        self._next_id += 1
        self.updates[update.update_id] = update
        return update

    def remove(self, update_id: int) -> None:
        self.updates.pop(update_id, None)

    def should_sample_future_utility(
        self, *, generator: torch.Generator | None = None
    ) -> bool:
        """Sample delayed utility on a small fraction of examples (spec
        section 13.1: "do not run this dual evaluation on every example")."""

        return bool(torch.rand((), generator=generator).item() < self.config.future_utility_probe_rate)

    def decide(self, update: ProvisionalUpdate) -> ProbationOutcome:
        """The spec's four-way probation outcome (section 13.2)."""

        if update.future_trials < self.config.minimum_future_trials:
            return ProbationOutcome.PENDING
        if (
            update.future_utility_ema > 0
            and update.contexts_helped >= self.config.minimum_contexts_helped
        ):
            return ProbationOutcome.CONSOLIDATE
        if update.future_utility_ema < 0:
            return ProbationOutcome.ROLLBACK
        if update.contexts_helped >= 1 and update.contexts_hurt == 0:
            return ProbationOutcome.RETAIN
        return ProbationOutcome.DECAY

    @torch.no_grad()
    def apply_outcome(
        self,
        outcome: ProbationOutcome,
        update: ProvisionalUpdate,
        state: FastWeightState,
    ) -> None:
        """Mutate ``state`` per the decided outcome.

        CONSOLIDATE is deliberately a no-op here — committing a provisional
        delta into ``theta_slow`` is Procoim-S Phase 4 (consolidation.py)'s
        job, using ``FastWeightState.commit_``; this method only tracks the
        decision. PENDING and RETAIN also leave state untouched (probation
        continues, or the delta stays as a session-scoped fast weight).
        """

        if outcome is ProbationOutcome.DECAY:
            delta = state.get_group(update.group)
            if delta is not None:
                delta.scale_(self.config.fast_weight_decay)
        elif outcome is ProbationOutcome.ROLLBACK:
            state.remove_group(update.group)
            self.remove(update.update_id)

    def expire_stale(self, current_step: int) -> list[ProvisionalUpdate]:
        """Drop provisional updates past their expiry window without a
        resolved outcome (spec section 13's ``provisional_expiry_examples``)."""

        expired = [update for update in self.updates.values() if update.is_expired(current_step)]
        for update in expired:
            del self.updates[update.update_id]
        return expired
