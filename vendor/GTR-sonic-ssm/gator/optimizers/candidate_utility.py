"""Aggregate candidate-utility scoring for Procoim-S (Phase 2).

Implements the Procoim-S design spec section 11: replace the old per-item
AND-gate veto (every retained example must individually approve a candidate)
with a single aggregate utility score plus a tiny set of hard catastrophic
canaries. Ordinary replay regression contributes a soft, quadratic-above-
tolerance cost term instead of vetoing outright.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from gator.optimizers.procoim_s import ProcoimSConfig

__all__ = [
    "ReplayEvaluation",
    "CandidateUtilityResult",
    "soft_replay_regression_cost",
    "catastrophic_violation",
    "evaluate_candidate_utility",
]


@dataclass(frozen=True)
class ReplayEvaluation:
    """One replay item's loss under a candidate, versus its reference loss."""

    item_id: str
    loss_after: float
    reference_loss: float
    tolerance: float = 0.0
    catastrophic_ceiling: float = math.inf
    weight: float = 1.0
    is_canary: bool = False

    @property
    def regression(self) -> float:
        return self.loss_after - self.reference_loss


def soft_replay_regression_cost(evaluations: Sequence[ReplayEvaluation]) -> float:
    """Quadratic-above-tolerance cost, summed over non-canary replay items.

    Canaries are excluded here — they are handled by
    :func:`catastrophic_violation` instead, since they are hard vetoes, not
    a soft cost term (design spec section 11.4).
    """

    total = 0.0
    for evaluation in evaluations:
        if evaluation.is_canary:
            continue
        overage = max(0.0, evaluation.regression - evaluation.tolerance)
        total += evaluation.weight * overage * overage
    return total


def catastrophic_violation(evaluations: Sequence[ReplayEvaluation]) -> bool:
    """True if any canary — or any item past its catastrophic ceiling — fired.

    Ordinary (non-canary) replay items only veto via their catastrophic
    ceiling, which is expected to be a much larger bound than their ordinary
    tolerance (design spec section 11.2: ``tau[j]`` vs ``h[j]``).
    """

    for evaluation in evaluations:
        if evaluation.regression > evaluation.catastrophic_ceiling:
            return True
    return False


@dataclass(frozen=True)
class CandidateUtilityResult:
    current_gain: float
    replay_cost: float
    future_utility: float
    delta_norm: float
    protection_debt: float
    catastrophic: bool
    aggregate_utility: float
    accepted: bool
    rejection_reason: str | None = None


def evaluate_candidate_utility(
    *,
    current_gain: float,
    replay_evaluations: Sequence[ReplayEvaluation] = (),
    future_utility: float = 0.0,
    delta_norm: float = 0.0,
    protection_debt: float = 0.0,
    config: ProcoimSConfig | None = None,
) -> CandidateUtilityResult:
    """Score one candidate and decide accept/reject (design spec section 11.3).

    ``U = current_gain + future_weight * future_utility
          - replay_weight * replay_soft_cost
          - norm_weight * ||delta||^2
          - debt_weight * protection_debt``

    Accepted only when ``U > minimum_utility`` AND ``current_gain > 0`` AND
    no catastrophic canary violation AND every term is finite.
    """

    config = config or ProcoimSConfig()
    replay_cost = soft_replay_regression_cost(replay_evaluations)
    catastrophic = catastrophic_violation(replay_evaluations)

    aggregate = (
        current_gain
        + config.future_utility_weight * future_utility
        - config.replay_weight * replay_cost
        - config.delta_norm_weight * (delta_norm * delta_norm)
        - config.protection_debt_weight * protection_debt
    )

    finite = math.isfinite(aggregate)
    reason: str | None = None
    if catastrophic:
        reason = "catastrophic_canary_violation"
    elif not finite:
        reason = "non_finite_utility"
    elif current_gain <= 0:
        reason = "non_positive_current_gain"
    elif aggregate <= config.minimum_utility:
        reason = "aggregate_utility_below_minimum"

    accepted = reason is None

    return CandidateUtilityResult(
        current_gain=current_gain,
        replay_cost=replay_cost,
        future_utility=future_utility,
        delta_norm=delta_norm,
        protection_debt=protection_debt,
        catastrophic=catastrophic,
        aggregate_utility=aggregate if finite else float("-inf"),
        accepted=accepted,
        rejection_reason=reason,
    )
