"""Consolidation: committing provisional fast weights into theta_slow
(Procoim-S design spec section 14).

Reuses rather than duplicates: :func:`~gator.optimizers.fast_weights.apply_fast_delta`
for temporarily applying a scaled delta, :func:`~gator.optimizers.fast_weights.FastWeightState.commit_`
for the actual ``theta_slow += alpha*delta; delta_fast *= (1-alpha)`` rule
(spec section 14.5, already built in Phase 1), and
:func:`~gator.optimizers.candidate_utility.evaluate_candidate_utility` for
the accept/reject decision at each candidate alpha (Phase 2). This module's
own job is purely the line-search/componentwise-fallback orchestration
around those primitives (spec sections 14.3-14.6).

Model- and task-specific concerns (how to actually run a forward pass and
score replay items with a given delta applied) are deliberately kept out of
this module and supplied by the caller as callables — the same decoupling
``candidate_utility.py``/``store.py`` already use to stay independent of any
specific GTR-L configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from gator.optimizers.candidate_utility import (
    CandidateUtilityResult,
    ReplayEvaluation,
    evaluate_candidate_utility,
)
from gator.optimizers.fast_weights import FastWeightState, apply_fast_delta
from gator.optimizers.procoim_s import ProcoimSConfig

__all__ = [
    "apply_scaled_fast_delta",
    "ConsolidationResult",
    "GroupConsolidationResult",
    "consolidate",
    "consolidate_componentwise",
    "run_consolidation",
]


@contextmanager
def apply_scaled_fast_delta(
    model: nn.Module,
    state: FastWeightState,
    alpha: float,
    *,
    group: str | None = None,
) -> Iterator[None]:
    """Temporarily apply ``alpha * delta_fast`` (optionally restricted to one
    group) without mutating ``state`` — the line search's probing step
    (spec section 14.3: ``theta(alpha) = theta_slow + alpha * delta_fast``).
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0.0:
        yield
        return
    scoped = FastWeightState()
    if group is not None:
        delta = state.get_group(group)
        if delta is not None:
            scoped.add_group(delta, alpha=alpha)
    else:
        for group_name in state.groups:
            delta = state.get_group(group_name)
            if delta is not None:
                scoped.add_group(delta, alpha=alpha)
    with apply_fast_delta(model, scoped):
        yield


@dataclass(frozen=True)
class ConsolidationResult:
    committed: bool
    alpha: float
    utility: CandidateUtilityResult | None
    attempted_alphas: tuple[float, ...] = ()


@dataclass(frozen=True)
class GroupConsolidationResult:
    group: str
    result: ConsolidationResult


EvaluateFn = Callable[[], tuple[float, Sequence[ReplayEvaluation]]]


def consolidate(
    model: nn.Module,
    state: FastWeightState,
    *,
    evaluate: EvaluateFn,
    config: ProcoimSConfig,
    group: str | None = None,
) -> ConsolidationResult:
    """Line search over ``config.consolidation_line_search`` alphas, largest
    first, committing at the first alpha whose aggregate utility is accepted
    (spec section 14.3: "choose the largest alpha with positive aggregate
    utility and no catastrophic violation").

    ``evaluate`` is called once per candidate alpha with that alpha's scaled
    delta already applied to ``model`` (via :func:`apply_scaled_fast_delta`);
    it must return ``(current_gain, replay_evaluations)`` measured under
    that temporary state. Nothing is committed if every alpha is rejected —
    see :func:`run_consolidation` for the componentwise fallback and the
    section 14.6 failure response.
    """

    attempted: list[float] = []
    last_utility: CandidateUtilityResult | None = None
    for alpha in config.consolidation_line_search:
        with apply_scaled_fast_delta(model, state, alpha, group=group):
            current_gain, replay_evaluations = evaluate()
        result = evaluate_candidate_utility(
            current_gain=current_gain,
            replay_evaluations=replay_evaluations,
            config=config,
        )
        attempted.append(alpha)
        last_utility = result
        if result.accepted:
            state.commit_(model, alpha=alpha, group=group)
            return ConsolidationResult(
                committed=True,
                alpha=alpha,
                utility=result,
                attempted_alphas=tuple(attempted),
            )
    return ConsolidationResult(
        committed=False,
        alpha=0.0,
        utility=last_utility,
        attempted_alphas=tuple(attempted),
    )


def consolidate_componentwise(
    model: nn.Module,
    state: FastWeightState,
    *,
    evaluate_group: Callable[[str], tuple[float, Sequence[ReplayEvaluation]]],
    config: ProcoimSConfig,
    groups: Sequence[str] | None = None,
) -> tuple[GroupConsolidationResult, ...]:
    """Retry the line search independently per group (spec section 14.4):
    "decompose by parameter group... test the most useful components
    first... do not discard the entire window because one direction caused
    regression." Groups are tried in the order given (or ``state.groups``'
    insertion order), each with its own independent line search.
    """

    target_groups = list(groups) if groups is not None else list(state.groups)
    results = []
    for group in target_groups:
        group_result = consolidate(
            model,
            state,
            evaluate=lambda group=group: evaluate_group(group),
            config=config,
            group=group,
        )
        results.append(GroupConsolidationResult(group=group, result=group_result))
    return tuple(results)


def run_consolidation(
    model: nn.Module,
    state: FastWeightState,
    *,
    evaluate: EvaluateFn,
    evaluate_group: Callable[[str], tuple[float, Sequence[ReplayEvaluation]]] | None = None,
    config: ProcoimSConfig,
) -> tuple[ConsolidationResult, tuple[GroupConsolidationResult, ...]]:
    """Full consolidation pass: try the joint delta first, fall back to a
    componentwise pass on failure (if enabled), and shrink whatever is left
    on total failure (spec section 14.6: "shrink delta_fast" is the one
    concretely actionable item here without the repair-queue machinery of
    Procoim-S Phase 7, which this module does not yet build).

    Returns ``(joint_result, componentwise_results)`` — ``componentwise_results``
    is empty unless the joint attempt failed and componentwise fallback ran.
    """

    joint_result = consolidate(model, state, evaluate=evaluate, config=config)
    if joint_result.committed:
        return joint_result, ()

    if not config.allow_componentwise_commit or evaluate_group is None:
        _shrink_on_total_failure(state, config)
        return joint_result, ()

    componentwise_results = consolidate_componentwise(
        model, state, evaluate_group=evaluate_group, config=config
    )
    if not any(item.result.committed for item in componentwise_results):
        _shrink_on_total_failure(state, config)
    return joint_result, componentwise_results


@torch.no_grad()
def _shrink_on_total_failure(state: FastWeightState, config: ProcoimSConfig) -> None:
    state.decay_(config.fast_weight_decay)
