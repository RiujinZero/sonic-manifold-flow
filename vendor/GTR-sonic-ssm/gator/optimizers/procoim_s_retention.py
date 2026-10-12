"""Aggregate-utility-based Procoim retention window (Procoim-S integration).

An alternative to ``gator.training.ProcoimRetentionWindow``'s anchor-hard-
audit consolidation. Tonight's earlier investigation found that a hard
per-item AND-gate (every retained anchor must individually approve a
candidate) collapses acceptance to a handful of updates regardless of how
far other thresholds are loosened — that is exactly what
``ProcoimRetentionWindow``'s anchor audit does, and it has been disabled
(``--procoim-consolidation-interval-examples 0``) in every run tonight as a
result.

This class keeps the same snapshot/interpolate/restore/line-search
structure (``theta_base`` + implicit ``delta_work``, largest safe alpha
wins) but replaces the anchor audit with Procoim-S's aggregate utility
scorer (``candidate_utility.evaluate_candidate_utility``) against a
stratified replay reservoir sampled from recent episodes — the mechanism
that, in Procoim-S's own test suite, was shown to accept updates the old
AND-gate rejected.

Same external shape as ``ProcoimRetentionWindow`` (``due``,
``has_provisional_updates``, ``windows_committed``/``windows_rolled_back``,
returns a ``gator.training.ProcoimConsolidationResult``) so it is a
drop-in alternative at the call site in ``gator_linear_prefix_train.py``,
selected by a new ``--procoim-consolidation-mode utility`` flag — the
default stays ``anchor``, so every existing run's behavior is unchanged
unless this is explicitly requested.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from gator.continual.replay_reservoir import ReplayCategory, ReplayItem, ReplayReservoir
from gator.optimizers.candidate_utility import ReplayEvaluation, evaluate_candidate_utility
from gator.optimizers.procoim_s import ProcoimSConfig
from gator.training import ProcoimConsolidationResult, ProcoimTrainingEngine

__all__ = ["ProcoimSRetentionWindow"]

ScalarLossFn = Callable[[nn.Module, tuple[Tensor, Tensor], dict[str, Any]], float]


class ProcoimSRetentionWindow:
    def __init__(
        self,
        engine: ProcoimTrainingEngine,
        *,
        interval_examples: int,
        commit_alphas: tuple[float, ...],
        replay_reservoir: ReplayReservoir,
        config: ProcoimSConfig,
        replay_sample_size: int = 4,
    ) -> None:
        if interval_examples <= 0:
            raise ValueError("interval_examples must be positive")
        if not commit_alphas:
            raise ValueError("commit_alphas cannot be empty")
        if any(not 0 < alpha <= 1 for alpha in commit_alphas):
            raise ValueError("commit alphas must be in (0, 1]")
        if replay_sample_size < 0:
            raise ValueError("replay_sample_size must be nonnegative")
        self.engine = engine
        self.model = engine.model
        self.interval_examples = interval_examples
        self.commit_alphas = commit_alphas
        self.replay_reservoir = replay_reservoir
        self.config = config
        self.replay_sample_size = replay_sample_size
        self.examples_in_window = 0
        self.windows_committed = 0
        self.windows_rolled_back = 0
        self._base = self._snapshot_parameters()
        self._current_batch: tuple[Tensor, Tensor] | None = None
        self._current_forward_kwargs: dict[str, Any] = {}
        self._current_example_id: str | None = None

    @torch.no_grad()
    def _snapshot_parameters(self) -> dict[str, Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}

    @torch.no_grad()
    def _restore_parameters(self, snapshot: Mapping[str, Tensor]) -> None:
        current = dict(self.model.named_parameters())
        if current.keys() != snapshot.keys():
            raise RuntimeError("model parameter layout changed within a window")
        for name, value in snapshot.items():
            current[name].copy_(value)

    @torch.no_grad()
    def _interpolate_parameters(self, live: Mapping[str, Tensor], alpha: float) -> None:
        current = dict(self.model.named_parameters())
        if current.keys() != self._base.keys() or current.keys() != live.keys():
            raise RuntimeError("model parameter layout changed within a window")
        for name, parameter in current.items():
            parameter.copy_(self._base[name] + alpha * (live[name] - self._base[name]))

    def record_episode(
        self,
        *,
        batch: tuple[Tensor, Tensor] | None = None,
        forward_kwargs: dict[str, Any] | None = None,
        example_id: str | None = None,
    ) -> None:
        self.examples_in_window += 1
        if batch is None:
            return
        self._current_batch = batch
        self._current_forward_kwargs = forward_kwargs or {}
        self._current_example_id = example_id
        observed, targets = batch
        self.replay_reservoir.add(
            ReplayItem(
                input_ids=observed.detach().clone(),
                target_ids=targets.detach().clone(),
                valid_mask=torch.ones_like(targets, dtype=torch.bool),
                reference_loss=0.0,
                task_id=example_id or "",
            ),
            ReplayCategory.RECENT,
        )

    @property
    def due(self) -> bool:
        return self.examples_in_window >= self.interval_examples

    @property
    def has_provisional_updates(self) -> bool:
        return self.examples_in_window > 0

    @torch.no_grad()
    def consolidate(
        self,
        *,
        scalar_loss_fn: ScalarLossFn,
    ) -> ProcoimConsolidationResult:
        """Commit the largest alpha whose aggregate utility is accepted, or
        restore ``theta_base`` exactly (mirrors
        ``ProcoimRetentionWindow.consolidate``'s contract)."""

        live = self._snapshot_parameters()

        if self._current_batch is None:
            # Nothing accumulated to score against this window -- matches
            # ProcoimRetentionWindow's own "no anchors" fallback of trusting
            # the live state outright.
            self._restore_parameters(live)
            self._base = self._snapshot_parameters()
            self.windows_committed += 1
            self.examples_in_window = 0
            return ProcoimConsolidationResult(
                committed_alpha=1.0,
                audit_forwards=0,
                audit_penalty=0.0,
                hard_violation=False,
                rolled_back=False,
            )

        audit_forwards = 0
        self._restore_parameters(self._base)
        base_current_loss = scalar_loss_fn(self.model, self._current_batch, self._current_forward_kwargs)
        audit_forwards += 1

        replay_sample = self.replay_reservoir.sample_stratified(
            recent=self.replay_sample_size,
            confirmed=self.replay_sample_size,
        )
        base_replay_losses = [
            scalar_loss_fn(self.model, (item.input_ids, item.target_ids), self._current_forward_kwargs)
            for item in replay_sample
        ]
        audit_forwards += len(replay_sample)

        selected_alpha = 0.0
        selected_utility = None
        for alpha in self.commit_alphas:
            self._interpolate_parameters(live, alpha)
            current_loss = scalar_loss_fn(self.model, self._current_batch, self._current_forward_kwargs)
            audit_forwards += 1
            current_gain = base_current_loss - current_loss

            replay_evaluations = []
            for item, reference_loss in zip(replay_sample, base_replay_losses, strict=True):
                loss_after = scalar_loss_fn(
                    self.model, (item.input_ids, item.target_ids), self._current_forward_kwargs
                )
                audit_forwards += 1
                replay_evaluations.append(
                    ReplayEvaluation(
                        item_id=item.item_id,
                        loss_after=loss_after,
                        reference_loss=reference_loss,
                        tolerance=self.config.replay_soft_tolerance,
                        catastrophic_ceiling=(
                            self.config.replay_soft_tolerance
                            + self.config.replay_catastrophic_tolerance
                        ),
                    )
                )

            result = evaluate_candidate_utility(
                current_gain=current_gain,
                replay_evaluations=replay_evaluations,
                config=self.config,
            )
            if result.accepted:
                selected_alpha = alpha
                selected_utility = result
                break

        rolled_back = selected_alpha == 0.0
        if rolled_back:
            self._restore_parameters(self._base)
            self.windows_rolled_back += 1
        else:
            self._interpolate_parameters(live, selected_alpha)
            self._base = self._snapshot_parameters()
            self.windows_committed += 1

        self.examples_in_window = 0
        self._current_batch = None
        return ProcoimConsolidationResult(
            committed_alpha=selected_alpha,
            audit_forwards=audit_forwards,
            audit_penalty=(selected_utility.replay_cost if selected_utility else 0.0),
            hard_violation=(selected_utility.catastrophic if selected_utility else False),
            rolled_back=rolled_back,
        )

    @torch.no_grad()
    def rollback_unconsolidated(self) -> None:
        if self.has_provisional_updates:
            self._restore_parameters(self._base)
            self.examples_in_window = 0
            self.windows_rolled_back += 1
            self._current_batch = None
