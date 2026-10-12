"""Stable programmatic training API for Gator models."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .optimization import (
    OptimizationConfig,
    accumulate_gradients,
    apply_optimizer_step,
    build_optimizer,
    build_scheduler,
    optimization_step,
)
from .optimizers.procoim import (
    LossProvider,
    ProcoimAnchor,
    ProcoimAnchorAuditResult,
    ProcoimConfig,
    ProcoimEpisodeResult,
    ProcoimOptimizer,
)


@dataclass(frozen=True)
class StepResult:
    """Optimizer diagnostics returned after one update."""

    step: int
    loss: float
    learning_rate: float
    gradient_norm: float


class OptimizationEngine:
    """Own optimizer state and expose a model-independent update API."""

    def __init__(
        self,
        model: nn.Module,
        config: OptimizationConfig,
        *,
        parameter_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self._named_parameters = list(model.named_parameters())
        self._name_by_parameter_object = {
            id(parameter): name for name, parameter in self._named_parameters
        }
        optimizer_parameters = (
            model.parameters()
            if parameter_groups is None
            else parameter_groups
        )
        self.optimizer = build_optimizer(optimizer_parameters, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.step_number = 0
        self.last_optimizer_state_migration: tuple[int, int] | None = None

    def _parameter_names_by_group(self) -> list[list[str]]:
        return [
            [
                self._name_by_parameter_object[id(parameter)]
                for parameter in group["params"]
            ]
            for group in self.optimizer.param_groups
        ]

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self, loss: torch.Tensor) -> StepResult:
        try:
            gradient_norm = optimization_step(
                loss=loss,
                model=self.model,
                optimizer=self.optimizer,
                config=self.config,
                scheduler=self.scheduler,
            )
        except FloatingPointError as error:
            raise FloatingPointError(
                f"optimizer step {self.step_number + 1} at "
                f"lr={self.learning_rate:.9g}: {error}"
            ) from error
        self.step_number += 1
        return StepResult(
            step=self.step_number,
            loss=float(loss.detach()),
            learning_rate=self.learning_rate,
            gradient_norm=gradient_norm,
        )

    def zero_grad(self) -> None:
        """Clear gradients before starting a new accumulation window."""

        self.optimizer.zero_grad(set_to_none=True)

    def accumulate(self, loss: torch.Tensor, *, scale: float = 1.0) -> None:
        """Backpropagate one micro-batch of a gradient-accumulation window.

        Call :meth:`zero_grad` once before the first micro-batch of a window,
        this once per micro-batch, and :meth:`apply_accumulated_step` once
        after the last micro-batch.
        """

        try:
            accumulate_gradients(loss, scale=scale)
        except FloatingPointError as error:
            raise FloatingPointError(
                f"gradient accumulation before step {self.step_number + 1} "
                f"at lr={self.learning_rate:.9g}: {error}"
            ) from error

    def apply_accumulated_step(self, loss: torch.Tensor) -> StepResult:
        """Clip, update, and advance state after a full accumulation window.

        ``loss`` is only used for the returned diagnostic (typically the
        final micro-batch's loss, or a running mean) — gradients must
        already be populated via prior :meth:`accumulate` calls.
        """

        try:
            gradient_norm = apply_optimizer_step(
                model=self.model,
                optimizer=self.optimizer,
                config=self.config,
                scheduler=self.scheduler,
            )
        except FloatingPointError as error:
            raise FloatingPointError(
                f"optimizer step {self.step_number + 1} at "
                f"lr={self.learning_rate:.9g}: {error}"
            ) from error
        self.step_number += 1
        return StepResult(
            step=self.step_number,
            loss=float(loss.detach()),
            learning_rate=self.learning_rate,
            gradient_norm=gradient_norm,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_number,
            "config": self.config.to_dict(),
            "optimizer": self.optimizer.state_dict(),
            "parameter_names_by_group": self._parameter_names_by_group(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        learning_rate: float | None = None,
        migrate_optimizer_by_name: bool = False,
        preserve_group_lr_multipliers: bool = False,
    ) -> None:
        """Restore training state, optionally beginning a fresh LR schedule.

        Loading optimizer moments while leaving ``load_scheduler`` false is the
        continuation-training path: Adam history survives, the newly
        configured scheduler starts at zero, and ``learning_rate`` explicitly
        sets the first LR of the new training phase.
        """

        if learning_rate is not None and learning_rate <= 0:
            raise ValueError("learning_rate must be positive or None")
        self.last_optimizer_state_migration = None
        if load_optimizer and migrate_optimizer_by_name:
            source_optimizer = state["optimizer"]
            source_name_groups = state.get("parameter_names_by_group")
            if source_name_groups is None:
                if len(source_optimizer["param_groups"]) != 1:
                    raise RuntimeError(
                        "optimizer checkpoint lacks parameter names and has "
                        "multiple parameter groups"
                    )
                source_names = [
                    name for name, _ in self._named_parameters
                ]
            else:
                source_names = [
                    str(name)
                    for group_names in source_name_groups
                    for name in group_names
                ]
            source_ids = [
                parameter_id
                for group in source_optimizer["param_groups"]
                for parameter_id in group["params"]
            ]
            if len(source_names) != len(source_ids):
                raise RuntimeError(
                    "optimizer checkpoint parameter names do not match its "
                    "parameter IDs"
                )
            source_id_by_name = dict(
                zip(source_names, source_ids, strict=True)
            )
            current_template = self.optimizer.state_dict()
            current_ids = [
                parameter_id
                for group in current_template["param_groups"]
                for parameter_id in group["params"]
            ]
            current_names = [
                name
                for group_names in self._parameter_names_by_group()
                for name in group_names
            ]
            translated_state = {}
            migrated = 0
            for name, current_id in zip(
                current_names,
                current_ids,
                strict=True,
            ):
                source_id = source_id_by_name.get(name)
                if (
                    source_id is not None
                    and source_id in source_optimizer["state"]
                ):
                    translated_state[current_id] = copy.deepcopy(
                        source_optimizer["state"][source_id]
                    )
                    migrated += 1
            self.optimizer.load_state_dict(
                {
                    "state": translated_state,
                    # Preserve the newly requested group topology and scalar
                    # hyperparameters while migrating only tensor moments.
                    "param_groups": current_template["param_groups"],
                }
            )
            self.last_optimizer_state_migration = (
                migrated,
                len(current_names),
            )
        elif load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        scheduler_state = state.get("scheduler")
        if (
            load_scheduler
            and self.scheduler is not None
            and scheduler_state is not None
        ):
            self.scheduler.load_state_dict(scheduler_state)
        if learning_rate is not None:
            for parameter_group in self.optimizer.param_groups:
                multiplier = (
                    float(parameter_group.get("lr_multiplier", 1.0))
                    if preserve_group_lr_multipliers
                    else 1.0
                )
                parameter_group["lr"] = learning_rate * multiplier
        self.step_number = int(state.get("step", 0))


class ProcoimTrainingEngine:
    """Own a forward-only Procoim optimizer and count forwards as steps.

    Unlike :class:`OptimizationEngine`, this engine receives a fixed batch
    plus a slot-loss provider because candidate selection must run the model
    repeatedly. ``step_number`` is the cumulative number of Procoim search
    forwards, not the number of episodes or accepted parameter updates.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ProcoimConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = ProcoimOptimizer(config)
        self.step_number = 0

    @property
    def episode_number(self) -> int:
        return self.optimizer.episodes

    @property
    def accepted_updates(self) -> int:
        return self.optimizer.total_accepted_updates

    def run_episode(
        self,
        batch: Any,
        loss_provider: LossProvider,
        *,
        forward_kwargs: dict[str, Any] | None = None,
        example_id: Any | None = None,
        anchors: tuple[ProcoimAnchor, ...] = (),
        accumulation_batches: tuple[Any, ...] = (),
        reference_batch: Any | None = None,
        reference_forward_kwargs: dict[str, Any] | None = None,
        reference_reduction_fraction: float | None = None,
        reference_batch_provider: Callable[[], Any] | None = None,
        reference_rotation_forwards: int | None = None,
        max_forward_calls: int | None = None,
    ) -> ProcoimEpisodeResult:
        try:
            result = self.optimizer.run_episode(
                self.model,
                batch,
                loss_provider,
                forward_kwargs=forward_kwargs,
                example_id=example_id,
                anchors=anchors,
                accumulation_batches=accumulation_batches,
                reference_batch=reference_batch,
                reference_forward_kwargs=reference_forward_kwargs,
                reference_reduction_fraction=reference_reduction_fraction,
                reference_batch_provider=reference_batch_provider,
                reference_rotation_forwards=reference_rotation_forwards,
                max_forward_calls=max_forward_calls,
            )
        finally:
            self.step_number = self.optimizer.total_forward_calls
        return result

    def audit_anchors(
        self,
        anchors: tuple[ProcoimAnchor, ...],
        loss_provider: LossProvider,
        *,
        forward_kwargs: dict[str, Any] | None = None,
    ) -> ProcoimAnchorAuditResult:
        """Run a deterministic retention audit and count every forward."""

        try:
            return self.optimizer.audit_anchors(
                self.model,
                loss_provider,
                anchors,
                forward_kwargs=forward_kwargs,
            )
        finally:
            self.step_number = self.optimizer.total_forward_calls

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_number,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        saved_step = int(state.get("step", self.optimizer.total_forward_calls))
        if saved_step != self.optimizer.total_forward_calls:
            raise ValueError(
                "Procoim engine step must equal optimizer forward-call count"
            )
        self.step_number = saved_step


@dataclass(frozen=True)
class ProcoimConsolidationResult:
    """Outcome of one provisional theta_base + delta_work audit."""

    committed_alpha: float
    audit_forwards: int
    audit_penalty: float
    hard_violation: bool
    rolled_back: bool
    audit_retained_reduction: float | None = None


class ProcoimRetentionWindow:
    """Stage accepted Procoim updates and consolidate only safe windows.

    The model itself is ``theta_live``. A full FP32 snapshot stores
    ``theta_base``; their difference is the implicit ``delta_work``. At
    consolidation, the largest safe forward-audited interpolation is kept.
    """

    def __init__(
        self,
        engine: ProcoimTrainingEngine,
        *,
        interval_examples: int = 16,
        commit_alphas: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
        shrink_factor: float = 0.5,
    ) -> None:
        if interval_examples <= 0:
            raise ValueError("interval_examples must be positive")
        if not commit_alphas:
            raise ValueError("commit_alphas cannot be empty")
        if any(not 0 < alpha <= 1 for alpha in commit_alphas):
            raise ValueError("commit alphas must be in (0, 1]")
        if any(
            left <= right
            for left, right in zip(commit_alphas, commit_alphas[1:])
        ):
            raise ValueError("commit alphas must strictly decrease")
        if not 0 < shrink_factor <= 1:
            raise ValueError("shrink_factor must be in (0, 1]")
        self.engine = engine
        self.model = engine.model
        self.interval_examples = interval_examples
        factors = list(commit_alphas)
        if shrink_factor not in factors:
            factors.append(shrink_factor)
            factors.sort(reverse=True)
        self.commit_alphas = tuple(factors)
        self.shrink_factor = shrink_factor
        self.examples_in_window = 0
        self.windows_committed = 0
        self.windows_rolled_back = 0
        self._base = self._snapshot_parameters()

    @torch.no_grad()
    def _snapshot_parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

    @torch.no_grad()
    def _restore_parameters(
        self,
        snapshot: Mapping[str, torch.Tensor],
    ) -> None:
        current = dict(self.model.named_parameters())
        if current.keys() != snapshot.keys():
            raise RuntimeError("model parameter layout changed within a window")
        for name, value in snapshot.items():
            current[name].copy_(value)

    @torch.no_grad()
    def _interpolate_parameters(
        self,
        live: Mapping[str, torch.Tensor],
        alpha: float,
    ) -> None:
        current = dict(self.model.named_parameters())
        if current.keys() != self._base.keys() or current.keys() != live.keys():
            raise RuntimeError("model parameter layout changed within a window")
        for name, parameter in current.items():
            parameter.copy_(
                self._base[name] + alpha * (live[name] - self._base[name])
            )

    def record_episode(self) -> None:
        self.examples_in_window += 1

    @property
    def due(self) -> bool:
        return self.examples_in_window >= self.interval_examples

    @property
    def has_provisional_updates(self) -> bool:
        return self.examples_in_window > 0

    @torch.no_grad()
    def consolidate(
        self,
        anchors: tuple[ProcoimAnchor, ...],
        loss_provider: LossProvider,
        *,
        forward_kwargs: dict[str, Any] | None = None,
        max_forward_calls: int | None = None,
    ) -> ProcoimConsolidationResult:
        """Commit the largest safe alpha, or restore theta_base exactly."""

        if max_forward_calls is not None and max_forward_calls < 0:
            raise ValueError("max_forward_calls must be nonnegative or None")
        live = self._snapshot_parameters()
        audit_forwards = 0
        selected_alpha = 0.0
        selected_penalty = 0.0
        last_violation = False
        retained_reduction: float | None = None

        if not anchors:
            selected_alpha = 1.0
        else:
            required = len(anchors)
            for alpha in self.commit_alphas:
                if (
                    max_forward_calls is not None
                    and audit_forwards + required > max_forward_calls
                ):
                    break
                self._interpolate_parameters(live, alpha)
                audit = self.engine.audit_anchors(
                    anchors,
                    loss_provider,
                    forward_kwargs=forward_kwargs,
                )
                audit_forwards += audit.anchor_forwards
                selected_penalty = audit.penalty
                last_violation = audit.hard_violation
                retained_reduction = audit.retained_reduction
                # Soft replay tolerance shapes candidate utility; it is not a
                # second hard boundary. Consolidation must preserve the same
                # bounded-regression contract used during search, otherwise
                # an update accepted between the soft and hard tolerances is
                # guaranteed to be rolled back here. The penalty remains
                # visible for audit/selection diagnostics.
                if not audit.hard_violation:
                    selected_alpha = alpha
                    break

        rolled_back = selected_alpha == 0.0
        if rolled_back:
            self._restore_parameters(self._base)
            self.windows_rolled_back += 1
        else:
            # The model is already at the selected interpolation (or at live
            # when there were no anchors). This becomes the next trusted
            # theta_base and delta_work returns to zero.
            if not anchors:
                self._restore_parameters(live)
            self._base = self._snapshot_parameters()
            self.windows_committed += 1
        if rolled_back:
            self._base = self._snapshot_parameters()
        self.examples_in_window = 0
        return ProcoimConsolidationResult(
            committed_alpha=selected_alpha,
            audit_forwards=audit_forwards,
            audit_penalty=selected_penalty,
            hard_violation=last_violation,
            rolled_back=rolled_back,
            audit_retained_reduction=retained_reduction,
        )

    @torch.no_grad()
    def rollback_unconsolidated(self) -> None:
        """Restore theta_base when no forward budget remains for an audit."""

        if self.has_provisional_updates:
            self._restore_parameters(self._base)
            self.examples_in_window = 0
            self.windows_rolled_back += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "interval_examples": self.interval_examples,
            "commit_alphas": self.commit_alphas,
            "shrink_factor": self.shrink_factor,
            "examples_in_window": self.examples_in_window,
            "windows_committed": self.windows_committed,
            "windows_rolled_back": self.windows_rolled_back,
            "base_parameters": {
                name: value.detach().clone()
                for name, value in self._base.items()
            },
        }
