"""Progressive Continuous Improvement (Procoim) forward-only optimizer.

Procoim repeatedly probes structured parameter perturbations on one fixed
example or microbatch. It accepts only forward-evaluated changes that improve
unresolved output slots while preserving progress anchored by solved slots.
It never constructs an autograd graph and never calls backward.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import IntEnum
import math
import random
import time
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .slot_loss import SlotLossResult, batch_mean_slot_loss


CompletionMode = Literal["all_slots", "aggregate_and_fraction", "budget"]
SlotObjectiveReduction = Literal["per_example", "batch_mean"]
ParameterGroupGranularity = Literal["semantic", "relay", "tensor"]
DirectionalBasisMode = Literal[
    "fixed_rademacher",
    "rotating_factorized",
]
CandidateSource = Literal[
    "random",
    "momentum",
    "antithetic_estimate",
    "directional_adam",
]
LossProvider = Callable[
    [nn.Module, Any, Mapping[str, Any]],
    SlotLossResult,
]
ReferenceBatchProvider = Callable[[], Any]


@dataclass(frozen=True)
class ProcoimConfig:
    """Validated configuration for one sequence of Procoim episodes."""

    reduction_fraction: float = 0.005
    adaptive_reduction_fraction: bool = False
    reduction_fraction_decay_rate: float = 0.1
    minimum_reduction_fraction: float = 1e-6
    loss_floor: float = 0.01
    slot_objective_reduction: SlotObjectiveReduction = "per_example"
    candidate_directions: int = 32
    adaptive_candidate_directions: bool = False
    minimum_candidate_directions: int = 16
    candidate_directions_growth: float = 2.0
    antithetic: bool = True
    max_candidate_banks: int = 16
    max_accepted_updates: int = 64
    verify_accepted_candidate: bool = True
    stability_required: int = 3
    soft_regression_fraction: float = 0.02
    hard_regression_fraction: float = 0.08
    absolute_regression_tolerance: float = 1e-4
    protection_weight: float = 2.0
    global_regression_weight: float = 0.0
    minimum_active_gain: float = 1e-4
    probe_scale: float = 0.03
    probe_scale_decay: float = 0.5
    minimum_probe_scale: float = 1e-6
    parameter_rms_floor: float = 1e-3
    reuse_successful_directions: bool = False
    direction_momentum: float = 0.9
    use_antithetic_estimate: bool = False
    antithetic_estimate_only: bool = False
    antithetic_estimate_anchor_aware: bool = False
    antithetic_slot_estimates: bool = False
    antithetic_estimate_step_scale: float | None = None
    directional_adam: bool = False
    directional_adam_learning_rate: float = 0.03
    directional_adam_beta1: float = 0.9
    directional_adam_beta2: float = 0.99
    directional_adam_epsilon: float = 1e-8
    directional_adam_weight_decay: float = 0.0
    directional_adam_projection_iterations: int = 64
    directional_basis_mode: DirectionalBasisMode = "rotating_factorized"
    directional_factor_rank: int = 1
    directional_basis_rotation_banks: int = 8
    directional_basis_rotation_forwards: int | None = None
    directional_basis_rotation_stall_banks: int | None = None
    directional_basis_consolidation_cosine_margin: float | None = None
    directional_basis_consolidation_ema_beta: float = 0.9
    directional_accumulation_batches: int = 1
    reference_reduction_fraction: float | None = None
    candidate_backtrack_factors: tuple[float, ...] = (
        1.0,
        0.5,
        0.25,
        0.125,
    )
    adaptive_probe_scale: bool = False
    probe_scale_growth: float = 1.1
    maximum_probe_scale: float = 0.1
    adaptive_group_selection: bool = False
    group_score_beta: float = 0.9
    group_exploration: float = 1e-3
    anchor_regression_fraction: float = 0.02
    anchor_soft_regression_fraction: float = 0.03
    anchor_hard_regression_fraction: float = 0.10
    anchor_minimum_retained_reduction_fraction: float | None = None
    anchor_absolute_tolerance: float = 1e-4
    anchor_weight: float = 1.0
    anchor_hard_protection: bool = True
    gap_epsilon: float = 1e-8
    parameter_group_granularity: ParameterGroupGranularity = "semantic"
    parameter_group_schedule: tuple[str, ...] = (
        "routing",
        "receiver_memory",
        "stream_mix",
        "node_compute",
        "readout",
    )
    completion_mode: CompletionMode = "all_slots"
    protected_fraction: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        finite_nonnegative = (
            "reduction_fraction_decay_rate",
            "minimum_reduction_fraction",
            "loss_floor",
            "soft_regression_fraction",
            "hard_regression_fraction",
            "absolute_regression_tolerance",
            "protection_weight",
            "global_regression_weight",
            "minimum_active_gain",
            "minimum_probe_scale",
            "parameter_rms_floor",
            "group_exploration",
            "anchor_regression_fraction",
            "anchor_soft_regression_fraction",
            "anchor_hard_regression_fraction",
            "anchor_absolute_tolerance",
            "anchor_weight",
            "directional_adam_weight_decay",
        )
        for name in finite_nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        finite_positive = (
            "probe_scale",
            "probe_scale_decay",
            "probe_scale_growth",
            "maximum_probe_scale",
            "candidate_directions_growth",
            "gap_epsilon",
            "directional_adam_learning_rate",
            "directional_adam_epsilon",
        )
        for name in finite_positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.reduction_fraction <= 1:
            raise ValueError("reduction_fraction must be in [0, 1]")
        if (
            self.adaptive_reduction_fraction
            and self.minimum_reduction_fraction > self.reduction_fraction
        ):
            raise ValueError(
                "minimum_reduction_fraction cannot exceed reduction_fraction"
            )
        if self.candidate_directions_growth <= 1.0:
            raise ValueError("candidate_directions_growth must exceed 1.0")
        if self.soft_regression_fraction > self.hard_regression_fraction:
            raise ValueError(
                "soft_regression_fraction cannot exceed "
                "hard_regression_fraction"
            )
        if (
            self.anchor_soft_regression_fraction
            > self.anchor_hard_regression_fraction
        ):
            raise ValueError("anchor soft tolerance cannot exceed hard tolerance")
        if self.anchor_minimum_retained_reduction_fraction is not None:
            retained = self.anchor_minimum_retained_reduction_fraction
            if (
                not math.isfinite(retained)
                or retained < 0
                or retained > self.reduction_fraction
            ):
                raise ValueError(
                    "anchor_minimum_retained_reduction_fraction must be "
                    "finite and in [0, reduction_fraction]"
                )
        if self.reference_reduction_fraction is not None and (
            not math.isfinite(self.reference_reduction_fraction)
            or not 0 <= self.reference_reduction_fraction <= 1
        ):
            raise ValueError(
                "reference_reduction_fraction must be finite and in [0, 1]"
            )
        if self.probe_scale_decay > 1:
            raise ValueError("probe_scale_decay must be in (0, 1]")
        if self.minimum_probe_scale > self.probe_scale:
            raise ValueError("minimum_probe_scale cannot exceed probe_scale")
        if (
            self.adaptive_probe_scale
            and self.maximum_probe_scale < self.probe_scale
        ):
            raise ValueError("maximum_probe_scale cannot be below probe_scale")
        if (
            self.adaptive_candidate_directions
            and self.minimum_candidate_directions > self.candidate_directions
        ):
            raise ValueError(
                "minimum_candidate_directions cannot exceed "
                "candidate_directions"
            )
        if self.slot_objective_reduction not in {
            "per_example",
            "batch_mean",
        }:
            raise ValueError("unknown slot_objective_reduction")
        if (
            self.directional_accumulation_batches > 1
            and self.slot_objective_reduction != "batch_mean"
        ):
            raise ValueError(
                "directional accumulation requires batch_mean slot objectives"
            )
        if not 0 <= self.direction_momentum < 1:
            raise ValueError("direction_momentum must be in [0, 1)")
        if not 0 <= self.group_score_beta < 1:
            raise ValueError("group_score_beta must be in [0, 1)")
        if not 0 <= self.directional_adam_beta1 < 1:
            raise ValueError("directional_adam_beta1 must be in [0, 1)")
        if not 0 <= self.directional_adam_beta2 < 1:
            raise ValueError("directional_adam_beta2 must be in [0, 1)")
        if (
            self.directional_adam
            and not self.antithetic
        ):
            raise ValueError(
                "directional Adam requires antithetic candidates"
            )
        if self.directional_adam and (
            self.reuse_successful_directions
            or self.use_antithetic_estimate
            or self.antithetic_estimate_only
            or self.antithetic_estimate_anchor_aware
            or self.antithetic_slot_estimates
        ):
            raise ValueError(
                "directional Adam replaces momentum and antithetic "
                "estimate modes"
            )
        if self.directional_adam_learning_rate * self.directional_adam_weight_decay >= 1:
            raise ValueError("directional Adam learning_rate * weight_decay must be < 1")
        if self.directional_basis_mode not in {
            "fixed_rademacher",
            "rotating_factorized",
        }:
            raise ValueError("unknown directional_basis_mode")
        if self.use_antithetic_estimate and not self.antithetic:
            raise ValueError(
                "use_antithetic_estimate requires antithetic candidates"
            )
        if self.antithetic_estimate_only and not self.use_antithetic_estimate:
            raise ValueError(
                "antithetic_estimate_only requires use_antithetic_estimate"
            )
        if (
            self.antithetic_estimate_anchor_aware
            and not self.use_antithetic_estimate
        ):
            raise ValueError("anchor-aware estimates require use_antithetic_estimate")
        if (
            self.antithetic_slot_estimates
            and not self.use_antithetic_estimate
        ):
            raise ValueError(
                "slot estimates require use_antithetic_estimate"
            )
        if self.antithetic_estimate_step_scale is not None and (
            not math.isfinite(self.antithetic_estimate_step_scale)
            or self.antithetic_estimate_step_scale <= 0
        ):
            raise ValueError(
                "antithetic_estimate_step_scale must be finite and positive"
            )
        if not self.candidate_backtrack_factors:
            raise ValueError("candidate_backtrack_factors cannot be empty")
        if self.candidate_backtrack_factors[0] != 1.0:
            raise ValueError("candidate_backtrack_factors must start at 1.0")
        if any(
            not math.isfinite(value) or not 0 < value <= 1
            for value in self.candidate_backtrack_factors
        ):
            raise ValueError("candidate backtrack factors must be in (0, 1]")
        if any(
            left <= right
            for left, right in zip(
                self.candidate_backtrack_factors,
                self.candidate_backtrack_factors[1:],
            )
        ):
            raise ValueError("candidate backtrack factors must strictly decrease")
        if self.parameter_group_granularity not in {
            "semantic",
            "relay",
            "tensor",
        }:
            raise ValueError("unknown parameter_group_granularity")
        for name in (
            "candidate_directions",
            "minimum_candidate_directions",
            "max_candidate_banks",
            "max_accepted_updates",
            "stability_required",
            "directional_adam_projection_iterations",
            "directional_factor_rank",
            "directional_basis_rotation_banks",
            "directional_accumulation_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.directional_basis_rotation_forwards is not None and (
            isinstance(self.directional_basis_rotation_forwards, bool)
            or not isinstance(self.directional_basis_rotation_forwards, int)
            or self.directional_basis_rotation_forwards <= 0
        ):
            raise ValueError(
                "directional_basis_rotation_forwards must be a positive "
                "integer or None"
            )
        if self.directional_basis_rotation_stall_banks is not None and (
            isinstance(self.directional_basis_rotation_stall_banks, bool)
            or not isinstance(self.directional_basis_rotation_stall_banks, int)
            or self.directional_basis_rotation_stall_banks <= 0
        ):
            raise ValueError(
                "directional_basis_rotation_stall_banks must be a positive "
                "integer or None"
            )
        if (
            self.directional_basis_rotation_forwards is not None
            and self.directional_basis_rotation_stall_banks is not None
        ):
            raise ValueError(
                "directional_basis_rotation_forwards and "
                "directional_basis_rotation_stall_banks are alternative "
                "rotation triggers and cannot both be set"
            )
        if self.directional_basis_consolidation_cosine_margin is not None:
            if (
                not math.isfinite(
                    self.directional_basis_consolidation_cosine_margin
                )
                or self.directional_basis_consolidation_cosine_margin <= 0
            ):
                raise ValueError(
                    "directional_basis_consolidation_cosine_margin must be "
                    "a positive finite float or None"
                )
            if self.directional_basis_mode != "rotating_factorized":
                raise ValueError(
                    "directional_basis_consolidation_cosine_margin requires "
                    "directional_basis_mode='rotating_factorized'"
                )
        if not (
            math.isfinite(self.directional_basis_consolidation_ema_beta)
            and 0.0 <= self.directional_basis_consolidation_ema_beta < 1.0
        ):
            raise ValueError(
                "directional_basis_consolidation_ema_beta must be in [0, 1)"
            )
        if not isinstance(self.verify_accepted_candidate, bool):
            raise TypeError(
                "verify_accepted_candidate must be boolean"
            )
        if not self.parameter_group_schedule:
            raise ValueError("parameter_group_schedule cannot be empty")
        if any(not str(name).strip() for name in self.parameter_group_schedule):
            raise ValueError("parameter group names cannot be empty")
        if len(set(self.parameter_group_schedule)) != len(
            self.parameter_group_schedule
        ):
            raise ValueError("parameter_group_schedule cannot contain duplicates")
        if self.completion_mode not in {
            "all_slots",
            "aggregate_and_fraction",
            "budget",
        }:
            raise ValueError("unknown completion_mode")
        if not 0 <= self.protected_fraction <= 1:
            raise ValueError("protected_fraction must be in [0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


class SlotStatus(IntEnum):
    """Episode-local optimization status for one valid output slot."""

    ACTIVE = 0
    SOFT_FROZEN = 1
    PROTECTED = 2


@dataclass
class ProcoimAnchor:
    """A replay example with non-drifting, per-slot protection anchors.

    New anchors preserve a fraction of each slot's *saved gain* from
    ``start_losses`` to ``best_losses``. ``best_aggregate`` remains as a
    compatibility fallback for checkpoints and callers created before
    slotwise replay was introduced.
    """

    batch: Any
    forward_kwargs: Mapping[str, Any] | None = None
    best_aggregate: float | None = None
    anchor_id: Any | None = None
    start_losses: Tensor | None = None
    best_losses: Tensor | None = None
    valid_mask: Tensor | None = None
    protected_mask: Tensor | None = None
    hard_mask: Tensor | None = None
    soft_regression_fraction: float | None = None
    hard_regression_fraction: float | None = None
    absolute_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.best_aggregate is not None and (
            not math.isfinite(self.best_aggregate)
            or self.best_aggregate < 0
        ):
            raise ValueError(
                "anchor best_aggregate must be finite and nonnegative"
            )

        for name in (
            "soft_regression_fraction",
            "hard_regression_fraction",
            "absolute_tolerance",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or value < 0
            ):
                raise ValueError(
                    f"anchor {name} must be finite and nonnegative"
                )
        if (
            self.soft_regression_fraction is not None
            and self.hard_regression_fraction is not None
            and self.soft_regression_fraction > self.hard_regression_fraction
        ):
            raise ValueError("anchor soft tolerance cannot exceed hard tolerance")

        slot_fields = (
            self.start_losses,
            self.best_losses,
            self.valid_mask,
            self.protected_mask,
        )
        populated = tuple(value is not None for value in slot_fields)
        if any(populated) and not all(populated):
            raise ValueError(
                "slotwise anchors require start_losses, best_losses, "
                "valid_mask, and protected_mask"
            )
        if not all(populated):
            return

        assert self.start_losses is not None
        assert self.best_losses is not None
        assert self.valid_mask is not None
        assert self.protected_mask is not None
        shape = self.start_losses.shape
        if any(value.shape != shape for value in slot_fields[1:]):
            raise ValueError("slotwise anchor tensors must have matching shapes")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("anchor valid_mask must be boolean")
        if self.protected_mask.dtype != torch.bool:
            raise TypeError("anchor protected_mask must be boolean")
        if bool((self.protected_mask & ~self.valid_mask).any()):
            raise ValueError("anchor cannot protect an invalid or empty slot")
        if self.hard_mask is None:
            # Legacy anchors treated every retained objective as a hard
            # replay constraint. Keep that behavior for manually-created
            # and checkpoint-restored anchors unless lifecycle information
            # is explicitly supplied.
            self.hard_mask = self.protected_mask.detach().clone()
        if self.hard_mask.shape != shape:
            raise ValueError("anchor hard_mask must match slotwise tensors")
        if self.hard_mask.dtype != torch.bool:
            raise TypeError("anchor hard_mask must be boolean")
        if bool((self.hard_mask & ~self.protected_mask).any()):
            raise ValueError("hard anchor slots must also be retained slots")
        if not bool(torch.isfinite(self.start_losses[self.valid_mask]).all()):
            raise ValueError("anchor start losses must be finite on valid slots")
        if not bool(torch.isfinite(self.best_losses[self.valid_mask]).all()):
            raise ValueError("anchor best losses must be finite on valid slots")
        if bool(
            self.best_losses[self.protected_mask].gt(
                self.start_losses[self.protected_mask]
            ).any()
        ):
            raise ValueError("protected anchor slots must preserve a saved gain")

    @property
    def slotwise(self) -> bool:
        return self.start_losses is not None

    @classmethod
    def from_episode(
        cls,
        *,
        batch: Any,
        state: "ProcoimEpisodeState",
        forward_kwargs: Mapping[str, Any] | None = None,
        anchor_id: Any | None = None,
    ) -> "ProcoimAnchor":
        """Create an anchor without allowing padded/empty slots to write."""

        gain = state.start_loss - state.best_loss
        reached = state.soft_frozen_mask | state.protected_mask
        retained = state.valid_mask & reached & gain.gt(0)
        hard = state.valid_mask & state.protected_mask & gain.gt(0)
        return cls(
            batch=batch,
            forward_kwargs=forward_kwargs,
            best_aggregate=float(
                state.best_loss[state.valid_mask].mean()
            ),
            anchor_id=anchor_id,
            start_losses=state.start_loss.detach().clone(),
            best_losses=state.best_loss.detach().clone(),
            valid_mask=state.valid_mask.detach().clone(),
            protected_mask=retained.detach().clone(),
            hard_mask=hard.detach().clone(),
        )


@dataclass
class ProcoimEpisodeState:
    """All mutable slot-level state for one Procoim episode."""

    start_loss: Tensor
    target_loss: Tensor
    best_loss: Tensor
    current_loss: Tensor
    valid_mask: Tensor
    status: Tensor
    stable_count: Tensor
    current_guard_loss: float | None = None

    def clone(self) -> "ProcoimEpisodeState":
        return ProcoimEpisodeState(
            self.start_loss.clone(),
            self.target_loss.clone(),
            self.best_loss.clone(),
            self.current_loss.clone(),
            self.valid_mask.clone(),
            self.status.clone(),
            self.stable_count.clone(),
            self.current_guard_loss,
        )

    @property
    def active_mask(self) -> Tensor:
        return self.valid_mask & self.status.eq(int(SlotStatus.ACTIVE))

    @property
    def soft_frozen_mask(self) -> Tensor:
        return self.valid_mask & self.status.eq(int(SlotStatus.SOFT_FROZEN))

    @property
    def protected_mask(self) -> Tensor:
        return self.valid_mask & self.status.eq(int(SlotStatus.PROTECTED))

    def count(self, status: SlotStatus) -> int:
        return int((self.valid_mask & self.status.eq(int(status))).sum())


@dataclass(frozen=True)
class CandidateResult:
    """Forward-only score for one signed perturbation candidate."""

    group_name: str
    direction_index: int
    sign: int
    losses: Tensor
    aggregate: float
    active_gain: float
    soft_penalty: float
    global_regression: float
    utility: float
    hard_violation: bool
    eligible: bool
    guard_aggregate: float | None = None
    reference_aggregate: float | None = None
    reference_gain: float = 0.0
    reference_violation: bool = False
    source: CandidateSource = "random"
    anchor_penalty: float = 0.0
    anchor_violation: bool = False
    anchor_aggregates: tuple[float, ...] = ()
    anchor_losses: tuple[Tensor, ...] = ()
    anchor_retained_reduction: float | None = None
    step_fraction: float = 1.0
    error: str | None = None


@dataclass(frozen=True)
class ProcoimAnchorAuditResult:
    """Forward-counted retention audit over a selected anchor set."""

    anchor_forwards: int
    penalty: float
    hard_violation: bool
    aggregates: tuple[float, ...]
    losses: tuple[Tensor, ...]
    retained_reduction: float | None = None


@dataclass
class ProcoimEpisodeResult:
    """Episode outcome and logging metrics."""

    state: ProcoimEpisodeState
    example_id: Any | None
    start_aggregate_loss: float
    end_aggregate_loss: float
    accepted_updates: int
    candidate_forwards: int
    anchor_forwards: int
    verification_forwards: int
    stability_forwards: int
    hard_vetoes: int
    anchor_vetoes: int
    stall_count: int
    current_probe_scale: float
    parameter_group_acceptance_counts: dict[str, int]
    parameter_group_probe_scales: dict[str, float]
    parameter_group_candidate_directions: dict[str, int]
    effective_reduction_fraction: float
    parameter_group_gain_per_forward: dict[str, float]
    parameter_group_basis_epochs: dict[str, int]
    reused_direction_accepts: int
    antithetic_estimate_accepts: int
    directional_adam_accepts: int
    total_reused_direction_accepts: int
    total_antithetic_estimate_accepts: int
    total_directional_adam_accepts: int
    total_anchor_vetoes: int
    completed: bool
    completion_reason: str
    wall_clock_seconds: float
    baseline_forwards: int = 1
    reference_forwards: int = 0
    reference_vetoes: int = 0
    reference_rotations: int = 0
    reference_start_loss: float | None = None
    reference_end_loss: float | None = None

    @property
    def total_forward_calls(self) -> int:
        return (
            self.baseline_forwards
            + self.candidate_forwards
            + self.anchor_forwards
            + self.reference_forwards
            + self.verification_forwards
            + self.stability_forwards
        )

    @property
    def reference_reduction(self) -> float | None:
        if (
            self.reference_start_loss is None
            or self.reference_end_loss is None
        ):
            return None
        denominator = max(abs(self.reference_start_loss), 1e-12)
        return (
            self.reference_start_loss - self.reference_end_loss
        ) / denominator

    @property
    def aggregate_reduction(self) -> float:
        denominator = max(abs(self.start_aggregate_loss), 1e-12)
        return (
            self.start_aggregate_loss - self.end_aggregate_loss
        ) / denominator

    @property
    def active_slots(self) -> int:
        return self.state.count(SlotStatus.ACTIVE)

    @property
    def soft_frozen_slots(self) -> int:
        return self.state.count(SlotStatus.SOFT_FROZEN)

    @property
    def protected_slots(self) -> int:
        return self.state.count(SlotStatus.PROTECTED)

    @property
    def slot_target_hit_rate(self) -> float:
        hit = self.state.current_loss <= self.state.target_loss
        return float(hit[self.state.valid_mask].float().mean())

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_updates / max(self.candidate_forwards, 1)

    @property
    def accepted_updates_per_forward(self) -> float:
        return self.accepted_updates / max(self.total_forward_calls, 1)

    @property
    def hard_veto_rate(self) -> float:
        return self.hard_vetoes / max(self.candidate_forwards, 1)

    @property
    def protected_regressions(self) -> Tensor:
        mask = self.state.protected_mask
        if not bool(mask.any()):
            return self.state.current_loss.new_zeros((0,))
        return (self.state.current_loss - self.state.best_loss)[mask]

    @property
    def max_protected_regression(self) -> float:
        regressions = self.protected_regressions
        return 0.0 if regressions.numel() == 0 else float(regressions.max())

    @property
    def mean_protected_regression(self) -> float:
        regressions = self.protected_regressions
        return 0.0 if regressions.numel() == 0 else float(regressions.mean())

    def metrics(self) -> dict[str, Any]:
        """Return the requested episode logging fields as plain values."""

        return {
            "example_id": self.example_id,
            "start_aggregate_loss": self.start_aggregate_loss,
            "end_aggregate_loss": self.end_aggregate_loss,
            "aggregate_reduction": self.aggregate_reduction,
            "active_slots": self.active_slots,
            "soft_frozen_slots": self.soft_frozen_slots,
            "protected_slots": self.protected_slots,
            "slot_target_hit_rate": self.slot_target_hit_rate,
            "accepted_updates": self.accepted_updates,
            "candidate_forwards": self.candidate_forwards,
            "anchor_forwards": self.anchor_forwards,
            "baseline_forwards": self.baseline_forwards,
            "reference_forwards": self.reference_forwards,
            "reference_vetoes": self.reference_vetoes,
            "reference_rotations": self.reference_rotations,
            "reference_start_loss": self.reference_start_loss,
            "reference_end_loss": self.reference_end_loss,
            "reference_reduction": self.reference_reduction,
            "verification_forwards": self.verification_forwards,
            "stability_forwards": self.stability_forwards,
            "total_forward_calls": self.total_forward_calls,
            "acceptance_rate": self.acceptance_rate,
            "accepted_updates_per_forward": self.accepted_updates_per_forward,
            "hard_veto_rate": self.hard_veto_rate,
            "anchor_vetoes": self.anchor_vetoes,
            "stall_count": self.stall_count,
            "max_protected_regression": self.max_protected_regression,
            "mean_protected_regression": self.mean_protected_regression,
            "current_probe_scale": self.current_probe_scale,
            "parameter_group_acceptance_counts": dict(
                self.parameter_group_acceptance_counts
            ),
            "parameter_group_probe_scales": dict(
                self.parameter_group_probe_scales
            ),
            "parameter_group_candidate_directions": dict(
                self.parameter_group_candidate_directions
            ),
            "effective_reduction_fraction": self.effective_reduction_fraction,
            "parameter_group_gain_per_forward": dict(
                self.parameter_group_gain_per_forward
            ),
            "parameter_group_basis_epochs": dict(
                self.parameter_group_basis_epochs
            ),
            "reused_direction_accepts": self.reused_direction_accepts,
            "antithetic_estimate_accepts": self.antithetic_estimate_accepts,
            "total_reused_direction_accepts": (
                self.total_reused_direction_accepts
            ),
            "total_antithetic_estimate_accepts": (
                self.total_antithetic_estimate_accepts
            ),
            "directional_adam_accepts": self.directional_adam_accepts,
            "total_directional_adam_accepts": (
                self.total_directional_adam_accepts
            ),
            "total_anchor_vetoes": self.total_anchor_vetoes,
            "wall_clock_seconds": self.wall_clock_seconds,
            "completed": self.completed,
            "completion_reason": self.completion_reason,
        }


@dataclass(frozen=True)
class _ParameterGroup:
    name: str
    named_parameters: tuple[tuple[str, nn.Parameter], ...]

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for _, parameter in self.named_parameters)


@dataclass(frozen=True)
class _DirectionalAdamCommit:
    group_name: str
    first_moment: Tensor
    second_moment: Tensor
    step: int
    last_decay_step: int
    basis_epoch: int


@dataclass(frozen=True)
class _CandidateBankResult:
    winner: CandidateResult | None
    winner_direction: tuple[Tensor, ...] | None
    candidate_forwards: int
    anchor_forwards: int
    hard_vetoes: int
    anchor_vetoes: int
    reference_forwards: int = 0
    reference_vetoes: int = 0
    directional_adam_commit: _DirectionalAdamCommit | None = None

    @property
    def search_forwards(self) -> int:
        return (
            self.candidate_forwards
            + self.anchor_forwards
            + self.reference_forwards
        )


@dataclass
class _GlobalRNGState:
    python_state: object
    torch_cpu_state: Tensor
    torch_cuda_states: list[Tensor] | None
    numpy_state: object | None


def _capture_global_rng() -> _GlobalRNGState:
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except ImportError:
        pass
    return _GlobalRNGState(
        random.getstate(),
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        numpy_state,
    )


def _restore_global_rng(state: _GlobalRNGState) -> None:
    random.setstate(state.python_state)
    torch.random.set_rng_state(state.torch_cpu_state)
    if state.torch_cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda_states)
    if state.numpy_state is not None:
        try:
            import numpy as np

            np.random.set_state(state.numpy_state)
        except ImportError:
            pass


class ProcoimOptimizer:
    """Forward-only structured perturbation optimizer with slot protection."""

    format_version = 19

    def __init__(self, config: ProcoimConfig | None = None) -> None:
        self.config = ProcoimConfig() if config is None else config
        if not isinstance(self.config, ProcoimConfig):
            raise TypeError("config must be a ProcoimConfig")
        self.episodes = 0
        self.total_forward_calls = 0
        self.total_candidate_forwards = 0
        self.total_anchor_forwards = 0
        self.total_reference_forwards = 0
        self.total_accepted_updates = 0
        self.total_reused_direction_accepts = 0
        self.total_antithetic_estimate_accepts = 0
        self.total_directional_adam_accepts = 0
        self.total_anchor_vetoes = 0
        self.total_reference_vetoes = 0
        self.total_consolidation_freezes = 0
        self.parameter_group_acceptance_counts: dict[str, int] = {}
        self._next_group_index = 0
        self._group_probe_scales: dict[str, float] = {}
        self._group_candidate_directions: dict[str, int] = {}
        self._group_basis_forward_counts: dict[str, int] = {}
        self._group_basis_stall_streak: dict[str, int] = {}
        self._group_basis_stall_epochs: dict[str, int] = {}
        self._group_gain_ema: dict[str, float] = {}
        self._group_bank_counts: dict[str, int] = {}
        self._direction_momentum: dict[str, tuple[Tensor, ...]] = {}
        self._group_window_accum: dict[str, tuple[Tensor, ...]] = {}
        self._group_prev_window_vector: dict[str, tuple[Tensor, ...]] = {}
        self._group_window_cosine_baseline: dict[str, float] = {}
        self._group_epoch_freeze: dict[str, int] = {}
        self._group_last_natural_epoch: dict[str, int] = {}
        self._directional_adam_m: dict[str, Tensor] = {}
        self._directional_adam_v: dict[str, Tensor] = {}
        self._directional_adam_steps: dict[str, int] = {}
        self._directional_adam_last_decay_step: dict[str, int] = {}
        self._directional_adam_basis_epochs: dict[str, int] = {}
        self._generators: dict[str, torch.Generator] = {}
        self._pending_generator_states: dict[str, Tensor] = {}
        self._group_signature: tuple[
            tuple[str, tuple[tuple[str, tuple[int, ...], str], ...]], ...
        ] | None = None
        self._expected_group_signature = None

    @staticmethod
    def _assert_no_gradients(model: nn.Module) -> None:
        populated = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ]
        if populated:
            preview = ", ".join(populated[:8])
            raise RuntimeError(
                "Procoim requires empty .grad fields; found gradients for "
                f"{preview}"
            )

    @staticmethod
    def _snapshot_buffers(model: nn.Module) -> dict[str, Tensor]:
        return {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
        }

    @staticmethod
    @torch.no_grad()
    def _restore_buffers(model: nn.Module, snapshot: Mapping[str, Tensor]) -> None:
        current = dict(model.named_buffers())
        if current.keys() != snapshot.keys():
            raise RuntimeError("model buffer layout changed during Procoim episode")
        for name, value in snapshot.items():
            current[name].copy_(value)

    @staticmethod
    def _parameter_name_map(model: nn.Module) -> dict[int, str]:
        return {
            id(parameter): name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def select_parameter_groups(self, model: nn.Module) -> tuple[_ParameterGroup, ...]:
        """Resolve deterministic, disjoint trainable parameter groups."""

        name_by_id = self._parameter_name_map(model)
        if not name_by_id:
            raise ValueError("model has no trainable parameters")
        method = getattr(model, "procoim_parameter_groups", None)
        if method is None:
            named = tuple(
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            groups = (_ParameterGroup("all_parameters", named),)
        else:
            exposed = method()
            if not isinstance(exposed, Mapping):
                raise TypeError("procoim_parameter_groups() must return a mapping")
            seen: set[int] = set()
            available: dict[str, _ParameterGroup] = {}
            for raw_name, raw_parameters in exposed.items():
                group_name = str(raw_name)
                if not group_name:
                    raise ValueError("Procoim parameter group names cannot be empty")
                named_parameters: list[tuple[str, nn.Parameter]] = []
                for parameter in raw_parameters:
                    if not isinstance(parameter, nn.Parameter):
                        raise TypeError(
                            f"parameter group {group_name!r} contains a non-Parameter"
                        )
                    if not parameter.requires_grad:
                        continue
                    identity = id(parameter)
                    if identity not in name_by_id:
                        raise ValueError(
                            f"parameter group {group_name!r} contains a parameter "
                            "that is not a trainable member of the model"
                        )
                    if identity in seen:
                        raise ValueError(
                            "Procoim parameter groups overlap at "
                            f"{name_by_id[identity]!r}"
                        )
                    seen.add(identity)
                    named_parameters.append((name_by_id[identity], parameter))
                if named_parameters:
                    available[group_name] = _ParameterGroup(
                        group_name,
                        tuple(named_parameters),
                    )
            unknown = [
                name
                for name in self.config.parameter_group_schedule
                if name not in available
            ]
            if unknown:
                available_names = ", ".join(available) or "<none>"
                raise ValueError(
                    "parameter_group_schedule contains unavailable groups "
                    f"{unknown}; available groups: {available_names}"
                )
            selected = [
                available[name]
                for name in self.config.parameter_group_schedule
            ]
            if self.config.parameter_group_granularity == "relay":
                relay_groups: list[_ParameterGroup] = []
                for group in selected:
                    buckets: dict[
                        str,
                        list[tuple[str, nn.Parameter]],
                    ] = {}
                    for parameter_name, parameter in group.named_parameters:
                        parts = parameter_name.split(".")
                        if (
                            len(parts) >= 4
                            and parts[0] == "core"
                            and parts[1] == "relays"
                            and parts[2].isdigit()
                        ):
                            locality = f"relay{parts[2]}"
                        else:
                            locality = "shared"
                        buckets.setdefault(locality, []).append(
                            (parameter_name, parameter)
                        )
                    relay_groups.extend(
                        _ParameterGroup(
                            f"{group.name}::{locality}",
                            tuple(named_parameters),
                        )
                        for locality, named_parameters in buckets.items()
                    )
                groups = tuple(relay_groups)
            elif self.config.parameter_group_granularity == "tensor":
                groups = tuple(
                    _ParameterGroup(
                        f"{group.name}::{parameter_name}",
                        ((parameter_name, parameter),),
                    )
                    for group in selected
                    for parameter_name, parameter in group.named_parameters
                )
            else:
                groups = tuple(selected)

        storage_owners: dict[tuple[str, int], str] = {}
        for group in groups:
            for name, parameter in group.named_parameters:
                if parameter.is_meta or parameter.is_sparse:
                    raise TypeError("Procoim does not support meta or sparse parameters")
                if parameter.is_complex():
                    raise TypeError("Procoim does not support complex parameters")
                if parameter.dtype in (torch.float16, torch.bfloat16):
                    raise TypeError(
                        "Procoim MVP requires FP32/FP64 parameters; low-precision "
                        "models need FP32 master weights"
                    )
                if not parameter.is_floating_point():
                    raise TypeError("Procoim parameters must be floating point")
                storage_key = (
                    str(parameter.device),
                    parameter.untyped_storage().data_ptr(),
                )
                previous = storage_owners.get(storage_key)
                if previous is not None and previous != name:
                    raise ValueError(
                        "distinct Procoim parameters share storage: "
                        f"{previous!r} and {name!r}"
                    )
                storage_owners[storage_key] = name

        signature = tuple(
            (
                group.name,
                tuple(
                    (name, tuple(parameter.shape), str(parameter.dtype))
                    for name, parameter in group.named_parameters
                ),
            )
            for group in groups
        )
        if (
            self._expected_group_signature is not None
            and signature != self._expected_group_signature
        ):
            raise RuntimeError(
                "checkpointed Procoim parameter-group layout does not match model"
            )
        self._group_signature = signature
        self._ensure_group_search_state(groups)
        return groups

    @staticmethod
    def _restore_direction_tensor_mapping(
        mapping: dict[str, tuple[Tensor, ...]],
        group: _ParameterGroup,
        *,
        label: str,
    ) -> None:
        """Validate and device/dtype-restore one checkpointed per-tensor
        direction (a per-parameter tuple shaped like the group) in place."""

        value = mapping.get(group.name)
        if value is None:
            return
        if len(value) != len(group.parameters):
            raise RuntimeError(
                f"checkpointed {label} for {group.name!r} has the wrong "
                "parameter count"
            )
        restored: list[Tensor] = []
        for tensor, parameter in zip(value, group.parameters, strict=True):
            if tensor.shape != parameter.shape:
                raise RuntimeError(
                    f"checkpointed {label} shape for {group.name!r} does "
                    "not match the model"
                )
            restored.append(
                tensor.detach().to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ).clone()
            )
        mapping[group.name] = tuple(restored)

    def _ensure_group_search_state(
        self,
        groups: Sequence[_ParameterGroup],
    ) -> None:
        """Initialize and validate persistent search statistics lazily."""

        available = {group.name for group in groups}
        for mapping_name, mapping in (
            ("group_probe_scales", self._group_probe_scales),
            ("group_candidate_directions", self._group_candidate_directions),
            (
                "group_basis_forward_counts",
                self._group_basis_forward_counts,
            ),
            ("group_basis_stall_streak", self._group_basis_stall_streak),
            ("group_basis_stall_epochs", self._group_basis_stall_epochs),
            ("group_gain_ema", self._group_gain_ema),
            ("group_bank_counts", self._group_bank_counts),
            ("direction_momentum", self._direction_momentum),
            ("group_window_accum", self._group_window_accum),
            ("group_prev_window_vector", self._group_prev_window_vector),
            (
                "group_window_cosine_baseline",
                self._group_window_cosine_baseline,
            ),
            ("group_epoch_freeze", self._group_epoch_freeze),
            ("group_last_natural_epoch", self._group_last_natural_epoch),
            ("directional_adam_m", self._directional_adam_m),
            ("directional_adam_v", self._directional_adam_v),
            ("directional_adam_steps", self._directional_adam_steps),
            (
                "directional_adam_last_decay_step",
                self._directional_adam_last_decay_step,
            ),
            (
                "directional_adam_basis_epochs",
                self._directional_adam_basis_epochs,
            ),
        ):
            unknown = set(mapping) - available
            if unknown:
                raise RuntimeError(
                    f"checkpointed {mapping_name} contains unknown groups "
                    f"{sorted(unknown)}"
                )

        for group in groups:
            self._group_probe_scales.setdefault(
                group.name,
                self.config.probe_scale,
            )
            self._group_candidate_directions.setdefault(
                group.name,
                (
                    self.config.minimum_candidate_directions
                    if self.config.adaptive_candidate_directions
                    else self.config.candidate_directions
                ),
            )
            self._group_basis_forward_counts.setdefault(group.name, 0)
            self._group_basis_stall_streak.setdefault(group.name, 0)
            self._group_basis_stall_epochs.setdefault(group.name, 0)
            self._group_gain_ema.setdefault(group.name, 0.0)
            self._group_bank_counts.setdefault(group.name, 0)
            self._group_epoch_freeze.setdefault(group.name, 0)
            self._group_last_natural_epoch.setdefault(group.name, 0)
            self._restore_direction_tensor_mapping(
                self._direction_momentum,
                group,
                label="momentum",
            )
            self._restore_direction_tensor_mapping(
                self._group_window_accum,
                group,
                label="consolidation window accumulator",
            )
            self._restore_direction_tensor_mapping(
                self._group_prev_window_vector,
                group,
                label="consolidation previous window vector",
            )
            for mapping_name, mapping in (
                ("directional_adam_m", self._directional_adam_m),
                ("directional_adam_v", self._directional_adam_v),
            ):
                moment = mapping.get(group.name)
                if moment is None:
                    continue
                if tuple(moment.shape) != (self.config.candidate_directions,):
                    raise RuntimeError(
                        f"checkpointed {mapping_name} for {group.name!r} "
                        "does not match candidate_directions"
                    )
                if not bool(torch.isfinite(moment).all()):
                    raise ValueError(
                        f"checkpointed {mapping_name} contains non-finite values"
                    )
                mapping[group.name] = moment.detach().cpu().to(
                    dtype=torch.float64
                ).clone()
            step = self._directional_adam_steps.get(group.name, 0)
            decay_step = self._directional_adam_last_decay_step.get(
                group.name, 0
            )
            basis_epoch = self._directional_adam_basis_epochs.get(
                group.name,
                0,
            )
            if step < 0 or decay_step < 0 or basis_epoch < 0:
                raise ValueError("directional Adam steps cannot be negative")

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            stable_offset = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(key)
            )
            generator.manual_seed(self.config.seed + stable_offset)
            pending = self._pending_generator_states.pop(key, None)
            if pending is not None:
                generator.set_state(pending)
            self._generators[key] = generator
        return generator

    @staticmethod
    @torch.no_grad()
    def _snapshot_group(group: _ParameterGroup) -> tuple[Tensor, ...]:
        return tuple(
            parameter.detach().clone() for parameter in group.parameters
        )

    @staticmethod
    @torch.no_grad()
    def _restore_group(
        group: _ParameterGroup,
        snapshot: Sequence[Tensor],
    ) -> None:
        if len(group.parameters) != len(snapshot):
            raise ValueError("parameter snapshot length does not match group")
        for parameter, original in zip(
            group.parameters,
            snapshot,
            strict=True,
        ):
            parameter.copy_(original)

    @staticmethod
    @torch.no_grad()
    def _apply_direction(
        group: _ParameterGroup,
        direction: Sequence[Tensor],
        sign: int,
    ) -> None:
        if sign not in (-1, 1):
            raise ValueError("candidate sign must be -1 or +1")
        if len(group.parameters) != len(direction):
            raise ValueError("direction length does not match parameter group")
        for parameter, delta in zip(
            group.parameters,
            direction,
            strict=True,
        ):
            parameter.add_(delta, alpha=sign)

    def sample_direction(
        self,
        group: _ParameterGroup,
        snapshot: Sequence[Tensor],
        *,
        probe_scale: float,
    ) -> tuple[Tensor, ...]:
        """Sample independently normalized Rademacher noise per tensor.

        PROCOIM_OPTIMIZER_SPEC_V0.2 section 9.4 deliberately gives every
        tensor in a functional group its own L2-normalized direction and
        parameter-RMS-relative radius.  A single flattened group norm lets
        large matrices consume nearly all of the perturbation while small
        routing gates and biases receive negligible search energy.
        """

        if len(group.parameters) != len(snapshot):
            raise ValueError("parameter snapshot length does not match group")
        direction = []
        for parameter, original in zip(
            group.parameters,
            snapshot,
            strict=True,
        ):
            bits = torch.randint(
                0,
                2,
                parameter.shape,
                generator=self._generator(parameter.device),
                device=parameter.device,
                dtype=torch.int8,
            )
            signs = bits.to(dtype=parameter.dtype).mul_(2).sub_(1)
            signs.div_(math.sqrt(parameter.numel()))
            direction.append(
                signs.mul_(
                    self._tensor_probe_magnitude(original, probe_scale)
                )
            )
        return tuple(direction)

    def _tensor_probe_magnitude(
        self,
        parameter: Tensor,
        probe_scale: float,
    ) -> float:
        if parameter.numel() <= 0:
            raise ValueError("cannot perturb an empty parameter tensor")
        parameter_rms = float(
            parameter.detach().double().square().mean().sqrt().cpu()
        )
        return probe_scale * max(
            parameter_rms,
            self.config.parameter_rms_floor,
        )

    def _normalize_direction(
        self,
        direction: Sequence[Tensor],
    ) -> tuple[Tensor, ...] | None:
        normalized: list[Tensor] = []
        useful = False
        for value in direction:
            norm = float(value.detach().double().norm().cpu())
            if not math.isfinite(norm):
                return None
            if norm <= 0:
                normalized.append(torch.zeros_like(value))
                continue
            useful = True
            normalized.append(value.detach().clone().div_(norm))
        if not useful:
            return None
        return tuple(normalized)

    def _rescale_direction(
        self,
        snapshot: Sequence[Tensor],
        direction: Sequence[Tensor],
        *,
        probe_scale: float,
    ) -> tuple[Tensor, ...] | None:
        if len(snapshot) != len(direction):
            raise ValueError(
                "direction length does not match parameter snapshot"
            )
        normalized = self._normalize_direction(direction)
        if normalized is None:
            return None
        return tuple(
            value.mul(
                self._tensor_probe_magnitude(original, probe_scale)
            )
            for original, value in zip(
                snapshot,
                normalized,
                strict=True,
            )
        )

    def _persistent_direction(
        self,
        group: _ParameterGroup,
        snapshot: Sequence[Tensor],
        *,
        basis_index: int,
        probe_scale: float,
        tensor_magnitudes: Sequence[float] | None = None,
    ) -> tuple[Tensor, ...]:
        """Regenerate one persistent seeded Rademacher basis direction."""

        if not 0 <= basis_index < self.config.candidate_directions:
            raise ValueError("basis_index is outside candidate_directions")
        if len(group.parameters) != len(snapshot):
            raise ValueError("parameter snapshot length does not match group")
        if tensor_magnitudes is None:
            magnitudes = tuple(
                self._tensor_probe_magnitude(original, probe_scale)
                for original in snapshot
            )
        else:
            if len(tensor_magnitudes) != len(snapshot):
                raise ValueError(
                    "tensor magnitudes do not match parameter snapshot"
                )
            magnitudes = tuple(float(value) for value in tensor_magnitudes)
            if any(
                not math.isfinite(value) or value < 0
                for value in magnitudes
            ):
                raise ValueError("tensor magnitudes must be finite and nonnegative")
        group_offset = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(group.name)
        )
        directions: list[Tensor] = []
        modulus = (1 << 63) - 1
        for tensor_index, parameter in enumerate(group.parameters):
            seed = (
                self.config.seed
                + 1_000_003 * (basis_index + 1)
                + 97_409 * (tensor_index + 1)
                + group_offset
            ) % modulus
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(seed)
            bits = torch.randint(
                0,
                2,
                parameter.shape,
                generator=generator,
                device=parameter.device,
                dtype=torch.int8,
            )
            signs = bits.to(dtype=parameter.dtype).mul_(2).sub_(1)
            signs.div_(math.sqrt(parameter.numel()))
            directions.append(signs.mul_(magnitudes[tensor_index]))
        return tuple(directions)

    def _natural_basis_epoch(
        self,
        group: _ParameterGroup,
    ) -> int:
        """Return the basis window driven by the configured rotation
        trigger (banks / forwards / stall-banks), before any consolidation
        freeze is applied."""

        if self.config.directional_basis_mode == "fixed_rademacher":
            return 0
        if self.config.directional_basis_rotation_stall_banks is not None:
            return self._group_basis_stall_epochs.get(group.name, 0)
        if self.config.directional_basis_rotation_forwards is not None:
            forwards = self._group_basis_forward_counts.get(group.name, 0)
            return forwards // self.config.directional_basis_rotation_forwards
        bank_count = self._group_bank_counts.get(group.name, 0)
        return bank_count // self.config.directional_basis_rotation_banks

    def _directional_basis_epoch(
        self,
        group: _ParameterGroup,
    ) -> int:
        """Return the deterministic basis window for a parameter group.

        Equal to the natural rotation-trigger epoch, minus any windows
        deferred by the consolidation-cosine gate (see
        ``_maybe_gate_consolidation``): a group whose recently accepted
        update directions are unusually self-consistent has its basis held
        in place instead of rotated on schedule.
        """

        natural = self._natural_basis_epoch(group)
        if self.config.directional_basis_consolidation_cosine_margin is None:
            return natural
        freeze = self._group_epoch_freeze.get(group.name, 0)
        return max(0, natural - freeze)

    def _rotating_factorized_direction(
        self,
        group: _ParameterGroup,
        snapshot: Sequence[Tensor],
        *,
        basis_index: int,
        basis_epoch: int,
        probe_scale: float,
        tensor_magnitudes: Sequence[float] | None = None,
    ) -> tuple[Tensor, ...]:
        """Regenerate one seeded low-rank direction for a rotating window."""

        if not 0 <= basis_index < self.config.candidate_directions:
            raise ValueError("basis_index is outside candidate_directions")
        if basis_epoch < 0:
            raise ValueError("basis_epoch must be nonnegative")
        if len(group.parameters) != len(snapshot):
            raise ValueError("parameter snapshot length does not match group")
        if tensor_magnitudes is None:
            magnitudes = tuple(
                self._tensor_probe_magnitude(original, probe_scale)
                for original in snapshot
            )
        else:
            if len(tensor_magnitudes) != len(snapshot):
                raise ValueError(
                    "tensor magnitudes do not match parameter snapshot"
                )
            magnitudes = tuple(float(value) for value in tensor_magnitudes)
            if any(
                not math.isfinite(value) or value < 0
                for value in magnitudes
            ):
                raise ValueError("tensor magnitudes must be finite and nonnegative")

        group_offset = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(group.name)
        )
        directions: list[Tensor] = []
        modulus = (1 << 63) - 1
        for tensor_index, parameter in enumerate(group.parameters):
            seed = (
                self.config.seed
                + 1_000_003 * (basis_index + 1)
                + 97_409 * (tensor_index + 1)
                + 15_485_863 * (basis_epoch + 1)
                + group_offset
            ) % modulus
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(seed)
            if parameter.ndim < 2:
                bits = torch.randint(
                    0,
                    2,
                    parameter.shape,
                    generator=generator,
                    device=parameter.device,
                    dtype=torch.int8,
                )
                value = bits.to(dtype=parameter.dtype).mul_(2).sub_(1)
                value.div_(math.sqrt(parameter.numel()))
            else:
                rows = parameter.shape[0]
                columns = parameter.numel() // rows
                rank = min(
                    self.config.directional_factor_rank,
                    rows,
                    columns,
                )
                work_dtype = (
                    torch.float64
                    if parameter.dtype == torch.float64
                    else torch.float32
                )
                left_bits = torch.randint(
                    0,
                    2,
                    (rows, rank),
                    generator=generator,
                    device=parameter.device,
                    dtype=torch.int8,
                )
                right_bits = torch.randint(
                    0,
                    2,
                    (rank, columns),
                    generator=generator,
                    device=parameter.device,
                    dtype=torch.int8,
                )
                left = left_bits.to(dtype=work_dtype).mul_(2).sub_(1)
                right = right_bits.to(dtype=work_dtype).mul_(2).sub_(1)
                if rank == 1:
                    matrix = left * right
                    matrix.div_(math.sqrt(rows * columns))
                else:
                    matrix = left.matmul(right)
                    norm = torch.linalg.vector_norm(matrix)
                    matrix.div_(norm.clamp_min(torch.finfo(work_dtype).tiny))
                value = matrix.reshape(parameter.shape).to(
                    dtype=parameter.dtype
                )
            directions.append(value.mul_(magnitudes[tensor_index]))
        return tuple(directions)

    def _directional_basis_direction(
        self,
        group: _ParameterGroup,
        snapshot: Sequence[Tensor],
        *,
        basis_index: int,
        basis_epoch: int,
        probe_scale: float,
        tensor_magnitudes: Sequence[float] | None = None,
    ) -> tuple[Tensor, ...]:
        if self.config.directional_basis_mode == "fixed_rademacher":
            return self._persistent_direction(
                group,
                snapshot,
                basis_index=basis_index,
                probe_scale=probe_scale,
                tensor_magnitudes=tensor_magnitudes,
            )
        return self._rotating_factorized_direction(
            group,
            snapshot,
            basis_index=basis_index,
            basis_epoch=basis_epoch,
            probe_scale=probe_scale,
            tensor_magnitudes=tensor_magnitudes,
        )

    @staticmethod
    def _combine_directions(
        basis: Sequence[Sequence[Tensor]],
        coefficients: Tensor,
    ) -> tuple[Tensor, ...]:
        if not basis:
            raise ValueError("directional basis cannot be empty")
        if coefficients.ndim != 1 or coefficients.numel() != len(basis):
            raise ValueError("coefficient count does not match basis")
        tensor_count = len(basis[0])
        if any(len(direction) != tensor_count for direction in basis):
            raise ValueError("directional basis tensors do not align")
        combined = [
            torch.zeros_like(value)
            for value in basis[0]
        ]
        for coefficient, direction in zip(
            coefficients,
            basis,
            strict=True,
        ):
            alpha = float(coefficient)
            for destination, value in zip(
                combined,
                direction,
                strict=True,
            ):
                destination.add_(value, alpha=alpha)
        return tuple(combined)

    def _combine_directional_basis(
        self,
        group: _ParameterGroup,
        snapshot: Sequence[Tensor],
        coefficients: Tensor,
        *,
        basis_epoch: int,
        tensor_magnitudes: Sequence[float],
    ) -> tuple[Tensor, ...]:
        """Regenerate and combine a basis with O(P), not O(KP), memory."""

        if coefficients.shape != (self.config.candidate_directions,):
            raise ValueError(
                "coefficient count does not match directional basis"
            )
        if len(tensor_magnitudes) != len(snapshot):
            raise ValueError(
                "tensor magnitudes do not match parameter snapshot"
            )
        combined = [torch.zeros_like(value) for value in snapshot]
        coefficient_values = coefficients.detach().cpu().tolist()
        for basis_index, alpha in enumerate(coefficient_values):
            alpha = float(alpha)
            if alpha == 0.0:
                continue
            direction = self._directional_basis_direction(
                group,
                snapshot,
                basis_index=basis_index,
                basis_epoch=basis_epoch,
                probe_scale=1.0,
                tensor_magnitudes=tensor_magnitudes,
            )
            for destination, value in zip(
                combined,
                direction,
                strict=True,
            ):
                destination.add_(value, alpha=alpha)
            del direction
        return tuple(combined)

    def _project_directional_coefficients(
        self,
        proposal: Tensor,
        sensitivity: Tensor,
        state: ProcoimEpisodeState,
    ) -> Tensor | None:
        """Project a directional Adam proposal onto linearized slot guards."""

        if proposal.ndim != 1:
            raise ValueError("directional Adam proposal must be one-dimensional")
        if sensitivity.shape != (*state.current_loss.shape, proposal.numel()):
            raise ValueError("directional sensitivity shape is invalid")
        flat_sensitivity = sensitivity.reshape(-1, proposal.numel())
        current = state.current_loss.reshape(-1)
        best = state.best_loss.reshape(-1)
        start = state.start_loss.reshape(-1)
        target = state.target_loss.reshape(-1)
        active = state.active_mask.reshape(-1)
        soft = state.soft_frozen_mask.reshape(-1)
        protected = state.protected_mask.reshape(-1)
        constraints: list[tuple[Tensor, Tensor]] = []

        if bool(active.any()):
            active_direction = flat_sensitivity[active].mean(dim=0)
            active_loss = current[active].mean().abs().clamp_min(
                self.config.gap_epsilon
            )
            constraints.append(
                (
                    active_direction,
                    -self._effective_reduction_fraction() * active_loss,
                )
            )
            span = (start - target).clamp_min(self.config.gap_epsilon)
            normalized_direction = (
                flat_sensitivity / span.unsqueeze(-1)
            )[active].mean(dim=0)
            constraints.append(
                (
                    normalized_direction,
                    proposal.new_tensor(-self.config.minimum_active_gain),
                )
            )
        else:
            return None

        gain = (start - best).clamp_min(0.0)
        soft_tolerance = (
            self.config.soft_regression_fraction * gain
            + self.config.absolute_regression_tolerance
        )
        hard_tolerance = (
            self.config.hard_regression_fraction * gain
            + self.config.absolute_regression_tolerance
        )
        for index in torch.nonzero(soft, as_tuple=False).flatten().tolist():
            constraints.append(
                (
                    flat_sensitivity[index],
                    best[index] + soft_tolerance[index] - current[index],
                )
            )
        for index in torch.nonzero(protected, as_tuple=False).flatten().tolist():
            constraints.append(
                (
                    flat_sensitivity[index],
                    best[index] + hard_tolerance[index] - current[index],
                )
            )

        coefficients = proposal.detach().clone()
        corrections = [
            torch.zeros_like(coefficients)
            for _ in constraints
        ]
        for _ in range(self.config.directional_adam_projection_iterations):
            for constraint_index, (normal, bound) in enumerate(constraints):
                normal_norm = normal.square().sum()
                if float(normal_norm) <= self.config.gap_epsilon:
                    if float(bound) < -self.config.gap_epsilon:
                        return None
                    continue
                shifted = coefficients + corrections[constraint_index]
                violation = normal.dot(shifted) - bound
                if float(violation) > 0.0:
                    projected = shifted - (
                        violation / normal_norm
                    ) * normal
                else:
                    projected = shifted
                corrections[constraint_index] = shifted - projected
                coefficients = projected

        if not bool(torch.isfinite(coefficients).all()):
            return None
        tolerance = 10.0 * self.config.gap_epsilon
        if any(
            float(normal.dot(coefficients) - bound) > tolerance
            for normal, bound in constraints
        ):
            return None
        return coefficients

    def _directional_adam_proposal(
        self,
        group: _ParameterGroup,
        gradient: Tensor,
        *,
        basis_epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        if gradient.shape != (self.config.candidate_directions,):
            raise ValueError("directional gradient has the wrong shape")
        if basis_epoch < 0:
            raise ValueError("basis_epoch must be nonnegative")
        device = gradient.device
        dtype = gradient.dtype
        old_m = self._directional_adam_m.get(group.name)
        old_v = self._directional_adam_v.get(group.name)
        if (old_m is None) != (old_v is None):
            raise RuntimeError("directional Adam moment state is incomplete")
        saved_epoch = self._directional_adam_basis_epochs.get(
            group.name,
            0,
        )
        same_basis = old_m is not None and saved_epoch == basis_epoch
        if not same_basis:
            old_m_device = torch.zeros_like(gradient)
            old_v_device = torch.zeros_like(gradient)
            previous_step = 0
        else:
            assert old_m is not None and old_v is not None
            old_m_device = old_m.to(device=device, dtype=dtype)
            old_v_device = old_v.to(device=device, dtype=dtype)
            previous_step = self._directional_adam_steps.get(group.name, 0)
        beta1 = self.config.directional_adam_beta1
        beta2 = self.config.directional_adam_beta2
        new_m = old_m_device.mul(beta1).add(
            gradient, alpha=1.0 - beta1
        )
        new_v = old_v_device.mul(beta2).addcmul(
            gradient,
            gradient,
            value=1.0 - beta2,
        )
        step = previous_step + 1
        m_hat = new_m / (1.0 - beta1 ** step)
        v_hat = new_v / (1.0 - beta2 ** step)
        proposal = -self.config.directional_adam_learning_rate * (
            m_hat / (v_hat.sqrt() + self.config.directional_adam_epsilon)
        )
        return proposal, new_m, new_v, step

    def _directional_decay_delta(
        self,
        group: _ParameterGroup,
        snapshot: Sequence[Tensor],
    ) -> tuple[tuple[Tensor, ...], int]:
        accepted_step = self.total_accepted_updates + 1
        previous = self._directional_adam_last_decay_step.get(group.name, 0)
        elapsed = max(accepted_step - previous, 1)
        base = (
            1.0
            - self.config.directional_adam_learning_rate
            * self.config.directional_adam_weight_decay
        )
        factor = base ** elapsed
        return (
            tuple(value.mul(factor - 1.0) for value in snapshot),
            accepted_step,
        )

    def _commit_directional_adam(
        self,
        commit: _DirectionalAdamCommit,
    ) -> None:
        self._directional_adam_m[commit.group_name] = (
            commit.first_moment.detach().cpu().to(torch.float64).clone()
        )
        self._directional_adam_v[commit.group_name] = (
            commit.second_moment.detach().cpu().to(torch.float64).clone()
        )
        self._directional_adam_steps[commit.group_name] = commit.step
        self._directional_adam_last_decay_step[commit.group_name] = (
            commit.last_decay_step
        )
        self._directional_adam_basis_epochs[commit.group_name] = (
            commit.basis_epoch
        )

    def _momentum_direction(
        self,
        group: _ParameterGroup,
        snapshot: Sequence[Tensor],
        *,
        probe_scale: float,
    ) -> tuple[Tensor, ...] | None:
        if not self.config.reuse_successful_directions:
            return None
        momentum = self._direction_momentum.get(group.name)
        if momentum is None:
            return None
        return self._rescale_direction(
            snapshot,
            momentum,
            probe_scale=probe_scale,
        )

    def _update_direction_momentum(
        self,
        group: _ParameterGroup,
        direction: Sequence[Tensor],
        sign: int,
    ) -> None:
        if not self.config.reuse_successful_directions:
            return
        signed = tuple(value.mul(sign) for value in direction)
        normalized = self._normalize_direction(signed)
        if normalized is None:
            return
        previous = self._direction_momentum.get(group.name)
        if previous is None:
            updated = normalized
        else:
            beta = self.config.direction_momentum
            updated = tuple(
                old.mul(beta).add(new, alpha=1.0 - beta)
                for old, new in zip(previous, normalized, strict=True)
            )
            renormalized = self._normalize_direction(updated)
            if renormalized is not None:
                updated = renormalized
        self._direction_momentum[group.name] = tuple(
            value.detach().clone() for value in updated
        )

    def _accumulate_consolidation_window(
        self,
        group: _ParameterGroup,
        direction: Sequence[Tensor],
        sign: int,
    ) -> None:
        """Add one accepted update's applied delta into the current
        consolidation window, mirroring how an AdamW ``interval_delta`` is
        the net parameter change over a fixed step window."""

        if self.config.directional_basis_consolidation_cosine_margin is None:
            return
        signed = tuple(value.detach().mul(sign) for value in direction)
        accum = self._group_window_accum.get(group.name)
        if accum is None:
            self._group_window_accum[group.name] = tuple(
                value.clone() for value in signed
            )
        else:
            self._group_window_accum[group.name] = tuple(
                old.add(new)
                for old, new in zip(accum, signed, strict=True)
            )

    def _maybe_gate_consolidation(self, group: _ParameterGroup) -> None:
        """At each rotation-window boundary, compare this window's
        accepted-direction vector against the previous window's and defer
        the basis rotation by one window if the two are unusually aligned.

        This targets what
        ``Papers/PROCOIM_ADAMW_WEIGHT_GEOMETRY_REPORT.md``'s breakout trace
        found: AdamW's own update direction is uncorrelated between
        consecutive intervals almost everywhere, except for brief spikes
        that land exactly where a stalled output position locks in. Those
        rare aligned windows are read here as "this group is actively
        consolidating," and the rotating basis is held in place through
        them instead of rotating on the usual fixed schedule.
        """

        margin = self.config.directional_basis_consolidation_cosine_margin
        if margin is None:
            return
        natural = self._natural_basis_epoch(group)
        last_seen = self._group_last_natural_epoch.get(group.name, 0)
        if natural <= last_seen:
            return
        accum = self._group_window_accum.pop(group.name, None)
        if accum is not None:
            previous = self._group_prev_window_vector.get(group.name)
            if previous is not None:
                flat_accum = torch.cat(
                    [value.flatten() for value in accum]
                )
                flat_previous = torch.cat(
                    [value.flatten() for value in previous]
                )
                denom = (
                    flat_accum.norm().item() * flat_previous.norm().item()
                )
                if denom > 0:
                    cosine = (flat_accum @ flat_previous).item() / denom
                    baseline = self._group_window_cosine_baseline.get(
                        group.name
                    )
                    if baseline is not None and cosine >= baseline + margin:
                        self._group_epoch_freeze[group.name] = (
                            self._group_epoch_freeze.get(group.name, 0)
                            + (natural - last_seen)
                        )
                        self.total_consolidation_freezes += 1
                    beta = self.config.directional_basis_consolidation_ema_beta
                    self._group_window_cosine_baseline[group.name] = (
                        cosine
                        if baseline is None
                        else beta * baseline + (1.0 - beta) * cosine
                    )
            self._group_prev_window_vector[group.name] = accum
        self._group_last_natural_epoch[group.name] = natural

    def _select_group(
        self,
        groups: Sequence[_ParameterGroup],
    ) -> _ParameterGroup:
        if not self.config.adaptive_group_selection:
            return groups[self._next_group_index % len(groups)]

        for group in groups:
            if self._group_bank_counts[group.name] == 0:
                return group
        total_banks = sum(self._group_bank_counts.values())
        log_total = math.log(total_banks + 1.0)
        scored = []
        for schedule_index, group in enumerate(groups):
            count = self._group_bank_counts[group.name]
            exploration = self.config.group_exploration * math.sqrt(
                log_total / count
            )
            score = self._group_gain_ema[group.name] + exploration
            scored.append((score, -schedule_index, group))
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def _record_group_bank(
        self,
        group: _ParameterGroup,
        bank: _CandidateBankResult,
    ) -> None:
        reward = 0.0
        if bank.winner is not None:
            reward = max(bank.winner.utility, 0.0) / max(
                bank.search_forwards,
                1,
            )
        beta = self.config.group_score_beta
        previous = self._group_gain_ema[group.name]
        self._group_gain_ema[group.name] = (
            beta * previous + (1.0 - beta) * reward
        )
        self._group_bank_counts[group.name] += 1
        self._group_basis_forward_counts[group.name] += bank.search_forwards

        if self.config.directional_basis_rotation_stall_banks is not None:
            if bank.winner is None:
                streak = self._group_basis_stall_streak[group.name] + 1
                if streak >= self.config.directional_basis_rotation_stall_banks:
                    self._group_basis_stall_epochs[group.name] += 1
                    streak = 0
                self._group_basis_stall_streak[group.name] = streak
            else:
                # A success means the current basis is working; only a
                # stall streak should trigger a fresh subspace.
                self._group_basis_stall_streak[group.name] = 0

        if self.config.adaptive_probe_scale:
            scale = self._group_probe_scales[group.name]
            if bank.winner is None:
                scale *= self.config.probe_scale_decay
            else:
                scale *= self.config.probe_scale_growth
            self._group_probe_scales[group.name] = min(
                self.config.maximum_probe_scale,
                max(self.config.minimum_probe_scale, scale),
            )

        if self.config.adaptive_candidate_directions:
            if bank.winner is None:
                # A small probe count that already found nothing tells us
                # nothing useful; widen the search before another bank.
                current = self._group_candidate_directions[group.name]
                expanded = math.ceil(
                    current * self.config.candidate_directions_growth
                )
                self._group_candidate_directions[group.name] = min(
                    self.config.candidate_directions,
                    max(self.config.minimum_candidate_directions, expanded),
                )
            else:
                # A cheap narrow search already satisfied the acceptance
                # contract, so the next bank should start small again.
                self._group_candidate_directions[group.name] = (
                    self.config.minimum_candidate_directions
                )

        self._maybe_gate_consolidation(group)

    def _effective_reduction_fraction(self) -> float:
        """Return the current per-slot CE-reduction bar.

        Static unless ``adaptive_reduction_fraction`` is enabled, in which
        case the bar decays logarithmically with the number of completed
        episodes: early episodes demand the full requested reduction, while
        later episodes accept progressively smaller gains as the remaining
        headroom shrinks.
        """

        if not self.config.adaptive_reduction_fraction:
            return self.config.reduction_fraction
        decayed = self.config.reduction_fraction / (
            1.0
            + self.config.reduction_fraction_decay_rate
            * math.log1p(self.episodes)
        )
        return max(self.config.minimum_reduction_fraction, decayed)

    def initialize_episode_state(
        self,
        baseline: SlotLossResult,
    ) -> ProcoimEpisodeState:
        """Create immutable targets and mutable slot anchors for an episode."""

        baseline.validate()
        start = baseline.losses.detach().clone()
        target = torch.maximum(
            torch.full_like(start, self.config.loss_floor),
            (1.0 - self._effective_reduction_fraction()) * start,
        )
        status = torch.full_like(start, int(SlotStatus.ACTIVE), dtype=torch.int8)
        stable = torch.zeros_like(start, dtype=torch.long)
        initially_achieved = baseline.valid_mask & start.le(target)
        status[initially_achieved] = int(SlotStatus.SOFT_FROZEN)
        stable[initially_achieved] = 1
        if self.config.stability_required <= 1:
            status[initially_achieved] = int(SlotStatus.PROTECTED)
        return ProcoimEpisodeState(
            start_loss=start,
            target_loss=target,
            best_loss=start.clone(),
            current_loss=start.clone(),
            valid_mask=baseline.valid_mask.detach().clone(),
            status=status,
            stable_count=stable,
            current_guard_loss=float(
                baseline.aggregate
                if baseline.guard_aggregate is None
                else baseline.guard_aggregate
            ),
        )

    def normalized_gap(
        self,
        losses: Tensor,
        state: ProcoimEpisodeState,
    ) -> Tensor:
        denominator = (state.start_loss - state.target_loss).clamp_min(
            self.config.gap_epsilon
        )
        return ((losses - state.target_loss) / denominator).clamp_min(0.0)

    def score_candidate(
        self,
        state: ProcoimEpisodeState,
        candidate: SlotLossResult,
        *,
        group_name: str,
        direction_index: int,
        sign: int,
    ) -> CandidateResult:
        """Score a candidate against one frozen incumbent slot state."""

        candidate.validate()
        if not torch.equal(candidate.valid_mask, state.valid_mask):
            raise ValueError("candidate valid_mask changed inside the episode")
        active = state.active_mask
        zero = state.current_loss.new_zeros(())
        if bool(active.any()):
            current_gap = self.normalized_gap(
                state.current_loss, state
            )[active].mean()
            candidate_gap = self.normalized_gap(
                candidate.losses, state
            )[active].mean()
            active_gain_tensor = current_gap - candidate_gap
            current_active_loss = state.current_loss[active].mean()
            candidate_active_loss = candidate.losses[active].mean()
            relative_active_gain_tensor = (
                (current_active_loss - candidate_active_loss)
                / current_active_loss.abs().clamp_min(self.config.gap_epsilon)
            )
        else:
            active_gain_tensor = zero
            relative_active_gain_tensor = zero

        gain = (state.start_loss - state.best_loss).clamp_min(0.0)
        soft_tolerance = (
            self.config.soft_regression_fraction * gain
            + self.config.absolute_regression_tolerance
        )
        hard_tolerance = (
            self.config.hard_regression_fraction * gain
            + self.config.absolute_regression_tolerance
        )
        protected_objectives = state.soft_frozen_mask | state.protected_mask
        span = (state.start_loss - state.target_loss).clamp_min(
            self.config.gap_epsilon
        )
        if bool(protected_objectives.any()):
            raw_regression = (
                candidate.losses - state.best_loss - soft_tolerance
            ).clamp_min(0.0)
            soft_penalty_tensor = (
                raw_regression / span
            )[protected_objectives].mean()
        else:
            soft_penalty_tensor = zero
        hard_violation = bool(
            (
                state.protected_mask
                & candidate.losses.gt(state.best_loss + hard_tolerance)
            ).any()
        )
        global_scale = span[state.valid_mask].mean()
        candidate_aggregate = candidate.losses[state.valid_mask].mean()
        current_aggregate = state.current_loss[state.valid_mask].mean()
        candidate_guard = (
            candidate_aggregate
            if candidate.guard_aggregate is None
            else candidate.guard_aggregate
        )
        current_guard = (
            current_aggregate
            if state.current_guard_loss is None
            else current_aggregate.new_tensor(state.current_guard_loss)
        )
        global_regression_tensor = (
            (candidate_guard - current_guard).clamp_min(0.0)
            / global_scale
        )
        utility_tensor = (
            active_gain_tensor
            - self.config.protection_weight * soft_penalty_tensor
            - self.config.global_regression_weight * global_regression_tensor
        )
        active_gain = float(active_gain_tensor)
        relative_active_gain = float(relative_active_gain_tensor)
        utility = float(utility_tensor)
        eligible = (
            not hard_violation
            and bool(active.any())
            and active_gain >= self.config.minimum_active_gain
            and relative_active_gain >= self._effective_reduction_fraction()
            and utility > 0.0
        )
        return CandidateResult(
            group_name=group_name,
            direction_index=direction_index,
            sign=sign,
            losses=candidate.losses.detach().clone(),
            aggregate=float(candidate_aggregate),
            active_gain=active_gain,
            soft_penalty=float(soft_penalty_tensor),
            global_regression=float(global_regression_tensor),
            utility=utility,
            hard_violation=hard_violation,
            eligible=eligible,
            guard_aggregate=float(candidate_guard),
        )

    def _score_anchor_regression(
        self,
        candidate: CandidateResult,
        anchors: Sequence[ProcoimAnchor],
        results: Sequence[SlotLossResult],
        *,
        audit: bool = False,
    ) -> CandidateResult:
        if len(anchors) != len(results):
            raise ValueError("anchor result count does not match anchors")
        if not anchors:
            return candidate

        penalties: list[float] = []
        violation = False
        anchor_losses: list[Tensor] = []
        aggregates: list[float] = []
        retained_improvements: list[float] = []
        retained_baselines: list[float] = []
        retained = self.config.anchor_minimum_retained_reduction_fraction
        for anchor, result in zip(anchors, results, strict=True):
            result.validate()
            aggregates.append(float(result.aggregate))
            anchor_losses.append(result.losses.detach().clone())
            if anchor.slotwise:
                assert anchor.start_losses is not None
                assert anchor.best_losses is not None
                assert anchor.valid_mask is not None
                assert anchor.protected_mask is not None
                assert anchor.hard_mask is not None
                if result.losses.shape != anchor.best_losses.shape:
                    raise ValueError("anchor candidate loss shape changed")
                if not torch.equal(result.valid_mask, anchor.valid_mask):
                    raise ValueError("anchor valid_mask changed")
                protected = anchor.protected_mask & anchor.valid_mask
                saved_gain = (
                    anchor.start_losses - anchor.best_losses
                ).clamp_min(0.0)
                # Empty/padded and zero-gain slots cannot create replay
                # penalties or vetoes.
                protected &= saved_gain.gt(self.config.gap_epsilon)
                if not bool(protected.any()):
                    penalties.append(0.0)
                    continue
                soft_fraction = (
                    self.config.anchor_soft_regression_fraction
                    if anchor.soft_regression_fraction is None
                    else anchor.soft_regression_fraction
                )
                hard_fraction = (
                    self.config.anchor_hard_regression_fraction
                    if anchor.hard_regression_fraction is None
                    else anchor.hard_regression_fraction
                )
                absolute_tolerance = (
                    self.config.anchor_absolute_tolerance
                    if anchor.absolute_tolerance is None
                    else anchor.absolute_tolerance
                )
                soft_tolerance = (
                    soft_fraction * saved_gain + absolute_tolerance
                )
                hard_tolerance = (
                    hard_fraction * saved_gain + absolute_tolerance
                )
                hard_ceiling = anchor.best_losses + hard_tolerance
                if audit and retained is not None:
                    # Consolidation is distributional: permit individual
                    # anchors to trade a small amount of progress while the
                    # audited bank retains the requested net reduction.
                    # A separate per-slot ceiling still catches catastrophic
                    # regressions above the pre-episode baseline.
                    audit_catastrophe_tolerance = (
                        hard_fraction * anchor.start_losses.abs()
                        + absolute_tolerance
                    )
                    hard_ceiling = (
                        anchor.start_losses + audit_catastrophe_tolerance
                    )
                    retained_improvements.append(
                        float(
                            (
                                anchor.start_losses - result.losses
                            )[protected].sum()
                        )
                    )
                    retained_baselines.append(
                        float(anchor.start_losses[protected].abs().sum())
                    )
                excess = (
                    result.losses - anchor.best_losses - soft_tolerance
                ).clamp_min(0.0)
                penalties.append(
                    float(
                        (excess / saved_gain.clamp_min(
                            self.config.gap_epsilon
                        ))[protected].mean()
                    )
                )
                if self.config.anchor_hard_protection and bool(
                    (
                        anchor.hard_mask
                        & result.losses.gt(hard_ceiling)
                    ).any()
                ):
                    violation = True
                continue

            reference = anchor.best_aggregate
            if reference is None:
                raise RuntimeError("legacy anchor reference was not initialized")
            aggregate = float(result.aggregate)
            soft_fraction = (
                self.config.anchor_regression_fraction
                if anchor.soft_regression_fraction is None
                else anchor.soft_regression_fraction
            )
            hard_fraction = (
                self.config.anchor_regression_fraction
                if anchor.hard_regression_fraction is None
                else anchor.hard_regression_fraction
            )
            absolute_tolerance = (
                self.config.anchor_absolute_tolerance
                if anchor.absolute_tolerance is None
                else anchor.absolute_tolerance
            )
            soft_tolerance = soft_fraction * abs(reference) + absolute_tolerance
            hard_tolerance = hard_fraction * abs(reference) + absolute_tolerance
            excess = max(aggregate - reference - soft_tolerance, 0.0)
            penalties.append(
                excess / max(abs(reference), self.config.gap_epsilon)
            )
            if self.config.anchor_hard_protection and (
                not math.isfinite(aggregate)
                or aggregate > reference + hard_tolerance
            ):
                violation = True

        retained_reduction: float | None = None
        if audit and retained is not None and retained_baselines:
            retained_reduction = math.fsum(retained_improvements) / max(
                math.fsum(retained_baselines),
                self.config.gap_epsilon,
            )
            if (
                not math.isfinite(retained_reduction)
                or retained_reduction < retained
            ):
                violation = True

        penalty = math.fsum(penalties) / len(penalties)
        utility = candidate.utility - self.config.anchor_weight * penalty
        eligible = (
            candidate.eligible
            and not violation
            and math.isfinite(utility)
            and utility > 0.0
        )
        return replace(
            candidate,
            utility=utility,
            eligible=eligible,
            anchor_penalty=penalty,
            anchor_violation=violation,
            anchor_aggregates=tuple(float(value) for value in aggregates),
            anchor_losses=tuple(anchor_losses),
            anchor_retained_reduction=retained_reduction,
        )

    def audit_anchors(
        self,
        model: nn.Module,
        loss_provider: LossProvider,
        anchors: Sequence[ProcoimAnchor],
        *,
        forward_kwargs: Mapping[str, Any] | None = None,
    ) -> ProcoimAnchorAuditResult:
        """Evaluate a consolidation gate with deterministic counted forwards."""

        anchors = tuple(anchors)
        if any(not isinstance(anchor, ProcoimAnchor) for anchor in anchors):
            raise TypeError("anchors must contain only ProcoimAnchor objects")
        if not anchors:
            return ProcoimAnchorAuditResult(0, 0.0, False, (), ())
        kwargs = {} if forward_kwargs is None else dict(forward_kwargs)
        self._assert_no_gradients(model)
        module_modes = {
            module: module.training
            for module in model.modules()
        }
        model.eval()
        external_rng = _capture_global_rng()
        search_rng = _capture_global_rng()
        buffer_snapshot = self._snapshot_buffers(model)
        results: list[SlotLossResult] = []
        try:
            for anchor in anchors:
                self.total_anchor_forwards += 1
                results.append(
                    self._evaluate(
                        model,
                        anchor.batch,
                        loss_provider,
                        (
                            kwargs
                            if anchor.forward_kwargs is None
                            else anchor.forward_kwargs
                        ),
                        search_rng=search_rng,
                        buffer_snapshot=buffer_snapshot,
                    )
                )
            placeholder = CandidateResult(
                group_name="retention_audit",
                direction_index=-1,
                sign=0,
                losses=results[0].losses.detach().clone(),
                aggregate=float(results[0].aggregate),
                active_gain=0.0,
                soft_penalty=0.0,
                global_regression=0.0,
                utility=1.0e30,
                hard_violation=False,
                eligible=True,
                source="random",
            )
            scored = self._score_anchor_regression(
                placeholder,
                anchors,
                results,
                audit=True,
            )
            return ProcoimAnchorAuditResult(
                anchor_forwards=len(results),
                penalty=scored.anchor_penalty,
                hard_violation=scored.anchor_violation,
                aggregates=scored.anchor_aggregates,
                losses=scored.anchor_losses,
                retained_reduction=scored.anchor_retained_reduction,
            )
        finally:
            self._restore_buffers(model, buffer_snapshot)
            _restore_global_rng(external_rng)
            for module, training in module_modes.items():
                module.training = training
            self._assert_no_gradients(model)

    def _antithetic_smooth_coefficient(
        self,
        state: ProcoimEpisodeState,
        positive: CandidateResult,
        negative: CandidateResult,
        *,
        active_mask: Tensor | None = None,
    ) -> float:
        """Return a finite-difference weight from the unsaturated objective.

        Normalized gaps are ideal for episode targets, but their clamp at zero
        makes them a poor measurement objective when a probe is much larger
        than the requested reduction. Smooth active CE preserves directional
        information; protection and replay terms retain their original signs.
        Final candidates still pass the ordinary normalized-gap threshold.
        """

        active = state.active_mask
        if active_mask is not None:
            if active_mask.shape != active.shape:
                raise ValueError(
                    "antithetic active mask must match episode slots"
                )
            if active_mask.dtype is not torch.bool:
                raise TypeError("antithetic active mask must be boolean")
            active = active & active_mask
        if not bool(active.any()):
            return 0.0
        scale = float(
            state.current_loss[active]
            .mean()
            .abs()
            .clamp_min(self.config.gap_epsilon)
        )
        coefficient = (
            float(negative.losses[active].mean())
            - float(positive.losses[active].mean())
        ) / scale
        coefficient += self.config.protection_weight * (
            negative.soft_penalty - positive.soft_penalty
        )
        coefficient += self.config.global_regression_weight * (
            negative.global_regression - positive.global_regression
        )
        coefficient += self.config.anchor_weight * (
            negative.anchor_penalty - positive.anchor_penalty
        )
        return coefficient

    def _antithetic_composite(
        self,
        snapshot: Sequence[Tensor],
        pairs: Sequence[
            tuple[
                tuple[Tensor, ...],
                CandidateResult,
                CandidateResult,
            ]
        ],
        state: ProcoimEpisodeState,
        *,
        step_scale: float,
        active_mask: Tensor | None = None,
    ) -> tuple[Tensor, ...] | None:
        combined = [torch.zeros_like(value) for value in snapshot]
        useful = False
        for direction, positive, negative in pairs:
            if positive.error is not None or negative.error is not None:
                continue
            coefficient = self._antithetic_smooth_coefficient(
                state,
                positive,
                negative,
                active_mask=active_mask,
            )
            if not math.isfinite(coefficient) or coefficient == 0.0:
                continue
            useful = True
            for destination, value in zip(
                combined,
                direction,
                strict=True,
            ):
                destination.add_(value, alpha=coefficient)
        if not useful:
            return None
        return self._rescale_direction(
            snapshot,
            combined,
            probe_scale=step_scale,
        )

    def update_slot_states(
        self,
        state: ProcoimEpisodeState,
        accepted_losses: Tensor,
        accepted_guard_loss: float | None = None,
    ) -> None:
        """Atomically advance anchors and status after an accepted update.

        A SOFT_FROZEN slot that rises above target becomes ACTIVE again. This
        closes an otherwise stranded state in which the slot is unresolved
        but no longer participates in exploration.
        """

        if accepted_losses.shape != state.current_loss.shape:
            raise ValueError("accepted losses must match episode slot shape")
        if not bool(torch.isfinite(accepted_losses[state.valid_mask]).all()):
            raise FloatingPointError("accepted losses must be finite")
        state.current_loss[state.valid_mask] = accepted_losses[state.valid_mask]
        if accepted_guard_loss is None:
            state.current_guard_loss = float(
                accepted_losses[state.valid_mask].mean()
            )
        elif not math.isfinite(accepted_guard_loss):
            raise FloatingPointError("accepted guard loss must be finite")
        else:
            state.current_guard_loss = float(accepted_guard_loss)
        improved = state.valid_mask & accepted_losses.lt(state.best_loss)
        state.best_loss[improved] = accepted_losses[improved]

        soft = state.soft_frozen_mask
        regressed = soft & accepted_losses.gt(state.target_loss)
        state.status[regressed] = int(SlotStatus.ACTIVE)
        state.stable_count[regressed] = 0

        remaining_soft = state.soft_frozen_mask
        state.stable_count[remaining_soft] += 1
        newly_solved = state.active_mask & accepted_losses.le(state.target_loss)
        state.status[newly_solved] = int(SlotStatus.SOFT_FROZEN)
        state.stable_count[newly_solved] = 1

        ready = (
            state.soft_frozen_mask
            & state.stable_count.ge(self.config.stability_required)
        )
        state.status[ready] = int(SlotStatus.PROTECTED)

    def _episode_complete(
        self,
        state: ProcoimEpisodeState,
        start_aggregate: float,
    ) -> bool:
        if self.config.completion_mode == "budget":
            return False
        if self.config.completion_mode == "all_slots":
            return bool(
                (
                    ~state.valid_mask
                    | state.status.eq(int(SlotStatus.PROTECTED))
                ).all()
            )
        current = float(state.current_loss[state.valid_mask].mean())
        aggregate_target = max(
            self.config.loss_floor,
            (1.0 - self._effective_reduction_fraction()) * start_aggregate,
        )
        protected_fraction = float(
            state.protected_mask.sum() / state.valid_mask.sum()
        )
        return (
            current <= aggregate_target
            and protected_fraction >= self.config.protected_fraction
        )

    def _evaluate(
        self,
        model: nn.Module,
        batch: Any,
        loss_provider: LossProvider,
        forward_kwargs: Mapping[str, Any],
        *,
        search_rng: _GlobalRNGState,
        buffer_snapshot: Mapping[str, Tensor],
    ) -> SlotLossResult:
        self._restore_buffers(model, buffer_snapshot)
        _restore_global_rng(search_rng)
        self._assert_no_gradients(model)
        parameter_versions = {
            id(parameter): parameter._version
            for parameter in model.parameters()
        }
        self.total_forward_calls += 1
        with torch.inference_mode():
            result = loss_provider(model, batch, forward_kwargs)
        mutated = [
            name
            for name, parameter in model.named_parameters()
            if parameter._version != parameter_versions[id(parameter)]
        ]
        if mutated:
            preview = ", ".join(mutated[:8])
            raise RuntimeError(
                "loss_provider mutated model parameters during a forward: "
                f"{preview}"
            )
        if not isinstance(result, SlotLossResult):
            raise TypeError("loss_provider must return SlotLossResult")
        result.validate()
        if self.config.slot_objective_reduction == "batch_mean":
            result = batch_mean_slot_loss(result)
        self._assert_no_gradients(model)
        return result

    def _evaluate_objective(
        self,
        model: nn.Module,
        batches: Sequence[Any],
        loss_provider: LossProvider,
        forward_kwargs: Mapping[str, Any],
        *,
        search_rng: _GlobalRNGState,
        buffer_snapshot: Mapping[str, Tensor],
    ) -> SlotLossResult:
        """Average one search objective over counted sequential microbatches."""

        batches = tuple(batches)
        if not batches:
            raise ValueError("search objective requires at least one batch")
        results = tuple(
            self._evaluate(
                model,
                batch,
                loss_provider,
                forward_kwargs,
                search_rng=search_rng,
                buffer_snapshot=buffer_snapshot,
            )
            for batch in batches
        )
        if len(results) == 1:
            return results[0]
        if self.config.slot_objective_reduction != "batch_mean":
            raise RuntimeError(
                "multi-batch objectives require batch_mean slot reduction"
            )
        first = results[0]
        for result in results[1:]:
            if result.losses.shape != first.losses.shape:
                raise ValueError(
                    "accumulation batch changed reduced slot-loss shape"
                )
            if not torch.equal(result.valid_mask, first.valid_mask):
                raise ValueError(
                    "accumulation batch changed reduced valid slot mask"
                )
        losses = torch.stack(
            [result.losses for result in results],
            dim=0,
        ).mean(dim=0).detach()
        valid = first.valid_mask.detach().clone()
        aggregate = losses[valid].mean().detach()
        guard_values = [result.guard_aggregate for result in results]
        if all(value is None for value in guard_values):
            guard = None
        elif any(value is None for value in guard_values):
            raise ValueError(
                "accumulation batches disagree on guard aggregate presence"
            )
        else:
            guard = torch.stack(
                [value for value in guard_values if value is not None]
            ).mean().detach()
        return SlotLossResult(
            losses,
            valid,
            aggregate,
            guard,
        ).validate()

    def evaluate_candidate_bank(
        self,
        model: nn.Module,
        batch: Any,
        loss_provider: LossProvider,
        forward_kwargs: Mapping[str, Any],
        state: ProcoimEpisodeState,
        group: _ParameterGroup,
        *,
        probe_scale: float,
        search_rng: _GlobalRNGState,
        buffer_snapshot: Mapping[str, Tensor],
        anchors: Sequence[ProcoimAnchor] = (),
        accumulation_batches: Sequence[Any] = (),
        reference_batch: Any | None = None,
        reference_forward_kwargs: Mapping[str, Any] | None = None,
        reference_current_loss: float | None = None,
        reference_reduction_fraction: float | None = None,
        max_candidate_forwards: int | None = None,
        probe_directions: int | None = None,
    ) -> _CandidateBankResult:
        """Evaluate one antithetic bank transactionally from one exact block.

        ``probe_directions`` bounds how many of the ``candidate_directions``
        basis slots are actually probed with forward passes this bank (used
        by ``adaptive_candidate_directions`` to search narrow before wide).
        It defaults to the full declared width.
        """

        if max_candidate_forwards is not None and max_candidate_forwards < 0:
            raise ValueError("max_candidate_forwards must be nonnegative or None")
        if probe_directions is None:
            probe_directions = self.config.candidate_directions
        elif not 1 <= probe_directions <= self.config.candidate_directions:
            raise ValueError(
                "probe_directions must be in [1, candidate_directions]"
            )
        search_batches = (batch, *tuple(accumulation_batches))
        objective_forwards = len(search_batches)
        if objective_forwards != self.config.directional_accumulation_batches:
            raise ValueError(
                "search batch count does not match "
                "directional_accumulation_batches"
            )
        if (reference_batch is None) != (reference_current_loss is None):
            raise ValueError(
                "reference_batch and reference_current_loss must be provided together"
            )
        if reference_current_loss is not None and (
            not math.isfinite(reference_current_loss)
            or reference_current_loss < 0
        ):
            raise ValueError(
                "reference_current_loss must be finite and nonnegative"
            )
        reference_kwargs = (
            forward_kwargs
            if reference_forward_kwargs is None
            else reference_forward_kwargs
        )
        required_reference_gain = (
            self._effective_reduction_fraction()
            if reference_reduction_fraction is None
            else reference_reduction_fraction
        )
        if (
            not math.isfinite(required_reference_gain)
            or not -1 <= required_reference_gain <= 1
        ):
            raise ValueError(
                "reference_reduction_fraction must be finite and in [-1, 1]"
            )

        base = self._snapshot_group(group)
        winner: CandidateResult | None = None
        winner_direction: tuple[Tensor, ...] | None = None
        candidate_forwards = 0
        anchor_forwards = 0
        hard_vetoes = 0
        anchor_vetoes = 0
        reference_forwards = 0
        reference_vetoes = 0
        antithetic_pairs: list[
            tuple[
                tuple[Tensor, ...],
                CandidateResult,
                CandidateResult,
            ]
        ] = []

        def spent_forwards() -> int:
            return (
                candidate_forwards
                + anchor_forwards
                + reference_forwards
            )

        def evaluate_direction(
            direction: tuple[Tensor, ...],
            *,
            direction_index: int,
            sign: int,
            source: CandidateSource,
            check_anchors: bool,
            check_reference: bool,
            score_anchors_without_current_gate: bool = False,
            step_fraction: float = 1.0,
        ) -> CandidateResult:
            nonlocal candidate_forwards, anchor_forwards
            nonlocal reference_forwards, reference_vetoes
            self._restore_group(group, base)
            self._apply_direction(group, direction, sign)
            candidate_forwards += objective_forwards
            self.total_candidate_forwards += objective_forwards
            try:
                candidate = self._evaluate_objective(
                    model,
                    search_batches,
                    loss_provider,
                    forward_kwargs,
                    search_rng=search_rng,
                    buffer_snapshot=buffer_snapshot,
                )
                scored = self.score_candidate(
                    state,
                    candidate,
                    group_name=group.name,
                    direction_index=direction_index,
                    sign=sign,
                )
                scored = replace(
                    scored,
                    source=source,
                    step_fraction=step_fraction,
                )
                if (
                    reference_batch is not None
                    and check_reference
                    and scored.eligible
                ):
                    reference_forwards += 1
                    self.total_reference_forwards += 1
                    reference_result = self._evaluate(
                        model,
                        reference_batch,
                        loss_provider,
                        reference_kwargs,
                        search_rng=search_rng,
                        buffer_snapshot=buffer_snapshot,
                    )
                    assert reference_current_loss is not None
                    reference_aggregate = float(reference_result.aggregate)
                    reference_gain = (
                        reference_current_loss - reference_aggregate
                    ) / max(
                        abs(reference_current_loss),
                        self.config.gap_epsilon,
                    )
                    reference_violation = (
                        reference_gain < required_reference_gain
                    )
                    if reference_violation:
                        reference_vetoes += 1
                        self.total_reference_vetoes += 1
                    scored = replace(
                        scored,
                        eligible=(
                            scored.eligible and not reference_violation
                        ),
                        reference_aggregate=reference_aggregate,
                        reference_gain=reference_gain,
                        reference_violation=reference_violation,
                    )
                # Ordinary candidates reach replay only after Gate 1 and the
                # population reference guard. Anchor-aware finite-difference
                # probes are measurements,
                # not commit candidates, so they may include replay utility
                # before they individually clear the current threshold.
                if anchors and check_anchors and (
                    scored.eligible or score_anchors_without_current_gate
                ):
                    anchor_results: list[SlotLossResult] = []
                    for anchor in anchors:
                        anchor_forwards += 1
                        self.total_anchor_forwards += 1
                        anchor_result = self._evaluate(
                            model,
                            anchor.batch,
                            loss_provider,
                            (
                                forward_kwargs
                                if anchor.forward_kwargs is None
                                else anchor.forward_kwargs
                            ),
                            search_rng=search_rng,
                            buffer_snapshot=buffer_snapshot,
                        )
                        anchor_results.append(anchor_result)
                    scored = self._score_anchor_regression(
                        scored,
                        anchors,
                        anchor_results,
                    )
                return scored
            except FloatingPointError as error:
                return CandidateResult(
                    group_name=group.name,
                    direction_index=direction_index,
                    sign=sign,
                    losses=state.current_loss.detach().clone(),
                    aggregate=float("inf"),
                    active_gain=-math.inf,
                    soft_penalty=math.inf,
                    global_regression=math.inf,
                    utility=-math.inf,
                    hard_violation=False,
                    eligible=False,
                    source=source,
                    error=str(error),
                )

        def consider(
            scored: CandidateResult,
            direction: tuple[Tensor, ...],
        ) -> bool:
            nonlocal winner, winner_direction, hard_vetoes, anchor_vetoes
            if scored.hard_violation or scored.anchor_violation:
                hard_vetoes += 1
            if scored.anchor_violation:
                anchor_vetoes += 1
                self.total_anchor_vetoes += 1
            if scored.eligible and winner is None:
                winner = scored
                winner_direction = tuple(
                    value.detach().clone() for value in direction
                )
                return True
            return False

        try:
            if self.config.directional_adam:
                direction_count = probe_directions
                probe_cost = 2 * direction_count * objective_forwards
                if (
                    max_candidate_forwards is not None
                    and spent_forwards() + probe_cost
                    > max_candidate_forwards
                ):
                    return _CandidateBankResult(
                        winner=None,
                        winner_direction=None,
                        candidate_forwards=0,
                        anchor_forwards=0,
                        hard_vetoes=0,
                        anchor_vetoes=0,
                    )

                basis_epoch = self._directional_basis_epoch(group)
                probe_magnitudes = tuple(
                    self._tensor_probe_magnitude(original, probe_scale)
                    for original in base
                )
                unit_magnitudes = tuple(
                    value / probe_scale for value in probe_magnitudes
                )
                positive_losses: list[Tensor] = []
                negative_losses: list[Tensor] = []
                for direction_index in range(direction_count):
                    probe_direction = self._directional_basis_direction(
                        group,
                        base,
                        basis_index=direction_index,
                        basis_epoch=basis_epoch,
                        probe_scale=probe_scale,
                        tensor_magnitudes=probe_magnitudes,
                    )
                    positive = evaluate_direction(
                        probe_direction,
                        direction_index=direction_index,
                        sign=1,
                        source="directional_adam",
                        check_anchors=False,
                        check_reference=False,
                    )
                    negative = evaluate_direction(
                        probe_direction,
                        direction_index=direction_index,
                        sign=-1,
                        source="directional_adam",
                        check_anchors=False,
                        check_reference=False,
                    )
                    positive_losses.append(positive.losses)
                    negative_losses.append(negative.losses)

                sensitivity = torch.stack(
                    [
                        (positive - negative) / (2.0 * probe_scale)
                        for positive, negative in zip(
                            positive_losses,
                            negative_losses,
                            strict=True,
                        )
                    ],
                    dim=-1,
                )
                if direction_count < self.config.candidate_directions:
                    # The moment buffers and projection are always sized at
                    # the full declared width; unprobed basis slots this
                    # bank simply contribute zero sensitivity rather than
                    # changing that width.
                    sensitivity = torch.nn.functional.pad(
                        sensitivity,
                        (
                            0,
                            self.config.candidate_directions - direction_count,
                        ),
                    )
                if bool(torch.isfinite(sensitivity).all()):
                    span = (
                        state.start_loss - state.target_loss
                    ).clamp_min(self.config.gap_epsilon)
                    active = state.active_mask
                    gradient = (
                        sensitivity / span.unsqueeze(-1)
                    )[active].mean(dim=0)
                    proposal, new_m, new_v, adam_step = (
                        self._directional_adam_proposal(
                            group,
                            gradient,
                            basis_epoch=basis_epoch,
                        )
                    )
                    coefficients = self._project_directional_coefficients(
                        proposal,
                        sensitivity,
                        state,
                    )
                else:
                    coefficients = None

                commit: _DirectionalAdamCommit | None = None
                if coefficients is not None:
                    update_direction = self._combine_directional_basis(
                        group,
                        base,
                        coefficients,
                        basis_epoch=basis_epoch,
                        tensor_magnitudes=unit_magnitudes,
                    )
                    decay_delta, decay_step = self._directional_decay_delta(
                        group,
                        base,
                    )
                    for factor in self.config.candidate_backtrack_factors:
                        retry_cost = (
                            objective_forwards
                            + int(reference_batch is not None)
                            + len(anchors)
                        )
                        if (
                            max_candidate_forwards is not None
                            and spent_forwards() + retry_cost
                            > max_candidate_forwards
                        ):
                            break
                        candidate_direction = tuple(
                            decay + update * factor
                            for decay, update in zip(
                                decay_delta,
                                update_direction,
                                strict=True,
                            )
                        )
                        scored = evaluate_direction(
                            candidate_direction,
                            direction_index=direction_count,
                            sign=1,
                            source="directional_adam",
                            check_anchors=True,
                            check_reference=True,
                            step_fraction=factor,
                        )
                        if consider(scored, candidate_direction):
                            commit = _DirectionalAdamCommit(
                                group_name=group.name,
                                first_moment=new_m.detach().clone(),
                                second_moment=new_v.detach().clone(),
                                step=adam_step,
                                last_decay_step=decay_step,
                                basis_epoch=basis_epoch,
                            )
                            break
                return _CandidateBankResult(
                    winner=winner,
                    winner_direction=winner_direction,
                    candidate_forwards=candidate_forwards,
                    anchor_forwards=anchor_forwards,
                    hard_vetoes=hard_vetoes,
                    anchor_vetoes=anchor_vetoes,
                    reference_forwards=reference_forwards,
                    reference_vetoes=reference_vetoes,
                    directional_adam_commit=commit,
                )

            momentum = self._momentum_direction(
                group,
                base,
                probe_scale=probe_scale,
            )
            for direction_index in range(probe_directions):
                signs = (1, -1) if self.config.antithetic else (1,)
                anchor_aware_probe = bool(
                    anchors
                    and self.config.antithetic_estimate_only
                    and self.config.antithetic_estimate_anchor_aware
                )
                probe_checks_anchors = (
                    not self.config.antithetic_estimate_only
                    or anchor_aware_probe
                )
                direct_anchor_cost = len(anchors) if probe_checks_anchors else 0
                direct_reference_cost = int(
                    reference_batch is not None
                    and not self.config.antithetic_estimate_only
                )
                complete_direction_cost = len(signs) * (
                    objective_forwards
                    + direct_reference_cost
                    + direct_anchor_cost
                )
                if (
                    max_candidate_forwards is not None
                    and spent_forwards() + complete_direction_cost
                    > max_candidate_forwards
                ):
                    break
                if direction_index == 0 and momentum is not None:
                    direction = momentum
                    source: CandidateSource = "momentum"
                else:
                    direction = self.sample_direction(
                        group,
                        base,
                        probe_scale=probe_scale,
                    )
                    source = "random"
                signed_scores: dict[int, CandidateResult] = {}
                for sign in signs:
                    scored = evaluate_direction(
                        direction,
                        direction_index=direction_index,
                        sign=sign,
                        source=source,
                        check_anchors=probe_checks_anchors,
                        check_reference=(
                            not self.config.antithetic_estimate_only
                        ),
                        score_anchors_without_current_gate=anchor_aware_probe,
                    )
                    signed_scores[sign] = scored
                    if (
                        not self.config.antithetic_estimate_only
                        and consider(scored, direction)
                    ):
                        break
                # candidate_directions is a maximum search allowance. Rebase
                # immediately on the first perturbation that clears every
                # acceptance and protection threshold.
                if winner is not None:
                    break
                if self.config.use_antithetic_estimate:
                    antithetic_pairs.append(
                        (direction, signed_scores[1], signed_scores[-1])
                    )

            if winner is None and antithetic_pairs:
                step_scale = (
                    probe_scale
                    if self.config.antithetic_estimate_step_scale is None
                    else self.config.antithetic_estimate_step_scale
                )
                estimates: list[tuple[tuple[Tensor, ...], int | None]] = []
                aggregate_estimate = self._antithetic_composite(
                    base,
                    antithetic_pairs,
                    state,
                    step_scale=step_scale,
                )
                if aggregate_estimate is not None:
                    estimates.append((aggregate_estimate, None))

                if self.config.antithetic_slot_estimates:
                    active = state.active_mask
                    ranked_slots: list[tuple[float, int, Tensor]] = []
                    for slot_index in range(active.shape[-1]):
                        slot_mask = torch.zeros_like(active)
                        slot_mask[..., slot_index] = active[..., slot_index]
                        if not bool(slot_mask.any()):
                            continue
                        # Hardest physical positions are evaluated first.
                        # This directly counters front-slot dominance while
                        # the final scorer still requires whole-formation
                        # improvement.
                        slot_loss = float(
                            state.current_loss[slot_mask].mean()
                        )
                        ranked_slots.append(
                            (-slot_loss, slot_index, slot_mask)
                        )
                    ranked_slots.sort(key=lambda item: (item[0], item[1]))
                    for _, slot_index, slot_mask in ranked_slots:
                        slot_estimate = self._antithetic_composite(
                            base,
                            antithetic_pairs,
                            state,
                            step_scale=step_scale,
                            active_mask=slot_mask,
                        )
                        if slot_estimate is not None:
                            estimates.append((slot_estimate, slot_index))

                # Keep each finite-difference direction, then line search
                # its commit radius with forward passes only. Every proposal,
                # including a slot-directed one, must pass the unchanged
                # active-formation threshold and replay guards.
                for estimate_index, (estimate, _) in enumerate(estimates):
                    for factor in self.config.candidate_backtrack_factors:
                        retry_cost = (
                            objective_forwards
                            + int(reference_batch is not None)
                            + len(anchors)
                        )
                        if (
                            max_candidate_forwards is not None
                            and spent_forwards() + retry_cost
                            > max_candidate_forwards
                        ):
                            break
                        candidate_direction = (
                            estimate
                            if factor == 1.0
                            else tuple(
                                value * factor for value in estimate
                            )
                        )
                        scored = evaluate_direction(
                            candidate_direction,
                            direction_index=(
                                self.config.candidate_directions
                                + estimate_index
                            ),
                            sign=1,
                            source="antithetic_estimate",
                            check_anchors=True,
                            check_reference=True,
                            step_fraction=factor,
                        )
                        if consider(scored, candidate_direction):
                            break
                    if winner is not None:
                        break
        finally:
            self._restore_group(group, base)
            self._restore_buffers(model, buffer_snapshot)
        return _CandidateBankResult(
            winner=winner,
            winner_direction=winner_direction,
            candidate_forwards=candidate_forwards,
            anchor_forwards=anchor_forwards,
            hard_vetoes=hard_vetoes,
            anchor_vetoes=anchor_vetoes,
            reference_forwards=reference_forwards,
            reference_vetoes=reference_vetoes,
        )

    def minimum_episode_forward_calls(
        self,
        *,
        anchor_count: int = 0,
        reference_enabled: bool = False,
    ) -> int:
        """Return the smallest budget that can evaluate a real proposal.

        This prevents a nearly exhausted global budget from being consumed by
        repeated objective baselines when the remaining calls cannot cover
        the optimizer's next atomic candidate bank.
        """

        if (
            isinstance(anchor_count, bool)
            or not isinstance(anchor_count, int)
            or anchor_count < 0
        ):
            raise ValueError("anchor_count must be a nonnegative integer")
        if not isinstance(reference_enabled, bool):
            raise TypeError("reference_enabled must be boolean")

        objective_forwards = self.config.directional_accumulation_batches
        reference_forwards = int(reference_enabled)
        baseline_cost = objective_forwards + reference_forwards
        verification_cost = (
            objective_forwards
            if self.config.verify_accepted_candidate
            else 0
        )
        proposal_cost = (
            objective_forwards + reference_forwards + anchor_count
        )
        if self.config.directional_adam:
            search_cost = (
                2
                * self.config.candidate_directions
                * objective_forwards
                + proposal_cost
            )
        elif self.config.antithetic_estimate_only:
            probe_anchor_cost = (
                anchor_count
                if self.config.antithetic_estimate_anchor_aware
                else 0
            )
            search_cost = (
                2
                * self.config.candidate_directions
                * (objective_forwards + probe_anchor_cost)
                + proposal_cost
            )
        else:
            signs = 2 if self.config.antithetic else 1
            search_cost = signs * proposal_cost
        return baseline_cost + verification_cost + search_cost

    def _terminal_stability(
        self,
        model: nn.Module,
        batch: Any,
        loss_provider: LossProvider,
        forward_kwargs: Mapping[str, Any],
        state: ProcoimEpisodeState,
        *,
        search_rng: _GlobalRNGState,
        buffer_snapshot: Mapping[str, Tensor],
        accumulation_batches: Sequence[Any] = (),
        max_forward_calls: int | None = None,
    ) -> int:
        """Confirm terminal SOFT slots without requiring an impossible update.

        Stability normally advances across accepted shared updates. Once no
        ACTIVE slot remains, positive active gain is undefined, so no further
        candidate can legally be accepted. Deterministic no-op forward
        confirmations close that terminal state without relaxing protection
        or changing parameters.
        """

        if max_forward_calls is not None and max_forward_calls < 0:
            raise ValueError("max_forward_calls must be nonnegative or None")

        search_batches = (batch, *tuple(accumulation_batches))
        objective_forwards = len(search_batches)
        forwards = 0
        while not bool(state.active_mask.any()) and bool(
            state.soft_frozen_mask.any()
        ):
            if (
                max_forward_calls is not None
                and forwards + objective_forwards > max_forward_calls
            ):
                break
            result = self._evaluate_objective(
                model,
                search_batches,
                loss_provider,
                forward_kwargs,
                search_rng=search_rng,
                buffer_snapshot=buffer_snapshot,
            )
            forwards += objective_forwards
            if not torch.equal(result.valid_mask, state.valid_mask):
                raise ValueError(
                    "stability valid_mask changed inside the episode"
                )
            state.current_loss[state.valid_mask] = result.losses[state.valid_mask]
            improved = state.valid_mask & result.losses.lt(state.best_loss)
            state.best_loss[improved] = result.losses[improved]
            soft = state.soft_frozen_mask
            violated = soft & result.losses.gt(state.target_loss)
            state.status[violated] = int(SlotStatus.ACTIVE)
            state.stable_count[violated] = 0
            stable = state.soft_frozen_mask
            state.stable_count[stable] += 1
            ready = stable & state.stable_count.ge(
                self.config.stability_required
            )
            state.status[ready] = int(SlotStatus.PROTECTED)
        return forwards

    def run_episode(
        self,
        model: nn.Module,
        batch: Any,
        loss_provider: LossProvider,
        *,
        forward_kwargs: Mapping[str, Any] | None = None,
        example_id: Any | None = None,
        anchors: Sequence[ProcoimAnchor] = (),
        accumulation_batches: Sequence[Any] = (),
        reference_batch: Any | None = None,
        reference_forward_kwargs: Mapping[str, Any] | None = None,
        reference_reduction_fraction: float | None = None,
        reference_batch_provider: ReferenceBatchProvider | None = None,
        reference_rotation_forwards: int | None = None,
        max_forward_calls: int | None = None,
    ) -> ProcoimEpisodeResult:
        """Improve one fixed example/microbatch using forward passes only."""

        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if not callable(loss_provider):
            raise TypeError("loss_provider must be callable")
        if (
            max_forward_calls is not None
            and (
                isinstance(max_forward_calls, bool)
                or not isinstance(max_forward_calls, int)
                or max_forward_calls <= 0
            )
        ):
            raise ValueError("max_forward_calls must be a positive integer or None")
        kwargs = {} if forward_kwargs is None else dict(forward_kwargs)
        anchors = tuple(anchors)
        if any(not isinstance(anchor, ProcoimAnchor) for anchor in anchors):
            raise TypeError("anchors must contain only ProcoimAnchor objects")
        accumulation_batches = tuple(accumulation_batches)
        search_batches = (batch, *accumulation_batches)
        baseline_forwards = len(search_batches)
        if baseline_forwards != self.config.directional_accumulation_batches:
            raise ValueError(
                "search batch count does not match "
                "directional_accumulation_batches"
            )
        if (
            reference_batch is None
            and (
                reference_forward_kwargs is not None
                or reference_reduction_fraction is not None
                or reference_batch_provider is not None
                or reference_rotation_forwards is not None
            )
        ):
            raise ValueError(
                "reference options require reference_batch"
            )
        if (
            reference_batch_provider is not None
            and not callable(reference_batch_provider)
        ):
            raise TypeError("reference_batch_provider must be callable")
        if reference_rotation_forwards is not None and (
            isinstance(reference_rotation_forwards, bool)
            or not isinstance(reference_rotation_forwards, int)
            or reference_rotation_forwards <= 0
        ):
            raise ValueError(
                "reference_rotation_forwards must be a positive integer or None"
            )
        if (reference_batch_provider is None) != (
            reference_rotation_forwards is None
        ):
            raise ValueError(
                "reference partner rotation requires both a provider and "
                "reference_rotation_forwards"
            )
        if reference_reduction_fraction is not None and (
            not math.isfinite(reference_reduction_fraction)
            or not -1 <= reference_reduction_fraction <= 1
        ):
            raise ValueError(
                "reference_reduction_fraction must be finite and in [-1, 1]"
            )
        reference_kwargs = (
            kwargs
            if reference_forward_kwargs is None
            else dict(reference_forward_kwargs)
        )
        initial_forward_cost = baseline_forwards + int(
            reference_batch is not None
        )
        if (
            max_forward_calls is not None
            and max_forward_calls < initial_forward_cost
        ):
            raise ValueError(
                "max_forward_calls is smaller than the episode baselines"
            )
        groups = self.select_parameter_groups(model)
        self._assert_no_gradients(model)
        module_modes = {
            module: module.training
            for module in model.modules()
        }
        model.eval()
        external_rng = _capture_global_rng()
        search_rng = _capture_global_rng()
        buffer_snapshot = self._snapshot_buffers(model)
        started = time.perf_counter()
        forward_calls_before = self.total_forward_calls
        candidate_forwards_before = self.total_candidate_forwards
        anchor_forwards_before = self.total_anchor_forwards
        reference_forwards_before = self.total_reference_forwards
        accepted_updates_before = self.total_accepted_updates
        anchor_vetoes_before = self.total_anchor_vetoes
        reference_vetoes_before = self.total_reference_vetoes
        reused_accepts_before = self.total_reused_direction_accepts
        estimate_accepts_before = self.total_antithetic_estimate_accepts
        directional_accepts_before = self.total_directional_adam_accepts
        accepted_updates = 0
        candidate_forwards = 0
        anchor_forwards = 0
        reference_forwards = 0
        verification_forwards = 0
        stability_forwards = 0
        hard_vetoes = 0
        anchor_vetoes = 0
        reference_vetoes = 0
        reference_rotations = 0
        reference_start_loss: float | None = None
        reference_current_loss: float | None = None
        reference_window_start_forwards = 0
        stalls = 0
        banks_since_accept = 0
        group_acceptance_counts = {group.name: 0 for group in groups}
        reused_direction_accepts = 0
        antithetic_estimate_accepts = 0
        directional_adam_accepts = 0
        current_probe_scale = self.config.probe_scale
        completion_reason = "budget"
        completed = False

        def episode_forward_calls() -> int:
            return (
                baseline_forwards
                + candidate_forwards
                + anchor_forwards
                + reference_forwards
                + verification_forwards
                + stability_forwards
            )

        def remaining_forwards() -> int | None:
            if max_forward_calls is None:
                return None
            return max(max_forward_calls - episode_forward_calls(), 0)

        try:
            baseline = self._evaluate_objective(
                model,
                search_batches,
                loss_provider,
                kwargs,
                search_rng=search_rng,
                buffer_snapshot=buffer_snapshot,
            )
            state = self.initialize_episode_state(baseline)
            episode_reduction_fraction = self._effective_reduction_fraction()
            start_aggregate = float(baseline.aggregate)
            if reference_batch is not None:
                self.total_reference_forwards += 1
                reference_forwards += 1
                reference_result = self._evaluate(
                    model,
                    reference_batch,
                    loss_provider,
                    reference_kwargs,
                    search_rng=search_rng,
                    buffer_snapshot=buffer_snapshot,
                )
                reference_start_loss = float(reference_result.aggregate)
                reference_current_loss = reference_start_loss
                reference_window_start_forwards = episode_forward_calls()

            anchors_ready = True
            anchors_needing_reference = tuple(
                anchor
                for anchor in anchors
                if not anchor.slotwise and anchor.best_aggregate is None
            )
            remaining = remaining_forwards()
            if (
                remaining is not None
                and remaining < len(anchors_needing_reference)
            ):
                anchors_ready = False
                completion_reason = "forward_budget"
            else:
                # Established anchors already carry their non-drifting
                # references. Only initialize legacy references that lack
                # one; routine replay happens after Gate 1.
                for anchor in anchors_needing_reference:
                    self.total_anchor_forwards += 1
                    anchor_forwards += 1
                    anchor_result = self._evaluate(
                        model,
                        anchor.batch,
                        loss_provider,
                        (
                            kwargs
                            if anchor.forward_kwargs is None
                            else anchor.forward_kwargs
                        ),
                        search_rng=search_rng,
                        buffer_snapshot=buffer_snapshot,
                    )
                    aggregate = float(anchor_result.aggregate)
                    if (
                        anchor.best_aggregate is None
                        or aggregate < anchor.best_aggregate
                    ):
                        anchor.best_aggregate = aggregate

            if anchors_ready:
                stability_forwards += self._terminal_stability(
                    model,
                    batch,
                    loss_provider,
                    kwargs,
                    state,
                    search_rng=search_rng,
                    buffer_snapshot=buffer_snapshot,
                    accumulation_batches=accumulation_batches,
                    max_forward_calls=remaining_forwards(),
                )
                if self._episode_complete(state, start_aggregate):
                    completed = True
                    completion_reason = self.config.completion_mode

            while (
                not completed
                and anchors_ready
                and accepted_updates < self.config.max_accepted_updates
                and banks_since_accept < self.config.max_candidate_banks
            ):
                verification_reserve = (
                    baseline_forwards
                    if self.config.verify_accepted_candidate
                    else 0
                )
                if (
                    reference_batch_provider is not None
                    and reference_rotation_forwards is not None
                    and (
                        episode_forward_calls()
                        - reference_window_start_forwards
                        >= reference_rotation_forwards
                    )
                ):
                    remaining = remaining_forwards()
                    # Partners change only at candidate-bank boundaries so a
                    # bank is always scored against one stable reference.
                    if (
                        remaining is not None
                        and remaining <= verification_reserve + 1
                    ):
                        completion_reason = "forward_budget"
                        break
                    next_reference_batch = reference_batch_provider()
                    if next_reference_batch is None:
                        raise ValueError(
                            "reference_batch_provider returned None"
                        )
                    reference_batch = next_reference_batch
                    self.total_reference_forwards += 1
                    reference_forwards += 1
                    reference_result = self._evaluate(
                        model,
                        reference_batch,
                        loss_provider,
                        reference_kwargs,
                        search_rng=search_rng,
                        buffer_snapshot=buffer_snapshot,
                    )
                    reference_start_loss = float(
                        reference_result.aggregate
                    )
                    reference_current_loss = reference_start_loss
                    reference_rotations += 1
                    reference_window_start_forwards = (
                        episode_forward_calls()
                    )

                remaining = remaining_forwards()
                if (
                    remaining is not None
                    and remaining <= verification_reserve
                ):
                    completion_reason = "forward_budget"
                    break
                search_budget = (
                    None if remaining is None else remaining - verification_reserve
                )
                group = self._select_group(groups)
                if self.config.adaptive_probe_scale:
                    current_probe_scale = self._group_probe_scales[group.name]
                else:
                    current_probe_scale = self.config.probe_scale
                if self.config.adaptive_candidate_directions:
                    current_probe_directions = (
                        self._group_candidate_directions[group.name]
                    )
                else:
                    current_probe_directions = self.config.candidate_directions
                candidate_count_before = self.total_candidate_forwards
                try:
                    bank = self.evaluate_candidate_bank(
                        model,
                        batch,
                        loss_provider,
                        kwargs,
                        state,
                        group,
                        probe_scale=current_probe_scale,
                        search_rng=search_rng,
                        buffer_snapshot=buffer_snapshot,
                        anchors=anchors,
                        accumulation_batches=accumulation_batches,
                        reference_batch=reference_batch,
                        reference_forward_kwargs=reference_kwargs,
                        reference_current_loss=reference_current_loss,
                        reference_reduction_fraction=(
                            self.config.reference_reduction_fraction
                            if reference_reduction_fraction is None
                            else reference_reduction_fraction
                        ),
                        max_candidate_forwards=search_budget,
                        probe_directions=current_probe_directions,
                    )
                finally:
                    if self.total_candidate_forwards > candidate_count_before:
                        self._next_group_index += 1
                candidate_forwards += bank.candidate_forwards
                anchor_forwards += bank.anchor_forwards
                reference_forwards += bank.reference_forwards
                hard_vetoes += bank.hard_vetoes
                anchor_vetoes += bank.anchor_vetoes
                reference_vetoes += bank.reference_vetoes
                if bank.search_forwards == 0:
                    completion_reason = "forward_budget"
                    break
                if bank.winner is None:
                    self._record_group_bank(group, bank)
                    stalls += 1
                    banks_since_accept += 1
                    continue

                winner = bank.winner
                winner_direction = bank.winner_direction
                assert winner_direction is not None
                accepted_base = self._snapshot_group(group)
                state_before = state.clone()
                try:
                    self._apply_direction(
                        group,
                        winner_direction,
                        winner.sign,
                    )
                    accepted_result = SlotLossResult(
                        losses=winner.losses,
                        valid_mask=state.valid_mask,
                        aggregate=winner.losses[state.valid_mask].mean(),
                        guard_aggregate=(
                            None
                            if winner.guard_aggregate is None
                            else winner.losses.new_tensor(
                                winner.guard_aggregate
                            )
                        ),
                    )
                    if self.config.verify_accepted_candidate:
                        accepted_result = self._evaluate_objective(
                            model,
                            search_batches,
                            loss_provider,
                            kwargs,
                            search_rng=search_rng,
                            buffer_snapshot=buffer_snapshot,
                        )
                        verification_forwards += baseline_forwards
                        verified_score = self.score_candidate(
                            state,
                            accepted_result,
                            group_name=winner.group_name,
                            direction_index=winner.direction_index,
                            sign=winner.sign,
                        )
                        if not verified_score.eligible:
                            self._restore_group(group, accepted_base)
                            state = state_before
                            failed_bank = replace(
                                bank,
                                winner=None,
                                winner_direction=None,
                            )
                            self._record_group_bank(group, failed_bank)
                            stalls += 1
                            banks_since_accept += 1
                            continue
                    if reference_batch is not None:
                        if winner.reference_aggregate is None:
                            raise RuntimeError(
                                "accepted reference-guarded candidate "
                                "has no reference loss"
                            )
                        reference_current_loss = winner.reference_aggregate
                    self.update_slot_states(
                        state,
                        accepted_result.losses,
                        (
                            None
                            if accepted_result.guard_aggregate is None
                            else float(accepted_result.guard_aggregate)
                        ),
                    )
                    stability_forwards += self._terminal_stability(
                        model,
                        batch,
                        loss_provider,
                        kwargs,
                        state,
                        search_rng=search_rng,
                        buffer_snapshot=buffer_snapshot,
                        accumulation_batches=accumulation_batches,
                        max_forward_calls=remaining_forwards(),
                    )
                except Exception:
                    self._restore_group(group, accepted_base)
                    state = state_before
                    raise
                if bank.directional_adam_commit is not None:
                    self._commit_directional_adam(
                        bank.directional_adam_commit
                    )
                self._accumulate_consolidation_window(
                    group,
                    winner_direction,
                    winner.sign,
                )
                self._record_group_bank(group, bank)
                self._update_direction_momentum(
                    group,
                    winner_direction,
                    winner.sign,
                )
                if winner.anchor_losses:
                    for anchor, losses, aggregate in zip(
                        anchors,
                        winner.anchor_losses,
                        winner.anchor_aggregates,
                        strict=True,
                    ):
                        if anchor.slotwise:
                            assert anchor.best_losses is not None
                            assert anchor.valid_mask is not None
                            assert anchor.protected_mask is not None
                            improved = (
                                anchor.valid_mask
                                & anchor.protected_mask
                                & losses.lt(anchor.best_losses)
                            )
                            anchor.best_losses[improved] = losses[improved]
                            if bool(anchor.valid_mask.any()):
                                anchor.best_aggregate = float(
                                    anchor.best_losses[
                                        anchor.valid_mask
                                    ].mean()
                                )
                        elif (
                            anchor.best_aggregate is None
                            or aggregate < anchor.best_aggregate
                        ):
                            anchor.best_aggregate = aggregate
                accepted_updates += 1
                self.total_accepted_updates += 1
                banks_since_accept = 0
                group_acceptance_counts[group.name] += 1
                if winner.source == "momentum":
                    reused_direction_accepts += 1
                    self.total_reused_direction_accepts += 1
                elif winner.source == "antithetic_estimate":
                    antithetic_estimate_accepts += 1
                    self.total_antithetic_estimate_accepts += 1
                elif winner.source == "directional_adam":
                    directional_adam_accepts += 1
                    self.total_directional_adam_accepts += 1
                self.parameter_group_acceptance_counts[group.name] = (
                    self.parameter_group_acceptance_counts.get(group.name, 0)
                    + 1
                )
                if self._episode_complete(state, start_aggregate):
                    completed = True
                    completion_reason = self.config.completion_mode
                    break

            if not completed:
                if self.config.completion_mode == "budget":
                    completed = True
                    completion_reason = "budget"
                elif (
                    max_forward_calls is not None
                    and remaining_forwards() == 0
                ):
                    completion_reason = "forward_budget"
                elif accepted_updates >= self.config.max_accepted_updates:
                    completion_reason = "accepted_update_budget"
                elif banks_since_accept >= self.config.max_candidate_banks:
                    completion_reason = "candidate_bank_budget"
            end_aggregate = float(state.current_loss[state.valid_mask].mean())
        finally:
            self._restore_buffers(model, buffer_snapshot)
            _restore_global_rng(external_rng)
            for module, training in module_modes.items():
                module.training = training
            self._assert_no_gradients(model)

        wall_clock = time.perf_counter() - started
        result = ProcoimEpisodeResult(
            state=state,
            example_id=example_id,
            start_aggregate_loss=start_aggregate,
            end_aggregate_loss=end_aggregate,
            accepted_updates=accepted_updates,
            candidate_forwards=candidate_forwards,
            anchor_forwards=anchor_forwards,
            verification_forwards=verification_forwards,
            stability_forwards=stability_forwards,
            hard_vetoes=hard_vetoes,
            anchor_vetoes=anchor_vetoes,
            stall_count=stalls,
            current_probe_scale=current_probe_scale,
            parameter_group_acceptance_counts=group_acceptance_counts,
            parameter_group_probe_scales=dict(self._group_probe_scales),
            parameter_group_candidate_directions=dict(
                self._group_candidate_directions
            ),
            effective_reduction_fraction=episode_reduction_fraction,
            parameter_group_gain_per_forward=dict(self._group_gain_ema),
            parameter_group_basis_epochs=(
                {
                    group.name: self._directional_basis_epoch(group)
                    for group in groups
                }
                if self.config.directional_adam else {}
            ),
            reused_direction_accepts=reused_direction_accepts,
            antithetic_estimate_accepts=antithetic_estimate_accepts,
            directional_adam_accepts=directional_adam_accepts,
            total_reused_direction_accepts=(
                self.total_reused_direction_accepts
            ),
            total_antithetic_estimate_accepts=(
                self.total_antithetic_estimate_accepts
            ),
            total_directional_adam_accepts=(
                self.total_directional_adam_accepts
            ),
            total_anchor_vetoes=self.total_anchor_vetoes,
            completed=completed,
            completion_reason=completion_reason,
            wall_clock_seconds=wall_clock,
            baseline_forwards=baseline_forwards,
            reference_forwards=reference_forwards,
            reference_vetoes=reference_vetoes,
            reference_rotations=reference_rotations,
            reference_start_loss=reference_start_loss,
            reference_end_loss=reference_current_loss,
        )
        if (
            self.total_forward_calls - forward_calls_before
            != result.total_forward_calls
        ):
            raise RuntimeError("internal Procoim forward accounting mismatch")
        if (
            self.total_candidate_forwards - candidate_forwards_before
            != candidate_forwards
        ):
            raise RuntimeError("internal Procoim candidate accounting mismatch")
        if (
            self.total_anchor_forwards - anchor_forwards_before
            != anchor_forwards
        ):
            raise RuntimeError("internal Procoim anchor accounting mismatch")
        if (
            self.total_reference_forwards - reference_forwards_before
            != reference_forwards
        ):
            raise RuntimeError("internal Procoim reference accounting mismatch")
        if (
            self.total_accepted_updates - accepted_updates_before
            != accepted_updates
        ):
            raise RuntimeError("internal Procoim accepted-update accounting mismatch")
        if self.total_anchor_vetoes - anchor_vetoes_before != anchor_vetoes:
            raise RuntimeError("internal Procoim anchor-veto accounting mismatch")
        if (
            self.total_reference_vetoes - reference_vetoes_before
            != reference_vetoes
        ):
            raise RuntimeError("internal Procoim reference-veto accounting mismatch")
        if (
            self.total_reused_direction_accepts - reused_accepts_before
            != reused_direction_accepts
        ):
            raise RuntimeError("internal Procoim reuse accounting mismatch")
        if (
            self.total_antithetic_estimate_accepts - estimate_accepts_before
            != antithetic_estimate_accepts
        ):
            raise RuntimeError("internal Procoim estimate accounting mismatch")
        if (
            self.total_directional_adam_accepts - directional_accepts_before
            != directional_adam_accepts
        ):
            raise RuntimeError("internal directional Adam accounting mismatch")
        self.episodes += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return persistent search state; accepted weights live in the model."""

        generator_states = {
            key: generator.get_state()
            for key, generator in self._generators.items()
        }
        generator_states.update(self._pending_generator_states)
        return {
            "format": "gator-procoim",
            "version": self.format_version,
            "config": asdict(self.config),
            "episodes": self.episodes,
            "total_forward_calls": self.total_forward_calls,
            "total_candidate_forwards": self.total_candidate_forwards,
            "total_anchor_forwards": self.total_anchor_forwards,
            "total_reference_forwards": self.total_reference_forwards,
            "total_accepted_updates": self.total_accepted_updates,
            "total_reused_direction_accepts": (
                self.total_reused_direction_accepts
            ),
            "total_antithetic_estimate_accepts": (
                self.total_antithetic_estimate_accepts
            ),
            "total_directional_adam_accepts": self.total_directional_adam_accepts,
            "total_anchor_vetoes": self.total_anchor_vetoes,
            "total_reference_vetoes": self.total_reference_vetoes,
            "total_consolidation_freezes": self.total_consolidation_freezes,
            "parameter_group_acceptance_counts": dict(
                self.parameter_group_acceptance_counts
            ),
            "next_parameter_group_index": self._next_group_index,
            "group_probe_scales": dict(self._group_probe_scales),
            "group_candidate_directions": dict(
                self._group_candidate_directions
            ),
            "group_basis_forward_counts": dict(
                self._group_basis_forward_counts
            ),
            "group_basis_stall_streak": dict(self._group_basis_stall_streak),
            "group_basis_stall_epochs": dict(self._group_basis_stall_epochs),
            "group_gain_ema": dict(self._group_gain_ema),
            "group_bank_counts": dict(self._group_bank_counts),
            "direction_momentum": {
                name: tuple(value.detach().clone() for value in values)
                for name, values in self._direction_momentum.items()
            },
            "group_window_accum": {
                name: tuple(value.detach().clone() for value in values)
                for name, values in self._group_window_accum.items()
            },
            "group_prev_window_vector": {
                name: tuple(value.detach().clone() for value in values)
                for name, values in self._group_prev_window_vector.items()
            },
            "group_window_cosine_baseline": dict(
                self._group_window_cosine_baseline
            ),
            "group_epoch_freeze": dict(self._group_epoch_freeze),
            "group_last_natural_epoch": dict(
                self._group_last_natural_epoch
            ),
            "directional_adam_m": {
                name: value.detach().cpu().clone()
                for name, value in self._directional_adam_m.items()
            },
            "directional_adam_v": {
                name: value.detach().cpu().clone()
                for name, value in self._directional_adam_v.items()
            },
            "directional_adam_steps": dict(self._directional_adam_steps),
            "directional_adam_last_decay_step": dict(
                self._directional_adam_last_decay_step
            ),
            "directional_adam_basis_epochs": dict(
                self._directional_adam_basis_epochs
            ),
            "generator_states": generator_states,
            "group_signature": (
                self._group_signature
                if self._group_signature is not None
                else self._expected_group_signature
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore counters and dedicated candidate RNG state."""

        if state.get("format") != "gator-procoim":
            raise ValueError("not a Procoim state dictionary")
        version = int(state.get("version", -1))
        if version not in {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            self.format_version,
        }:
            raise ValueError("unsupported Procoim state version")
        saved_config = state.get("config")
        if saved_config is not None:
            current_config = asdict(self.config)
            normalized_saved = dict(saved_config)
            if version in {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            }:
                defaults = asdict(ProcoimConfig())
                legacy_directional_defaults = {
                    "directional_basis_mode": "fixed_rademacher",
                    "directional_factor_rank": 1,
                    "directional_basis_rotation_banks": 8,
                }
                for name, default in defaults.items():
                    if name not in normalized_saved:
                        default = legacy_directional_defaults.get(
                            name,
                            default,
                        )
                        if current_config[name] != default:
                            raise ValueError(
                                "legacy Procoim checkpoint cannot enable new "
                                f"configuration field {name!r}"
                            )
                        normalized_saved[name] = default
            if normalized_saved != current_config:
                raise ValueError("Procoim checkpoint configuration does not match")
        self.episodes = int(state.get("episodes", 0))
        self.total_forward_calls = int(state.get("total_forward_calls", 0))
        self.total_candidate_forwards = int(
            state.get("total_candidate_forwards", 0)
        )
        self.total_anchor_forwards = int(
            state.get("total_anchor_forwards", 0)
        )
        self.total_reference_forwards = int(
            state.get("total_reference_forwards", 0)
        )
        self.total_accepted_updates = int(
            state.get("total_accepted_updates", 0)
        )
        self.total_reused_direction_accepts = int(
            state.get("total_reused_direction_accepts", 0)
        )
        self.total_antithetic_estimate_accepts = int(
            state.get("total_antithetic_estimate_accepts", 0)
        )
        self.total_directional_adam_accepts = int(
            state.get("total_directional_adam_accepts", 0)
        )
        self.total_anchor_vetoes = int(
            state.get("total_anchor_vetoes", 0)
        )
        self.total_reference_vetoes = int(
            state.get("total_reference_vetoes", 0)
        )
        self.total_consolidation_freezes = int(
            state.get("total_consolidation_freezes", 0)
        )
        if any(
            value < 0
            for value in (
                self.total_reused_direction_accepts,
                self.total_antithetic_estimate_accepts,
                self.total_directional_adam_accepts,
                self.total_anchor_vetoes,
                self.total_reference_forwards,
                self.total_reference_vetoes,
                self.total_consolidation_freezes,
            )
        ):
            raise ValueError("checkpointed Procoim counters must be nonnegative")
        self.parameter_group_acceptance_counts = {
            str(name): int(count)
            for name, count in state.get(
                "parameter_group_acceptance_counts", {}
            ).items()
        }
        self._next_group_index = int(
            state.get("next_parameter_group_index", 0)
        )
        if self._next_group_index < 0:
            raise ValueError("next_parameter_group_index cannot be negative")
        self._group_probe_scales = {
            str(name): float(value)
            for name, value in state.get("group_probe_scales", {}).items()
        }
        if any(
            not math.isfinite(value) or value <= 0
            for value in self._group_probe_scales.values()
        ):
            raise ValueError("checkpointed group probe scales must be positive")
        self._group_candidate_directions = {
            str(name): int(value)
            for name, value in state.get(
                "group_candidate_directions", {}
            ).items()
        }
        if any(
            value <= 0 for value in self._group_candidate_directions.values()
        ):
            raise ValueError(
                "checkpointed group candidate directions must be positive"
            )
        self._group_basis_forward_counts = {
            str(name): int(value)
            for name, value in state.get(
                "group_basis_forward_counts", {}
            ).items()
        }
        if any(
            value < 0 for value in self._group_basis_forward_counts.values()
        ):
            raise ValueError(
                "checkpointed group basis forward counts must be nonnegative"
            )
        self._group_basis_stall_streak = {
            str(name): int(value)
            for name, value in state.get(
                "group_basis_stall_streak", {}
            ).items()
        }
        if any(
            value < 0 for value in self._group_basis_stall_streak.values()
        ):
            raise ValueError(
                "checkpointed group basis stall streaks must be nonnegative"
            )
        self._group_basis_stall_epochs = {
            str(name): int(value)
            for name, value in state.get(
                "group_basis_stall_epochs", {}
            ).items()
        }
        if any(
            value < 0 for value in self._group_basis_stall_epochs.values()
        ):
            raise ValueError(
                "checkpointed group basis stall epochs must be nonnegative"
            )
        self._group_gain_ema = {
            str(name): float(value)
            for name, value in state.get("group_gain_ema", {}).items()
        }
        if any(
            not math.isfinite(value) or value < 0
            for value in self._group_gain_ema.values()
        ):
            raise ValueError("checkpointed group gains must be nonnegative")
        self._group_bank_counts = {
            str(name): int(value)
            for name, value in state.get("group_bank_counts", {}).items()
        }
        if any(value < 0 for value in self._group_bank_counts.values()):
            raise ValueError("checkpointed group bank counts must be nonnegative")
        self._direction_momentum = {
            str(name): tuple(value.detach().clone() for value in values)
            for name, values in state.get("direction_momentum", {}).items()
        }
        self._group_window_accum = {
            str(name): tuple(value.detach().clone() for value in values)
            for name, values in state.get("group_window_accum", {}).items()
        }
        self._group_prev_window_vector = {
            str(name): tuple(value.detach().clone() for value in values)
            for name, values in state.get(
                "group_prev_window_vector", {}
            ).items()
        }
        self._group_window_cosine_baseline = {
            str(name): float(value)
            for name, value in state.get(
                "group_window_cosine_baseline", {}
            ).items()
        }
        if any(
            not math.isfinite(value)
            for value in self._group_window_cosine_baseline.values()
        ):
            raise ValueError(
                "checkpointed group window cosine baselines must be finite"
            )
        self._group_epoch_freeze = {
            str(name): int(value)
            for name, value in state.get("group_epoch_freeze", {}).items()
        }
        if any(value < 0 for value in self._group_epoch_freeze.values()):
            raise ValueError(
                "checkpointed group epoch freeze counts must be nonnegative"
            )
        self._group_last_natural_epoch = {
            str(name): int(value)
            for name, value in state.get(
                "group_last_natural_epoch", {}
            ).items()
        }
        if any(
            value < 0 for value in self._group_last_natural_epoch.values()
        ):
            raise ValueError(
                "checkpointed group last natural epochs must be nonnegative"
            )
        self._directional_adam_m = {
            str(name): value.detach().cpu().to(torch.float64).clone()
            for name, value in state.get("directional_adam_m", {}).items()
        }
        self._directional_adam_v = {
            str(name): value.detach().cpu().to(torch.float64).clone()
            for name, value in state.get("directional_adam_v", {}).items()
        }
        self._directional_adam_steps = {
            str(name): int(value)
            for name, value in state.get("directional_adam_steps", {}).items()
        }
        self._directional_adam_last_decay_step = {
            str(name): int(value)
            for name, value in state.get(
                "directional_adam_last_decay_step", {}
            ).items()
        }
        self._directional_adam_basis_epochs = {
            str(name): int(value)
            for name, value in state.get(
                "directional_adam_basis_epochs",
                (
                    {name: 0 for name in self._directional_adam_m}
                    if version <= 13
                    else {}
                ),
            ).items()
        }
        self._generators.clear()
        self._pending_generator_states = {
            str(key): value.detach().clone()
            for key, value in state.get("generator_states", {}).items()
        }
        self._expected_group_signature = state.get("group_signature")
        self._group_signature = self._expected_group_signature


def add_procoim_arguments(parser: argparse.ArgumentParser) -> None:
    """Add reusable Procoim controls to a training entrypoint."""

    group = parser.add_argument_group("Procoim forward-only optimization")
    group.add_argument(
        "--procoim-reduction-fraction",
        type=float,
        default=0.005,
        help="per-slot fractional CE reduction requested in each episode",
    )
    group.add_argument(
        "--procoim-adaptive-reduction-fraction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "decay the required CE reduction logarithmically with episode "
            "count instead of holding it fixed; early episodes demand the "
            "full --procoim-reduction-fraction, later ones accept less"
        ),
    )
    group.add_argument(
        "--procoim-reduction-fraction-decay-rate",
        type=float,
        default=0.1,
        help=(
            "controls how fast the log decay flattens the CE bar; larger "
            "values decay faster"
        ),
    )
    group.add_argument(
        "--procoim-minimum-reduction-fraction",
        type=float,
        default=1e-6,
        help="floor the decayed CE bar cannot fall below",
    )
    group.add_argument(
        "--procoim-slot-objective-reduction",
        choices=("per_example", "batch_mean"),
        default="batch_mean",
        help=(
            "batch_mean averages examples into one objective per physical "
            "output slot; per_example preserves one objective for every "
            "(example, slot) pair"
        ),
    )
    group.add_argument("--procoim-loss-floor", type=float, default=0.01)
    group.add_argument("--procoim-candidate-directions", type=int, default=32)
    group.add_argument(
        "--procoim-adaptive-candidate-directions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "start each bank's search at "
            "--procoim-minimum-candidate-directions and double it on "
            "failure, up to --procoim-candidate-directions; a bank that "
            "succeeds resets the next one back to the minimum"
        ),
    )
    group.add_argument(
        "--procoim-minimum-candidate-directions",
        type=int,
        default=16,
        help="starting probe width when --procoim-adaptive-candidate-directions is set",
    )
    group.add_argument(
        "--procoim-candidate-directions-growth",
        type=float,
        default=2.0,
        help="multiplier applied to the probe width after a failed bank",
    )
    group.add_argument(
        "--procoim-antithetic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument(
        "--procoim-max-candidate-banks",
        type=int,
        default=16,
        help=(
            "stop after this many consecutive candidate banks fail; an "
            "accepted update resets the stall budget"
        ),
    )
    group.add_argument(
        "--procoim-max-accepted-updates",
        type=int,
        default=64,
        help="maximum accepted perturbations in one episode",
    )
    group.add_argument(
        "--procoim-verify-accepted-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "rerun every permanently applied winner and use the verified "
            "losses for slot state; the verification forward counts as a step"
        ),
    )
    group.add_argument("--procoim-stability-required", type=int, default=3)
    group.add_argument(
        "--procoim-soft-regression-fraction", type=float, default=0.02
    )
    group.add_argument(
        "--procoim-hard-regression-fraction", type=float, default=0.08
    )
    group.add_argument(
        "--procoim-absolute-regression-tolerance",
        type=float,
        default=1e-4,
    )
    group.add_argument("--procoim-protection-weight", type=float, default=2.0)
    group.add_argument(
        "--procoim-global-regression-weight", type=float, default=0.0
    )
    group.add_argument("--procoim-minimum-active-gain", type=float, default=1e-4)
    group.add_argument("--procoim-probe-scale", type=float, default=0.03)
    group.add_argument("--procoim-probe-scale-decay", type=float, default=0.5)
    group.add_argument(
        "--procoim-minimum-probe-scale", type=float, default=1e-6
    )
    group.add_argument(
        "--procoim-parameter-rms-floor", type=float, default=1e-3
    )
    group.add_argument(
        "--procoim-reuse-successful-directions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reuse an EMA of each group's signed accepted directions",
    )
    group.add_argument(
        "--procoim-direction-momentum",
        type=float,
        default=0.9,
    )
    group.add_argument(
        "--procoim-antithetic-estimate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "synthesize and explicitly score one finite-difference composite "
            "from every complete antithetic bank"
        ),
    )
    group.add_argument(
        "--procoim-antithetic-estimate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use individual +/- probes only to construct the finite-"
            "difference composite; only the explicitly scored composite "
            "may be committed"
        ),
    )
    group.add_argument(
        "--procoim-antithetic-estimate-anchor-aware",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "include sentinel-anchor regression in every +/- finite-"
            "difference measurement so the synthesized direction searches "
            "for shared improvement before the final replay gate"
        ),
    )
    group.add_argument(
        "--procoim-antithetic-slot-estimates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "also synthesize one proposal per active physical output slot "
            "from each antithetic bank; proposals remain subject to the "
            "whole active-formation gain threshold and all protection gates"
        ),
    )
    group.add_argument(
        "--procoim-antithetic-estimate-step-scale",
        type=float,
        default=None,
        help=(
            "parameter-RMS-relative radius used to commit the finite-"
            "difference composite; defaults to --procoim-probe-scale so "
            "measurement and update radii can be tuned independently"
        ),
    )
    group.add_argument(
        "--procoim-directional-adam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "estimate gradients in a seeded perturbation basis, run Adam "
            "in coefficient space, project around protected slots, "
            "and commit moments only after forward validation succeeds"
        ),
    )
    group.add_argument(
        "--procoim-directional-adam-learning-rate",
        type=float,
        default=0.03,
        help="Adam coefficient-space learning rate",
    )
    group.add_argument(
        "--procoim-directional-adam-beta1",
        type=float,
        default=0.9,
    )
    group.add_argument(
        "--procoim-directional-adam-beta2",
        type=float,
        default=0.99,
    )
    group.add_argument(
        "--procoim-directional-adam-epsilon",
        type=float,
        default=1e-8,
    )
    group.add_argument(
        "--procoim-directional-adam-weight-decay",
        type=float,
        default=0.0,
        help=(
            "lazy decoupled decay applied to a touched block only when its "
            "projected update is accepted"
        ),
    )
    group.add_argument(
        "--procoim-directional-adam-projection-iterations",
        type=int,
        default=64,
        help=(
            "Dykstra half-space projection passes in directional "
            "coefficient space"
        ),
    )
    group.add_argument(
        "--procoim-directional-basis-mode",
        choices=("fixed_rademacher", "rotating_factorized"),
        default="rotating_factorized",
        help=(
            "fixed_rademacher preserves the original dense K-dimensional "
            "subspace; rotating_factorized periodically replaces it with "
            "seed-regenerated low-rank tensor directions"
        ),
    )
    group.add_argument(
        "--procoim-directional-factor-rank",
        type=int,
        default=1,
        help=(
            "matrix rank of each tensor-local rotating proposal factor; "
            "vectors continue to use seeded Rademacher directions"
        ),
    )
    group.add_argument(
        "--procoim-directional-basis-rotation-banks",
        type=int,
        default=8,
        help=(
            "group-local candidate banks before rotating to a fresh seeded "
            "factor basis; coefficient moments reset at each rotation. "
            "Ignored when --procoim-directional-basis-rotation-forwards or "
            "--procoim-directional-basis-rotation-stall-banks is set"
        ),
    )
    group.add_argument(
        "--procoim-directional-basis-rotation-forwards",
        type=int,
        default=None,
        help=(
            "rotate the directional basis every this many group-local "
            "forward calls instead of every N candidate banks; overrides "
            "--procoim-directional-basis-rotation-banks when set. Cannot be "
            "combined with --procoim-directional-basis-rotation-stall-banks"
        ),
    )
    group.add_argument(
        "--procoim-directional-basis-rotation-stall-banks",
        type=int,
        default=None,
        help=(
            "rotate a group's directional basis after this many CONSECUTIVE "
            "failed candidate banks, resetting the streak on rotation or on "
            "the next accepted candidate; a success never rotates. Overrides "
            "--procoim-directional-basis-rotation-banks when set, and cannot "
            "be combined with --procoim-directional-basis-rotation-forwards"
        ),
    )
    group.add_argument(
        "--procoim-directional-basis-consolidation-cosine-margin",
        type=float,
        default=None,
        help=(
            "when set, defer a group's next scheduled basis rotation "
            "(from whichever of --procoim-directional-basis-rotation-banks"
            "/-forwards/-stall-banks is active) by one window whenever "
            "that window's accepted-direction vector is more aligned with "
            "the previous window's than the group's running cosine "
            "baseline plus this margin; motivated by AdamW breakout "
            "traces where a directional-consistency spike marked exactly "
            "when a stalled output position locked in. Requires "
            "--procoim-directional-basis-mode=rotating_factorized. "
            "Disabled (None) by default"
        ),
    )
    group.add_argument(
        "--procoim-directional-basis-consolidation-ema-beta",
        type=float,
        default=0.9,
        help=(
            "smoothing factor for the per-group running cosine baseline "
            "used by --procoim-directional-basis-consolidation-cosine-"
            "margin; higher values react more slowly to new windows"
        ),
    )
    group.add_argument(
        "--procoim-directional-accumulation-batches",
        type=int,
        default=1,
        help=(
            "average each baseline, antithetic probe, proposal, and "
            "verification over this many independently generated "
            "microbatches; every underlying model forward remains a step"
        ),
    )
    group.add_argument(
        "--procoim-reference-reduction-fraction",
        type=float,
        default=None,
        help=(
            "fractional CE improvement required on a separate reference "
            "batch before a locally eligible proposal may commit; defaults "
            "to --procoim-reduction-fraction"
        ),
    )
    group.add_argument(
        "--procoim-candidate-backtrack-factors",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.125],
        help=(
            "descending commit-radius multipliers tried along a synthesized "
            "direction; every retry is a counted forward"
        ),
    )
    group.add_argument(
        "--procoim-adaptive-probe-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    group.add_argument(
        "--procoim-probe-scale-growth", type=float, default=1.1
    )
    group.add_argument(
        "--procoim-maximum-probe-scale", type=float, default=0.1
    )
    group.add_argument(
        "--procoim-adaptive-group-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allocate banks with deterministic gain-per-forward UCB scores",
    )
    group.add_argument("--procoim-group-score-beta", type=float, default=0.9)
    group.add_argument(
        "--procoim-group-exploration", type=float, default=1e-3
    )
    group.add_argument(
        "--procoim-anchor-buffer-size",
        type=int,
        default=64,
        help="bounded bank of prior examples retained as slotwise anchors",
    )
    group.add_argument(
        "--procoim-anchor-sentinel-count",
        type=int,
        default=4,
        help=(
            "rotating replay anchors checked only after a candidate passes "
            "the current-example gate"
        ),
    )
    group.add_argument(
        "--procoim-anchor-regression-fraction",
        type=float,
        default=0.02,
        help="legacy aggregate-anchor tolerance; new anchors are slotwise",
    )
    group.add_argument(
        "--procoim-anchor-soft-regression-fraction",
        type=float,
        default=0.03,
    )
    group.add_argument(
        "--procoim-anchor-hard-regression-fraction",
        type=float,
        default=0.10,
    )
    group.add_argument(
        "--procoim-anchor-minimum-retained-reduction-fraction",
        type=float,
        default=None,
        help=(
            "optional consolidation floor for net retained reduction across "
            "protected anchor slots; individual slots may redistribute "
            "within their catastrophe bounds"
        ),
    )
    group.add_argument(
        "--procoim-anchor-absolute-tolerance", type=float, default=1e-4
    )
    group.add_argument("--procoim-anchor-weight", type=float, default=1.0)
    group.add_argument(
        "--procoim-anchor-hard-protection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument(
        "--procoim-consolidation-interval-examples",
        type=int,
        default=16,
        help=(
            "examples accumulated in provisional delta_work before a "
            "retention audit; zero disables provisional windows"
        ),
    )
    group.add_argument(
        "--procoim-full-audit-anchor-count",
        type=int,
        default=32,
        help="maximum consolidated anchors included in a window audit",
    )
    group.add_argument(
        "--procoim-consolidation-alphas",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25],
        help="descending theta_base + alpha*delta_work commit fractions",
    )
    group.add_argument(
        "--procoim-provisional-shrink-factor",
        type=float,
        default=0.5,
        help=(
            "additional partial-recovery alpha included in consolidation "
            "when absent from --procoim-consolidation-alphas"
        ),
    )
    group.add_argument(
        "--procoim-parameter-group-granularity",
        choices=("semantic", "relay", "tensor"),
        default="semantic",
        help=(
            "semantic perturbs each broad model-defined block; relay "
            "splits core relay parameters by relay while preserving shared "
            "blocks; tensor "
            "splits selected semantic blocks into one search group per "
            "parameter tensor to reduce finite-difference dimensionality"
        ),
    )
    group.add_argument(
        "--procoim-parameter-groups",
        nargs="+",
        default=[
            "routing",
            "receiver_memory",
            "stream_mix",
            "node_compute",
            "readout",
        ],
    )
    group.add_argument(
        "--procoim-completion-mode",
        choices=["all_slots", "aggregate_and_fraction", "budget"],
        default="all_slots",
    )
    group.add_argument("--procoim-protected-fraction", type=float, default=1.0)
    group.add_argument("--procoim-seed", type=int, default=None)


def procoim_config_from_args(args: argparse.Namespace) -> ProcoimConfig:
    """Build a validated Procoim configuration from parsed CLI arguments."""

    return ProcoimConfig(
        reduction_fraction=args.procoim_reduction_fraction,
        adaptive_reduction_fraction=(
            args.procoim_adaptive_reduction_fraction
        ),
        reduction_fraction_decay_rate=(
            args.procoim_reduction_fraction_decay_rate
        ),
        minimum_reduction_fraction=(
            args.procoim_minimum_reduction_fraction
        ),
        loss_floor=args.procoim_loss_floor,
        slot_objective_reduction=(
            args.procoim_slot_objective_reduction
        ),
        candidate_directions=args.procoim_candidate_directions,
        adaptive_candidate_directions=(
            args.procoim_adaptive_candidate_directions
        ),
        minimum_candidate_directions=(
            args.procoim_minimum_candidate_directions
        ),
        candidate_directions_growth=(
            args.procoim_candidate_directions_growth
        ),
        antithetic=args.procoim_antithetic,
        max_candidate_banks=args.procoim_max_candidate_banks,
        max_accepted_updates=args.procoim_max_accepted_updates,
        verify_accepted_candidate=(
            args.procoim_verify_accepted_candidate
        ),
        stability_required=args.procoim_stability_required,
        soft_regression_fraction=args.procoim_soft_regression_fraction,
        hard_regression_fraction=args.procoim_hard_regression_fraction,
        absolute_regression_tolerance=(
            args.procoim_absolute_regression_tolerance
        ),
        protection_weight=args.procoim_protection_weight,
        global_regression_weight=args.procoim_global_regression_weight,
        minimum_active_gain=args.procoim_minimum_active_gain,
        probe_scale=args.procoim_probe_scale,
        probe_scale_decay=args.procoim_probe_scale_decay,
        minimum_probe_scale=args.procoim_minimum_probe_scale,
        parameter_rms_floor=args.procoim_parameter_rms_floor,
        reuse_successful_directions=(
            args.procoim_reuse_successful_directions
        ),
        direction_momentum=args.procoim_direction_momentum,
        use_antithetic_estimate=args.procoim_antithetic_estimate,
        antithetic_estimate_only=(
            args.procoim_antithetic_estimate_only
        ),
        antithetic_estimate_anchor_aware=(
            args.procoim_antithetic_estimate_anchor_aware
        ),
        antithetic_slot_estimates=(
            args.procoim_antithetic_slot_estimates
        ),
        antithetic_estimate_step_scale=(
            args.procoim_antithetic_estimate_step_scale
        ),
        directional_adam=args.procoim_directional_adam,
        directional_adam_learning_rate=(
            args.procoim_directional_adam_learning_rate
        ),
        directional_adam_beta1=args.procoim_directional_adam_beta1,
        directional_adam_beta2=args.procoim_directional_adam_beta2,
        directional_adam_epsilon=args.procoim_directional_adam_epsilon,
        directional_adam_weight_decay=(
            args.procoim_directional_adam_weight_decay
        ),
        directional_adam_projection_iterations=(
            args.procoim_directional_adam_projection_iterations
        ),
        directional_basis_mode=args.procoim_directional_basis_mode,
        directional_factor_rank=args.procoim_directional_factor_rank,
        directional_basis_rotation_banks=(
            args.procoim_directional_basis_rotation_banks
        ),
        directional_basis_rotation_forwards=(
            args.procoim_directional_basis_rotation_forwards
        ),
        directional_basis_rotation_stall_banks=(
            args.procoim_directional_basis_rotation_stall_banks
        ),
        directional_basis_consolidation_cosine_margin=(
            args.procoim_directional_basis_consolidation_cosine_margin
        ),
        directional_basis_consolidation_ema_beta=(
            args.procoim_directional_basis_consolidation_ema_beta
        ),
        directional_accumulation_batches=(
            args.procoim_directional_accumulation_batches
        ),
        reference_reduction_fraction=(
            args.procoim_reference_reduction_fraction
        ),
        candidate_backtrack_factors=tuple(
            args.procoim_candidate_backtrack_factors
        ),
        adaptive_probe_scale=args.procoim_adaptive_probe_scale,
        probe_scale_growth=args.procoim_probe_scale_growth,
        maximum_probe_scale=args.procoim_maximum_probe_scale,
        adaptive_group_selection=args.procoim_adaptive_group_selection,
        group_score_beta=args.procoim_group_score_beta,
        group_exploration=args.procoim_group_exploration,
        anchor_regression_fraction=args.procoim_anchor_regression_fraction,
        anchor_soft_regression_fraction=(
            args.procoim_anchor_soft_regression_fraction
        ),
        anchor_hard_regression_fraction=(
            args.procoim_anchor_hard_regression_fraction
        ),
        anchor_minimum_retained_reduction_fraction=(
            args.procoim_anchor_minimum_retained_reduction_fraction
        ),
        anchor_absolute_tolerance=args.procoim_anchor_absolute_tolerance,
        anchor_weight=args.procoim_anchor_weight,
        anchor_hard_protection=args.procoim_anchor_hard_protection,
        parameter_group_granularity=(
            args.procoim_parameter_group_granularity
        ),
        parameter_group_schedule=tuple(args.procoim_parameter_groups),
        completion_mode=args.procoim_completion_mode,
        protected_fraction=args.procoim_protected_fraction,
        seed=args.seed if args.procoim_seed is None else args.procoim_seed,
    )
