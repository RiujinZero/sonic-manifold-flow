"""Configuration surface for Procoim-S (Surprise-Gated Fast Plasticity and
Slow Consolidation).

This is the config-only entry point for the continual-learning extension
described in the Procoim-S design specification (see
``Papers/GTRL_PROCOIM_TRACE.md`` for the architecture/mechanism trace this
extends). It intentionally defines nothing beyond the dataclass: as of this
module's introduction only Phase 1 (the fast-weight container in
``fast_weights.py``) is implemented, and ``enabled`` defaults to ``False``
so importing or constructing this config has no effect on the existing
Procoim training path.

Every default below is an initial hypothesis from the design spec, not a
validated setting — see the spec's own framing in its configuration section.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ProcoimSConfig"]


@dataclass
class ProcoimSConfig:
    enabled: bool = False

    # Surprise triage
    weight_ticket_threshold: float = 0.55
    minimum_reliability: float = 0.65
    minimum_confirmations: int = 2
    forgetting_surprise_warning: float = 2.0
    forgetting_surprise_repair: float = 3.0

    # Fast weights
    fast_weight_max_norm: float = 0.02
    fast_weight_decay: float = 0.995
    fast_weight_window_examples: int = 16
    fast_weight_max_updates: int = 32

    # Candidate search
    minimum_candidate_directions: int = 16
    maximum_candidate_directions: int = 256
    candidate_direction_growth: float = 2.0
    factor_rank: int = 1
    probe_epsilon: float = 1e-3
    probe_routing_mode: str = "hard"  # hard | soft
    soft_probe_temperature: float = 1.25

    # Direction dictionary
    anchor_direction_fraction: float = 0.50
    recent_direction_fraction: float = 0.25
    exploration_direction_fraction: float = 0.25
    direction_utility_ema: float = 0.95
    stall_banks_before_refresh: int = 8
    transfer_moments_on_refresh: bool = True
    new_direction_variance_floor: float = 1e-4

    # Utility
    replay_weight: float = 1.0
    future_utility_weight: float = 0.5
    delta_norm_weight: float = 1e-3
    protection_debt_weight: float = 0.25
    minimum_utility: float = 0.0
    prefix_margin_weight: float = 0.10

    # Replay
    replay_capacity: int = 256
    replay_high_priority_per_check: int = 2
    replay_foundation_per_check: int = 1
    replay_random_per_check: int = 1
    replay_soft_tolerance: float = 0.02
    replay_catastrophic_tolerance: float = 0.20
    hard_canary_count: int = 2

    # Probation
    future_utility_probe_rate: float = 0.05
    minimum_future_trials: int = 8
    minimum_contexts_helped: int = 3
    provisional_expiry_examples: int = 64

    # Consolidation
    consolidation_interval_examples: int = 16
    consolidation_audit_items: int = 64
    consolidation_line_search: tuple[float, ...] = field(
        default_factory=lambda: (1.0, 0.75, 0.5, 0.25, 0.125)
    )
    allow_componentwise_commit: bool = True

    # Repair
    repair_queue_capacity: int = 32
    repair_budget_per_consolidation: int = 4

    # Diagnostics
    log_direction_capture: bool = True
    log_memory_ablation: bool = True
    log_fast_weight_ablation: bool = True

    def __post_init__(self) -> None:
        if self.probe_routing_mode not in ("hard", "soft"):
            raise ValueError("probe_routing_mode must be 'hard' or 'soft'")
        fractions = (
            self.anchor_direction_fraction,
            self.recent_direction_fraction,
            self.exploration_direction_fraction,
        )
        if any(value < 0 for value in fractions):
            raise ValueError("direction dictionary fractions must be nonnegative")
        total = sum(fractions)
        if total > 0 and abs(total - 1.0) > 1e-6:
            raise ValueError(
                "anchor_direction_fraction + recent_direction_fraction + "
                "exploration_direction_fraction must sum to 1.0, got "
                f"{total}"
            )
        if not 0.0 <= self.fast_weight_decay <= 1.0:
            raise ValueError("fast_weight_decay must be in [0, 1]")
        if self.fast_weight_max_norm < 0:
            raise ValueError("fast_weight_max_norm must be nonnegative")
        if self.minimum_candidate_directions <= 0:
            raise ValueError("minimum_candidate_directions must be positive")
        if self.maximum_candidate_directions < self.minimum_candidate_directions:
            raise ValueError(
                "maximum_candidate_directions must be >= "
                "minimum_candidate_directions"
            )
        if not self.consolidation_line_search:
            raise ValueError("consolidation_line_search must not be empty")
        if any(not 0.0 < alpha <= 1.0 for alpha in self.consolidation_line_search):
            raise ValueError("consolidation_line_search values must be in (0, 1]")
