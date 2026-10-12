"""Inference-time session-scoped adaptation (Procoim-S design spec section 20).

Enforces the spec's permission boundary as actual code, not just a
convention:

- fast-weight adaptation during inference is allowed, but isolated per
  session/user, restricted to an explicit allowlist of parameter groups
  ("selected routing/readout adapters only"), and bounded in norm;
- direct ``theta_slow`` modification (consolidation) is *never* automatic —
  :func:`consolidate_session` is the only path into
  :func:`~gator.optimizers.consolidation.consolidate`, and it raises
  ``PermissionError`` without an explicit ``approved=True``, matching
  section 20's "after user or system approval."
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch import nn

from gator.optimizers.candidate_utility import ReplayEvaluation
from gator.optimizers.consolidation import ConsolidationResult, consolidate
from gator.optimizers.fast_weights import FastWeightDelta, FastWeightState
from gator.optimizers.procoim_s import ProcoimSConfig

__all__ = [
    "SessionAdaptationPolicy",
    "SessionFastWeightRegistry",
    "consolidate_session",
]


@dataclass(frozen=True)
class SessionAdaptationPolicy:
    """What a session is allowed to adapt, and how much (design spec section
    20's "optionally allowed: small session-scoped delta_fast, selected
    routing/readout adapters only")."""

    allowed_groups: tuple[str, ...]
    max_session_delta_norm: float

    def __post_init__(self) -> None:
        if not self.allowed_groups:
            raise ValueError("allowed_groups must not be empty")
        if self.max_session_delta_norm <= 0:
            raise ValueError("max_session_delta_norm must be positive")


class SessionFastWeightRegistry:
    """Per-session (user) isolated :class:`FastWeightState`.

    Each session gets its own state object — never shared, never merged —
    so one user's inference-time adaptation can never leak into another's
    (design spec section 20: "for user-specific systems, session fast
    weights and memory must remain isolated per user").
    """

    def __init__(self, *, policy: SessionAdaptationPolicy) -> None:
        self.policy = policy
        self._sessions: dict[str, FastWeightState] = {}

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_or_create(self, session_id: str) -> FastWeightState:
        state = self._sessions.get(session_id)
        if state is None:
            state = FastWeightState()
            self._sessions[session_id] = state
        return state

    def apply_session_delta(
        self,
        session_id: str,
        delta: FastWeightDelta,
        *,
        alpha: float = 1.0,
    ) -> None:
        """Merge ``delta`` into this session's fast weights, enforcing the
        group allowlist and clamping to ``max_session_delta_norm``."""

        if delta.group not in self.policy.allowed_groups:
            raise PermissionError(
                f"group {delta.group!r} is not in the session-adaptation "
                f"allowlist {self.policy.allowed_groups}"
            )
        state = self.get_or_create(session_id)
        state.add_group(delta, alpha=alpha)
        group_delta = state.get_group(delta.group)
        norm = group_delta.norm()
        if norm > self.policy.max_session_delta_norm:
            group_delta.scale_(self.policy.max_session_delta_norm / norm)

    def end_session(self, session_id: str) -> FastWeightState | None:
        """Remove and return a session's fast weights (e.g. handing them to
        an offline consolidation review). Ending a session never itself
        commits anything into ``theta_slow`` — see :func:`consolidate_session`.
        """

        return self._sessions.pop(session_id, None)

    def no_shared_state_across_sessions(self) -> bool:
        """Invariant check: no two session IDs share the same underlying
        :class:`FastWeightState` object."""

        seen_ids: set[int] = set()
        for state in self._sessions.values():
            if id(state) in seen_ids:
                return False
            seen_ids.add(id(state))
        return True


def consolidate_session(
    session_id: str,
    registry: SessionFastWeightRegistry,
    model: nn.Module,
    *,
    evaluate: Callable[[], tuple[float, Sequence[ReplayEvaluation]]],
    config: ProcoimSConfig,
    approved: bool,
) -> ConsolidationResult:
    """The only path from a session's fast weights into ``theta_slow``.

    Raises ``PermissionError`` unless ``approved=True`` is passed
    explicitly — this function is never meant to be called automatically as
    part of ordinary inference request handling; the caller (a human
    reviewer, or an offline batch job with its own approval logic) must opt
    in for this specific session (design spec section 20: consolidation
    should occur offline, during idle time, after approval, and after a
    broad replay audit — the last of those is the caller's ``evaluate``
    callable's job, same as in ``consolidation.py``).
    """

    if not approved:
        raise PermissionError(
            "consolidating a session's fast weights into theta_slow requires "
            "explicit approval (design spec section 20)"
        )
    state = registry.get_or_create(session_id)
    return consolidate(model, state, evaluate=evaluate, config=config)
