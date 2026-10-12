"""Surprise-gated fast-weight ticketing (Procoim-S design spec sections 7-8).

``TicketController`` decides whether a surprising, reliable, *recurring*
event earns the right to open a :class:`FastWeightTicket` — authorization
for Procoim to spend a bounded forward budget searching specific parameter
groups. Surprise alone never opens a ticket (section 7.4's triage table);
reliability and recurrence both gate it, matching
``weight_ticket_threshold``/``minimum_reliability``/``minimum_confirmations``
already defined in ``ProcoimSConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from gator.optimizers.procoim_s import ProcoimSConfig

__all__ = ["FastWeightTicket", "TicketController"]


@dataclass(frozen=True)
class FastWeightTicket:
    """Authorization for Procoim to search specific parameter groups for a
    limited time (design spec section 8, reference dataclass)."""

    event_id: int
    source_example_id: int
    eligible_groups: tuple[str, ...]
    surprise: float
    reliability: float
    recurrence: float
    priority: float
    opened_step: int
    expires_step: int
    max_forward_budget: int
    max_delta_norm: float
    context_signature: Tensor | None = None

    def is_expired(self, current_step: int) -> bool:
        return current_step >= self.expires_step


class TicketController:
    """Tracks recurrence per signature and decides whether to open a ticket.

    ``signature`` is caller-defined — anything that identifies "this is the
    same recurring pattern" (e.g. a quantized/hashed context embedding, or
    the semantic-novelty match index into stable memory). The controller
    itself only counts how many times a signature has been seen; it has no
    opinion on what makes two events "the same."
    """

    def __init__(self, config: ProcoimSConfig) -> None:
        self.config = config
        self._next_event_id = 0
        self._recurrence_counts: dict[str, int] = {}

    def record_recurrence(self, signature: str) -> int:
        count = self._recurrence_counts.get(signature, 0) + 1
        self._recurrence_counts[signature] = count
        return count

    def weight_write_eligibility(
        self,
        *,
        surprise: float,
        reliability: float,
        recurrence: float,
        future_utility: float | None = None,
    ) -> float:
        """``E_weight = S * R * F`` (v0.1, section 7.3); with an optional
        ``future_utility`` term this becomes ``S * R * F * max(U, 0)``."""

        eligibility = surprise * reliability * recurrence
        if future_utility is not None:
            eligibility *= max(future_utility, 0.0)
        return eligibility

    def maybe_open(
        self,
        *,
        source_example_id: int,
        surprise: float,
        reliability: float,
        signature: str,
        eligible_groups: tuple[str, ...],
        opened_step: int,
        max_forward_budget: int | None = None,
        context_signature: Tensor | None = None,
    ) -> FastWeightTicket | None:
        """Section 7.4's triage table, operationalized: a ticket opens only
        for *repeated* high-surprise, high-reliability events — one-off
        surprises (however reliable) stay in memory (SGBM's job), not a
        weight-update ticket.
        """

        confirmations = self.record_recurrence(signature)
        if reliability < self.config.minimum_reliability:
            return None
        if confirmations < self.config.minimum_confirmations:
            return None

        recurrence = min(1.0, confirmations / max(1, self.config.minimum_confirmations))
        eligibility = self.weight_write_eligibility(
            surprise=surprise, reliability=reliability, recurrence=recurrence
        )
        if eligibility < self.config.weight_ticket_threshold:
            return None

        budget = (
            max_forward_budget
            if max_forward_budget is not None
            else 2 * self.config.maximum_candidate_directions
        )
        ticket = FastWeightTicket(
            event_id=self._next_event_id,
            source_example_id=source_example_id,
            eligible_groups=eligible_groups,
            surprise=surprise,
            reliability=reliability,
            recurrence=recurrence,
            priority=eligibility,
            opened_step=opened_step,
            expires_step=opened_step + self.config.fast_weight_window_examples,
            max_forward_budget=budget,
            max_delta_norm=self.config.fast_weight_max_norm,
            context_signature=context_signature,
        )
        self._next_event_id += 1
        return ticket
