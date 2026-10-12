"""Repair queue and repair utility (Procoim-S design spec section 15.1-15.2).

A REPAIR- or CRITICAL-severity forgetting event (``retention_monitor.py``)
enqueues a :class:`RepairTicket` targeting the parameter groups active when
the forgotten skill was last learned. Repair does not require restoring the
exact historical minimum loss — only the skill's original contract ceiling
(design spec section 15.2).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from gator.optimizers.procoim_s import ProcoimSConfig
from gator.optimizers.retention_monitor import ForgettingEvent, ForgettingSeverity

__all__ = ["RepairTicket", "RepairQueue", "repair_utility"]


@dataclass(frozen=True)
class RepairTicket:
    item_id: str
    severity: ForgettingSeverity
    target_groups: tuple[str, ...]
    opened_step: int


class RepairQueue:
    """A capacity-bounded FIFO of open :class:`RepairTicket` (design spec
    section 15: WARNING raises replay priority elsewhere and never enqueues
    here; only REPAIR and CRITICAL open a ticket)."""

    def __init__(self, config: ProcoimSConfig) -> None:
        self.config = config
        self._queue: deque[RepairTicket] = deque(maxlen=config.repair_queue_capacity)

    def __len__(self) -> int:
        return len(self._queue)

    def enqueue(
        self,
        event: ForgettingEvent,
        *,
        target_groups: tuple[str, ...],
        opened_step: int,
    ) -> RepairTicket | None:
        if event.severity in (ForgettingSeverity.NONE, ForgettingSeverity.WARNING):
            return None
        ticket = RepairTicket(
            item_id=event.item_id,
            severity=event.severity,
            target_groups=target_groups,
            opened_step=opened_step,
        )
        self._queue.append(ticket)
        return ticket

    def pop_budget(self, budget: int | None = None) -> list[RepairTicket]:
        """Dequeue up to ``budget`` tickets (default:
        ``config.repair_budget_per_consolidation``), highest-severity-first,
        FIFO within a severity tier."""

        budget = budget if budget is not None else self.config.repair_budget_per_consolidation
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        ordered = sorted(
            self._queue,
            key=lambda ticket: list(ForgettingSeverity).index(ticket.severity),
            reverse=True,
        )
        popped = ordered[:budget]
        for ticket in popped:
            self._queue.remove(ticket)
        return popped


def repair_utility(
    gain_forgotten: float,
    current_damage: float,
    other_replay_regression: float,
    *,
    damage_weight: float = 1.0,
    replay_weight: float = 1.0,
) -> float:
    """``U_repair = G_forgotten - lambda*G_current_damage - mu*R_other_replay``
    (design spec section 15.2)."""

    return gain_forgotten - damage_weight * current_damage - replay_weight * other_replay_regression
