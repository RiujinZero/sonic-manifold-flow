"""Naive shortcuts a model could exploit on the State-Update task instead
of genuinely replaying the event sequence, plus the value each one would
predict -- mirrors ``shortcuts.py``'s role for the Binding generator.

    Kevin has 5 parrots.
    Kevin receives 3 parrots.
    Kevin gives Maria 2 parrots.
    Q: How many parrots does Kevin have?  -> 6 (correct)
        last_number:            2   (just the most recent number seen)
        sum_all:                10  (5 + 3 + 2, ignoring direction/sign)
        ignore_direction:       10  (5 + 3 + 2, treating give/receive alike)
        ignore_ownership_transfer: 8 (5 + 3, never applies the -2 for giving
                                       -- only wrong when giving/losing is
                                       skipped; here it coincides with
                                       "ignore_direction" only by
                                       coincidence of this small example)
        initial_state_anchoring: 5  (the starting count, no updates applied)

Every function is internal-only bookkeeping, never rendered into text.
"""

from __future__ import annotations

from typing import Optional

from .state_update import Event, StateQuery


def last_number(events: list[Event], query: StateQuery) -> Optional[int]:
    """Ignores structure entirely: the most recent number mentioned for
    this exact (subject, noun) pair, in *either* role (primary actor or
    counterparty) and regardless of whether the line even changes state
    (an observation-only distractor counts too -- that's the point)."""

    key = (query.subject, query.noun)
    value: Optional[int] = None
    for e in events:
        if (e.subject, e.noun) == key or (e.counterparty, e.noun) == key:
            value = e.quantity
    return value


def sum_all(events: list[Event], query: StateQuery) -> Optional[int]:
    """Adds every number associated with the subject, regardless of noun,
    verb, or direction."""

    matches = [e.quantity for e in events if e.subject == query.subject]
    if not matches:
        return None
    return sum(matches)


def ignore_direction(events: list[Event], query: StateQuery) -> Optional[int]:
    """Replays the subject's own events but treats every verb as additive
    -- giving and receiving are counted identically."""

    key = (query.subject, query.noun)
    total = 0
    touched = False
    for e in events:
        if e.is_initial:
            if (e.subject, e.noun) == key:
                total, touched = e.quantity, True
            continue
        if not e.is_state_changing:
            continue
        if (e.subject, e.noun) == key:
            total += e.quantity
            touched = True
    return total if touched else None


def ignore_ownership_transfer(events: list[Event], query: StateQuery) -> Optional[int]:
    """Replays real signed deltas but only ever credits the literal named
    subject of each event -- a counterparty's side of a transfer is never
    updated, so this only diverges from the truth when the query asks
    about the counterparty (receiving) side of a paired event."""

    key = (query.subject, query.noun)
    total = 0
    touched = False
    for e in events:
        if e.is_initial:
            if (e.subject, e.noun) == key:
                total, touched = e.quantity, True
            continue
        if not e.is_state_changing:
            continue
        if (e.subject, e.noun) == key:
            total += e.signed_delta
            touched = True
    return total if touched else None


def initial_state_anchoring(events: list[Event], query: StateQuery) -> Optional[int]:
    """Returns the starting quantity and ignores every subsequent event."""

    key = (query.subject, query.noun)
    for e in events:
        if e.is_initial and (e.subject, e.noun) == key:
            return e.quantity
    return None


STATE_SHORTCUTS = {
    "last_number": last_number,
    "sum_all": sum_all,
    "ignore_direction": ignore_direction,
    "ignore_ownership_transfer": ignore_ownership_transfer,
    "initial_state_anchoring": initial_state_anchoring,
}

STATE_SHORTCUT_NAMES: tuple[str, ...] = tuple(STATE_SHORTCUTS.keys())


def state_shortcut_predictions(events: list[Event], query: StateQuery) -> dict[str, int]:
    predictions: dict[str, int] = {}
    for name, fn in STATE_SHORTCUTS.items():
        value = fn(events, query)
        if value is not None:
            predictions[name] = value
    return predictions
