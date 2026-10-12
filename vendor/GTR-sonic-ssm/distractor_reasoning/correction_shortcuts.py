"""Naive shortcuts a model could exploit on the Correction task instead of
genuinely tracking "the last statement about this pair wins," mirroring
``shortcuts.py``'s role for the Binding generator.

    Kevin owns 7 parrots.
    Correction: Kevin owns 4 parrots.
    Maria owns 9 parrots.
    Q: How many parrots does Kevin own?  -> 4 (correct)
        first_fact_anchoring:    7   (keeps only the original value)
        last_number_shortcut:    9   (Maria's decoy, textually last overall)
        merge_strategy:          11  (7 + 4, combines old and new)
        correction_word_shortcut: 4  (here it coincides with correct --
                                       diverges specifically when the last
                                       "Correction:"-phrased line in the
                                       text belongs to a *different*
                                       subject than the one being asked
                                       about)
"""

from __future__ import annotations

from typing import Optional

from .correction import Statement
from .query import Query


def first_fact_anchoring(statements: list[Statement], query: Query) -> Optional[int]:
    key = (query.subject, query.target)
    for s in statements:
        if (s.subject, s.noun) == key:
            return s.quantity
    return None


def last_number_shortcut(statements: list[Statement], query: Query) -> Optional[int]:
    if not statements:
        return None
    return statements[-1].quantity


def merge_strategy(statements: list[Statement], query: Query) -> Optional[int]:
    key = (query.subject, query.target)
    matches = [
        s.quantity for s in statements
        if (s.subject, s.noun) == key and s.kind != "decoy_observation"
    ]
    if not matches:
        return None
    return sum(matches)


def correction_word_shortcut(statements: list[Statement], query: Query) -> Optional[int]:
    correction_statements = [s for s in statements if s.kind in ("correction", "decoy_correction")]
    if not correction_statements:
        return None
    return correction_statements[-1].quantity


CORRECTION_SHORTCUTS = {
    "first_fact_anchoring": first_fact_anchoring,
    "last_number_shortcut": last_number_shortcut,
    "merge_strategy": merge_strategy,
    "correction_word_shortcut": correction_word_shortcut,
}

CORRECTION_SHORTCUT_NAMES: tuple[str, ...] = tuple(CORRECTION_SHORTCUTS.keys())


def correction_shortcut_predictions(statements: list[Statement], query: Query) -> dict[str, int]:
    predictions: dict[str, int] = {}
    for name, fn in CORRECTION_SHORTCUTS.items():
        value = fn(statements, query)
        if value is not None:
            predictions[name] = value
    return predictions
