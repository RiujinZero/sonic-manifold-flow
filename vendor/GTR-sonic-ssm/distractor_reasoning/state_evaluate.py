"""Held-out evaluation for the State-Update generator -- mirrors
``evaluate.py``'s role for the Binding generator, adapted to the
event-sequence example shape (``StateExample``) and its own shortcut set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .evaluate import Prediction, default_parse_answer
from .state_shortcuts import STATE_SHORTCUT_NAMES
from .state_update import ALL_STATE_DISTRACTOR_KINDS, StateExample, generate_state_update_example

PredictFn = Callable[[str], str]


def generate_state_eval_examples(
    n: int,
    *,
    seed: int = 12_345,
    level: Optional[int] = None,
    level_range: tuple[int, int] = (1, 8),
    style: str = "qa",
) -> list[StateExample]:
    """A fresh, seeded held-out stream -- same contract as
    ``evaluate.generate_eval_examples``, always using ``split="eval"``."""

    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        chosen_level = level if level is not None else rng.randint(*level_range)
        examples.append(generate_state_update_example(rng, chosen_level, split="eval", style=style))
    return examples


@dataclass
class StateEvalResult:
    example: StateExample
    prediction: Prediction

    @property
    def correct(self) -> bool:
        return self.prediction == self.example.answer_value


def _accuracy(results: list[StateEvalResult]) -> Optional[float]:
    if not results:
        return None
    return sum(1 for r in results if r.correct) / len(results)


def evaluate_state_model(
    predict_fn: PredictFn,
    examples: list[StateExample],
    *,
    parse_fn=default_parse_answer,
) -> dict[str, object]:
    results = [
        StateEvalResult(example=ex, prediction=parse_fn(predict_fn(ex.prompt)))
        for ex in examples
    ]

    unknown = [r for r in results if r.example.is_unknown_case]
    answerable = [r for r in results if not r.example.is_unknown_case]
    counterparty = [r for r in results if r.example.queries_counterparty]
    inverse_pair = [r for r in results if r.example.has_inverse_pair]

    by_distractor_type: dict[str, Optional[float]] = {}
    for kind in ALL_STATE_DISTRACTOR_KINDS:
        subset = [r for r in results if kind in r.example.distractor_kinds]
        by_distractor_type[kind] = _accuracy(subset)

    by_num_events: dict[int, Optional[float]] = {}
    for num_events in sorted({r.example.num_events for r in results}):
        subset = [r for r in results if r.example.num_events == num_events]
        by_num_events[num_events] = _accuracy(subset)

    phantom_answers = sum(1 for r in unknown if isinstance(r.prediction, int))

    wrong = [r for r in results if not r.correct]
    shortcut_selection_rate: dict[str, Optional[float]] = {}
    for name in STATE_SHORTCUT_NAMES:
        applicable = [r for r in wrong if name in r.example.shortcut_predictions]
        if not applicable:
            shortcut_selection_rate[name] = None
            continue
        matched = sum(
            1 for r in applicable
            if r.prediction == r.example.shortcut_predictions[name]
        )
        shortcut_selection_rate[name] = matched / len(applicable)
    shortcut_explained_wrong_rate = (
        sum(1 for r in wrong if r.prediction in r.example.shortcut_predictions.values())
        / len(wrong)
    ) if wrong else None

    return {
        "overall_accuracy": _accuracy(results),
        "answerable_accuracy": _accuracy(answerable),
        "unknown_accuracy": _accuracy(unknown),
        "counterparty_query_accuracy": _accuracy(counterparty),
        "inverse_pair_accuracy": _accuracy(inverse_pair),
        "accuracy_by_distractor_type": by_distractor_type,
        "accuracy_by_num_events": by_num_events,
        "phantom_answer_rate": (phantom_answers / len(unknown)) if unknown else None,
        "shortcut_selection_rate": shortcut_selection_rate,
        "shortcut_explained_wrong_rate": shortcut_explained_wrong_rate,
        "num_examples": len(results),
        "num_answerable": len(answerable),
        "num_unknown": len(unknown),
    }
