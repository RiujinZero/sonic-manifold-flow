"""Held-out evaluation for the Ontology generator -- mirrors
``evaluate.py``'s role for the Binding generator, adapted to
``OntologyExample`` and its own shortcut set. Always uses ``split="eval"``,
whose taxonomy uses entirely different invented words than train's, so a
model can only score well here by genuinely reading the in-context
definitions rather than recalling the training taxonomy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .evaluate import Prediction, default_parse_answer
from .ontology import ALL_DISTRACTOR_KINDS, OntologyExample, generate_ontology_example
from .ontology_shortcuts import ONTOLOGY_SHORTCUT_NAMES

PredictFn = Callable[[str], str]


def generate_ontology_eval_examples(
    n: int,
    *,
    seed: int = 12_345,
    level: Optional[int] = None,
    level_range: tuple[int, int] = (1, 6),
    style: str = "qa",
) -> list[OntologyExample]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        chosen_level = level if level is not None else rng.randint(*level_range)
        examples.append(generate_ontology_example(rng, chosen_level, split="eval", style=style))
    return examples


@dataclass
class OntologyEvalResult:
    example: OntologyExample
    prediction: Prediction

    @property
    def correct(self) -> bool:
        return self.prediction == self.example.answer_value


def _accuracy(results: list[OntologyEvalResult]) -> Optional[float]:
    if not results:
        return None
    return sum(1 for r in results if r.correct) / len(results)


def evaluate_ontology_model(
    predict_fn: PredictFn,
    examples: list[OntologyExample],
    *,
    parse_fn=default_parse_answer,
) -> dict[str, object]:
    results = [
        OntologyEvalResult(example=ex, prediction=parse_fn(predict_fn(ex.prompt)))
        for ex in examples
    ]

    unknown = [r for r in results if r.example.is_unknown_case]
    answerable = [r for r in results if not r.example.is_unknown_case]
    aggregation = [r for r in results if r.example.is_aggregation]

    by_query_level: dict[str, Optional[float]] = {}
    for query_level in ("leaf", "parent", "grandparent"):
        subset = [r for r in results if r.example.query_level == query_level]
        by_query_level[query_level] = _accuracy(subset)

    by_distractor_type: dict[str, Optional[float]] = {}
    for kind in ALL_DISTRACTOR_KINDS:
        subset = [r for r in results if kind in r.example.distractor_kinds]
        by_distractor_type[kind] = _accuracy(subset)

    phantom_answers = sum(1 for r in unknown if isinstance(r.prediction, int))

    wrong = [r for r in results if not r.correct]
    shortcut_selection_rate: dict[str, Optional[float]] = {}
    for name in ONTOLOGY_SHORTCUT_NAMES:
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
        "aggregation_accuracy": _accuracy(aggregation),
        "accuracy_by_query_level": by_query_level,
        "accuracy_by_distractor_type": by_distractor_type,
        "phantom_answer_rate": (phantom_answers / len(unknown)) if unknown else None,
        "shortcut_selection_rate": shortcut_selection_rate,
        "shortcut_explained_wrong_rate": shortcut_explained_wrong_rate,
        "num_examples": len(results),
        "num_answerable": len(answerable),
        "num_unknown": len(unknown),
    }
