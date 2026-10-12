"""Held-out evaluation for the Correction generator -- mirrors
``evaluate.py``'s role for the Binding generator, adapted to
``CorrectionExample`` and its own shortcut set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .correction import ALL_DECOY_KINDS, CorrectionExample, generate_correction_example
from .correction_shortcuts import CORRECTION_SHORTCUT_NAMES
from .evaluate import Prediction, default_parse_answer

PredictFn = Callable[[str], str]


def generate_correction_eval_examples(
    n: int,
    *,
    seed: int = 12_345,
    level: Optional[int] = None,
    level_range: tuple[int, int] = (1, 6),
    style: str = "qa",
) -> list[CorrectionExample]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        chosen_level = level if level is not None else rng.randint(*level_range)
        examples.append(generate_correction_example(rng, chosen_level, split="eval", style=style))
    return examples


@dataclass
class CorrectionEvalResult:
    example: CorrectionExample
    prediction: Prediction

    @property
    def correct(self) -> bool:
        return self.prediction == self.example.answer_value


def _accuracy(results: list[CorrectionEvalResult]) -> Optional[float]:
    if not results:
        return None
    return sum(1 for r in results if r.correct) / len(results)


def evaluate_correction_model(
    predict_fn: PredictFn,
    examples: list[CorrectionExample],
    *,
    parse_fn=default_parse_answer,
) -> dict[str, object]:
    results = [
        CorrectionEvalResult(example=ex, prediction=parse_fn(predict_fn(ex.prompt)))
        for ex in examples
    ]

    unknown = [r for r in results if r.example.is_unknown_case]
    answerable = [r for r in results if not r.example.is_unknown_case]
    corrected = [r for r in results if r.example.num_corrections > 0]
    uncorrected = [r for r in results if r.example.num_corrections == 0]
    pronoun = [r for r in results if r.example.has_pronoun_correction]

    by_decoy_type: dict[str, Optional[float]] = {}
    for kind in ALL_DECOY_KINDS:
        subset = [r for r in results if kind in r.example.decoy_kinds]
        by_decoy_type[kind] = _accuracy(subset)

    by_num_corrections: dict[int, Optional[float]] = {}
    for n in sorted({r.example.num_corrections for r in results}):
        subset = [r for r in results if r.example.num_corrections == n]
        by_num_corrections[n] = _accuracy(subset)

    phantom_answers = sum(1 for r in unknown if isinstance(r.prediction, int))

    wrong = [r for r in results if not r.correct]
    shortcut_selection_rate: dict[str, Optional[float]] = {}
    for name in CORRECTION_SHORTCUT_NAMES:
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
        "corrected_accuracy": _accuracy(corrected),
        "uncorrected_accuracy": _accuracy(uncorrected),
        "pronoun_correction_accuracy": _accuracy(pronoun),
        "accuracy_by_decoy_type": by_decoy_type,
        "accuracy_by_num_corrections": by_num_corrections,
        "phantom_answer_rate": (phantom_answers / len(unknown)) if unknown else None,
        "shortcut_selection_rate": shortcut_selection_rate,
        "shortcut_explained_wrong_rate": shortcut_explained_wrong_rate,
        "num_examples": len(results),
        "num_answerable": len(answerable),
        "num_unknown": len(unknown),
    }
