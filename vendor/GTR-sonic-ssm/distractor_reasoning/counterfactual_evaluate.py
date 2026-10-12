"""Held-out evaluation for the Counterfactual generator.

Unlike every other generator in this package, the headline metric is not
per-example accuracy but *pair* consistency (design doc: "Measure pair
consistency: correct(A) AND correct(B), rather than scoring each
independently"). ``inconsistent_rate`` is the specific diagnostic for "got
one member right by chance but failed the paired contrast" -- a model
relying on superficial pattern matching rather than the real subject/
relation/noun/quantity/belief distinction will show up there even if its
raw per-version accuracy looks fine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .counterfactual import ALL_AXES, CounterfactualExample, generate_counterfactual_example
from .evaluate import Prediction, default_parse_answer

PredictFn = Callable[[str], str]


def generate_counterfactual_eval_examples(
    n: int,
    *,
    seed: int = 12_345,
    level: Optional[int] = None,
    level_range: tuple[int, int] = (1, 6),
    style: str = "qa",
) -> list[CounterfactualExample]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        chosen_level = level if level is not None else rng.randint(*level_range)
        examples.append(generate_counterfactual_example(rng, chosen_level, split="eval", style=style))
    return examples


@dataclass
class CounterfactualEvalResult:
    example: CounterfactualExample
    prediction_a: Prediction
    prediction_b: Prediction

    @property
    def correct_a(self) -> bool:
        return self.prediction_a == self.example.answer_value_a

    @property
    def correct_b(self) -> bool:
        return self.prediction_b == self.example.answer_value_b

    @property
    def pair_consistent(self) -> bool:
        return self.correct_a and self.correct_b

    @property
    def inconsistent(self) -> bool:
        """Exactly one of the pair was answered correctly -- the specific
        signature of superficial pattern matching rather than the real
        distinction."""

        return self.correct_a != self.correct_b


def _rate(flags: list[bool]) -> Optional[float]:
    if not flags:
        return None
    return sum(flags) / len(flags)


def evaluate_counterfactual_model(
    predict_fn: PredictFn,
    examples: list[CounterfactualExample],
    *,
    parse_fn=default_parse_answer,
) -> dict[str, object]:
    results = [
        CounterfactualEvalResult(
            example=ex,
            prediction_a=parse_fn(predict_fn(ex.prompt_a)),
            prediction_b=parse_fn(predict_fn(ex.prompt_b)),
        )
        for ex in examples
    ]

    by_axis: dict[str, Optional[float]] = {}
    for axis in ALL_AXES:
        subset = [r for r in results if r.example.axis == axis]
        by_axis[axis] = _rate([r.pair_consistent for r in subset])

    return {
        "pair_consistency_rate": _rate([r.pair_consistent for r in results]),
        "version_a_accuracy": _rate([r.correct_a for r in results]),
        "version_b_accuracy": _rate([r.correct_b for r in results]),
        "inconsistent_rate": _rate([r.inconsistent for r in results]),
        "pair_consistency_by_axis": by_axis,
        "num_examples": len(results),
    }
