"""Held-out evaluation stream and metrics.

Uses ``split="eval"`` throughout (held-out names/nouns/templates, see
``vocab.py``), so surface memorization of anything the training generator
could have produced cannot inflate these numbers. The goal isn't just
arithmetic accuracy -- most of these metrics exist to catch a model that
answers correctly on average while actually failing to maintain
subject+relation+object bindings (e.g. one that's secretly pattern-matching
on partial overlap, or one that hallucinates a plausible-looking number
instead of admitting a question is unanswerable).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

from .difficulty import ALL_DISTRACTOR_KINDS
from .generate import Example, generate_example
from .query import UNKNOWN
from .shortcuts import SHORTCUT_NAMES

Prediction = Union[int, Literal["UNKNOWN"], None]
PredictFn = Callable[[str], str]
ParseFn = Callable[[str], Prediction]

# Answer targets are full sentences (see render.py/state_update.py), so
# parsing looks for stable substrings rather than expecting the value to
# be the very first token. Every UNKNOWN-flavored answer template (across
# every generator in this package) is written to contain at least one of
# these phrases -- add to this list, not around it, if a new template
# needs different wording.
UNKNOWN_INDICATORS: tuple[str, ...] = (
    "enough information",
    "nothing here that tells me",
    "can't tell",
    "cannot tell",
)


def default_parse_answer(raw_text: str) -> Prediction:
    """Parse a model's raw completion -- a full sentence, e.g. "Kevin has
    6 parrots." or "I don't have enough information to know how many
    parrots Kevin has." -- into an int, ``"UNKNOWN"``, or ``None``
    (unparseable -- always scored as incorrect)."""

    text = raw_text.strip()
    for prefix in ("Answer:", "A:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if text:
        text = text.splitlines()[0].strip()
    lowered = text.lower()
    if any(indicator in lowered for indicator in UNKNOWN_INDICATORS):
        return UNKNOWN
    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group())
    return None


def generate_eval_examples(
    n: int,
    *,
    seed: int = 12_345,
    level: Optional[int] = None,
    level_range: tuple[int, int] = (1, 10),
    style: str = "qa",
) -> list[Example]:
    """A fresh, seeded held-out stream -- call with the same seed to get
    a reproducible eval set, or a different seed for a fresh one."""

    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        chosen_level = level if level is not None else rng.randint(*level_range)
        examples.append(generate_example(rng, chosen_level, split="eval", style=style))
    return examples


@dataclass
class EvalResult:
    example: Example
    prediction: Prediction

    @property
    def correct(self) -> bool:
        return self.prediction == self.example.answer_value


def _accuracy(results: list[EvalResult]) -> Optional[float]:
    if not results:
        return None
    return sum(1 for r in results if r.correct) / len(results)


def evaluate_model(
    predict_fn: PredictFn,
    examples: list[Example],
    *,
    parse_fn: ParseFn = default_parse_answer,
) -> dict[str, object]:
    """Run ``predict_fn`` (prompt text -> raw completion text) over every
    example and compute the full metrics suite. ``predict_fn`` is the only
    place this module touches an actual model, so it stays decoupled from
    any specific model/tokenizer -- callers wrap whatever they're testing.
    """

    results = [
        EvalResult(example=ex, prediction=parse_fn(predict_fn(ex.prompt)))
        for ex in examples
    ]

    answerable = [r for r in results if not r.example.is_unknown_case]
    unknown = [r for r in results if r.example.is_unknown_case]
    zero_or_unknown = [r for r in results if r.example.is_zero_case or r.example.is_unknown_case]
    category = [r for r in results if r.example.is_category_question]

    by_distractor_type: dict[str, Optional[float]] = {}
    for kind in ALL_DISTRACTOR_KINDS:
        subset = [r for r in results if kind in r.example.distractor_kinds]
        by_distractor_type[kind] = _accuracy(subset)

    by_num_facts: dict[int, Optional[float]] = {}
    for num_facts in sorted({r.example.num_facts for r in results}):
        subset = [r for r in results if r.example.num_facts == num_facts]
        by_num_facts[num_facts] = _accuracy(subset)

    by_num_distractors: dict[int, Optional[float]] = {}
    for num_distractors in sorted({r.example.num_distractors for r in results}):
        subset = [r for r in results if r.example.num_distractors == num_distractors]
        by_num_distractors[num_distractors] = _accuracy(subset)

    phantom_answers = sum(1 for r in unknown if isinstance(r.prediction, int))

    wrong = [r for r in results if not r.correct]
    shortcut_selection_rate: dict[str, Optional[float]] = {}
    for name in SHORTCUT_NAMES:
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
        "zero_vs_unknown_accuracy": _accuracy(zero_or_unknown),
        "category_aggregation_accuracy": _accuracy(category),
        "accuracy_by_distractor_type": by_distractor_type,
        "accuracy_by_num_facts": by_num_facts,
        "accuracy_by_num_distractors": by_num_distractors,
        "phantom_answer_rate": (phantom_answers / len(unknown)) if unknown else None,
        # Of the model's WRONG answers, what fraction match the specific
        # value each naive shortcut would have guessed -- pinpoints which
        # shortcut (if any) explains a given failure mode, vs. noise.
        "shortcut_selection_rate": shortcut_selection_rate,
        "shortcut_explained_wrong_rate": shortcut_explained_wrong_rate,
        "num_examples": len(results),
        "num_answerable": len(answerable),
        "num_unknown": len(unknown),
    }
