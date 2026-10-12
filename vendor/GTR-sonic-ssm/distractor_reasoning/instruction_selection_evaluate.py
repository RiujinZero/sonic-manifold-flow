"""Held-out evaluation for the Instruction-Selection generator.

Unlike every other generator in this package, answers aren't uniformly
numeric-or-UNKNOWN, so parsing is dispatched per operation: retrieve/
aggregate reuse the shared digit/UNKNOWN parser, identify_holder/
classify_observed scan for known name/noun vocabulary, reject_unsupported
scans for yes/no phrasing, and repeat compares the stripped completion
text directly.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .evaluate import default_parse_answer
from .instruction_selection import (
    OPERATIONS,
    InstructionAnswer,
    InstructionExample,
    generate_instruction_example,
)
from .instruction_selection_shortcuts import INSTRUCTION_SHORTCUT_NAMES
from .vocab import Noun

PredictFn = Callable[[str], str]
InstructionPrediction = Optional[InstructionAnswer]


def _strip_prefix(raw_text: str) -> str:
    text = raw_text.strip()
    for prefix in ("Answer:", "A:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if text:
        text = text.splitlines()[0].strip()
    return text


def _contains_word(text: str, word: str) -> bool:
    # Word-boundary match -- a naive substring check would let "car"
    # false-positive-match inside "Carlos", or a short name inside a
    # longer one.
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def parse_names(text: str, candidates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({name for name in candidates if _contains_word(text, name)}))


def parse_nouns(text: str, nouns: dict[str, Noun]) -> tuple[str, ...]:
    found = {
        key for key, info in nouns.items()
        if _contains_word(text, info.singular) or _contains_word(text, info.plural)
    }
    return tuple(sorted(found))


def parse_yes_no(text: str) -> Optional[str]:
    lowered = text.lower()
    no_markers = ("no,", "no.", "isn't enough", "is not enough", "not enough information", "cannot", "can't")
    if lowered.startswith("no") or any(m in lowered for m in no_markers):
        return "NO"
    yes_markers = ("yes,", "yes.", "there's enough", "there is enough", "enough information")
    if lowered.startswith("yes") or any(m in lowered for m in yes_markers):
        return "YES"
    return None


def parse_instruction_answer(
    operation: str, raw_text: str, *, names: tuple[str, ...], nouns: dict[str, Noun],
) -> InstructionPrediction:
    if operation in ("retrieve", "aggregate"):
        return default_parse_answer(raw_text)
    text = _strip_prefix(raw_text)
    if operation == "identify_holder":
        return parse_names(text, names)
    if operation == "classify_observed":
        return parse_nouns(text, nouns)
    if operation == "reject_unsupported":
        return parse_yes_no(text)
    if operation == "repeat":
        return text
    raise ValueError(f"unknown operation: {operation!r}")


def generate_instruction_eval_examples(
    n: int,
    *,
    seed: int = 12_345,
    level: Optional[int] = None,
    level_range: tuple[int, int] = (1, 4),
    style: str = "qa",
) -> list[InstructionExample]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        chosen_level = level if level is not None else rng.randint(*level_range)
        examples.append(generate_instruction_example(rng, chosen_level, split="eval", style=style))
    return examples


@dataclass
class InstructionEvalResult:
    example: InstructionExample
    raw: str
    prediction: InstructionPrediction

    @property
    def correct(self) -> bool:
        return self.prediction == self.example.answer_value


def _accuracy(results: list[InstructionEvalResult]) -> Optional[float]:
    if not results:
        return None
    return sum(1 for r in results if r.correct) / len(results)


def _reparse_for_shortcut(
    shortcut_name: str, example: InstructionExample, raw: str,
) -> InstructionPrediction:
    """Shortcut predictions don't share the true operation's answer type
    (that mismatch IS the failure mode -- e.g. emitting a number where a
    name was expected), so re-parsing the raw completion with the
    operation's own parser would always report "no match" for those,
    regardless of whether the model actually followed the shortcut.
    ``word_problem_reflex``/``dominant_task_collapse``/
    ``default_qa_behavior`` always predict a number; ``keyword_routing``
    predicts the *other* wh-operation's answer shape."""

    if shortcut_name == "keyword_routing":
        text = _strip_prefix(raw)
        if example.operation == "identify_holder":
            return parse_nouns(text, example.scenario.nouns)
        if example.operation == "classify_observed":
            return parse_names(text, example.scenario.names)
        return None
    return default_parse_answer(raw)


def evaluate_instruction_model(
    predict_fn: PredictFn,
    examples: list[InstructionExample],
) -> dict[str, object]:
    results = []
    for ex in examples:
        raw = predict_fn(ex.prompt)
        prediction = parse_instruction_answer(
            ex.operation, raw, names=ex.scenario.names, nouns=ex.scenario.nouns,
        )
        results.append(InstructionEvalResult(example=ex, raw=raw, prediction=prediction))

    by_operation: dict[str, Optional[float]] = {}
    for operation in OPERATIONS:
        subset = [r for r in results if r.example.operation == operation]
        by_operation[operation] = _accuracy(subset)

    wrong = [r for r in results if not r.correct]
    shortcut_selection_rate: dict[str, Optional[float]] = {}
    for name in INSTRUCTION_SHORTCUT_NAMES:
        applicable = [r for r in wrong if name in r.example.shortcut_predictions]
        if not applicable:
            shortcut_selection_rate[name] = None
            continue
        matched = sum(
            1 for r in applicable
            if _reparse_for_shortcut(name, r.example, r.raw) == r.example.shortcut_predictions[name]
        )
        shortcut_selection_rate[name] = matched / len(applicable)

    def _explained_by_any_shortcut(r: InstructionEvalResult) -> bool:
        return any(
            _reparse_for_shortcut(name, r.example, r.raw) == value
            for name, value in r.example.shortcut_predictions.items()
        )

    shortcut_explained_wrong_rate = (
        sum(1 for r in wrong if _explained_by_any_shortcut(r)) / len(wrong)
    ) if wrong else None

    return {
        "overall_accuracy": _accuracy(results),
        "accuracy_by_operation": by_operation,
        "shortcut_selection_rate": shortcut_selection_rate,
        "shortcut_explained_wrong_rate": shortcut_explained_wrong_rate,
        "num_examples": len(results),
    }
