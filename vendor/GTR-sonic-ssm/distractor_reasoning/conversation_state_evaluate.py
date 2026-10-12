"""Held-out evaluation for the Conversation-State generator -- mirrors
``topic_switch_evaluate.py``'s approach (exact-match on a cleaned
completion, since the target is a full sentence trained verbatim) since
this generator is also multi-turn/dialogue-shaped rather than a single
symbolic answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .conversation_state import (
    CONVERSATION_STATE_DIFFICULTY_LEVELS,
    ConversationStateExample,
    generate_conversation_state_example,
)

PredictFn = Callable[[str], str]


def generate_conversation_state_eval_examples(
    n: int,
    *,
    seed: int = 12_345,
    level: Optional[int] = None,
    level_range: tuple[int, int] = (1, 3),
) -> list[ConversationStateExample]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        chosen_level = level if level is not None else rng.randint(*level_range)
        examples.append(generate_conversation_state_example(rng, chosen_level, split="eval"))
    return examples


def _clean(text: str) -> str:
    t = text.strip()
    if t.startswith("Assistant:"):
        t = t[len("Assistant:"):].strip()
    if t:
        t = t.splitlines()[0].strip()
    return t


@dataclass
class ConversationStateEvalResult:
    example: ConversationStateExample
    raw: str
    cleaned: str

    @property
    def correct(self) -> bool:
        return self.cleaned == self.example.answer_text


def _accuracy(results: list[ConversationStateEvalResult]) -> Optional[float]:
    if not results:
        return None
    return sum(1 for r in results if r.correct) / len(results)


def evaluate_conversation_state_model(
    predict_fn: PredictFn,
    examples: list[ConversationStateExample],
) -> dict[str, object]:
    results = []
    for ex in examples:
        raw = predict_fn(ex.prompt)
        results.append(ConversationStateEvalResult(example=ex, raw=raw, cleaned=_clean(raw)))

    by_num_topics: dict[int, Optional[float]] = {}
    for n in sorted({r.example.num_topics for r in results}):
        subset = [r for r in results if r.example.num_topics == n]
        by_num_topics[n] = _accuracy(subset)

    wrong = [r for r in results if not r.correct]
    shortcut_names = ("current_turn_only", "generic_answer", "last_noun_heuristic", "topic_word_persistence")
    shortcut_selection_rate: dict[str, Optional[float]] = {}
    for name in shortcut_names:
        applicable = [r for r in wrong if name in r.example.shortcut_predictions]
        if not applicable:
            shortcut_selection_rate[name] = None
            continue
        matched = sum(1 for r in applicable if r.cleaned == r.example.shortcut_predictions[name])
        shortcut_selection_rate[name] = matched / len(applicable)
    shortcut_explained_wrong_rate = (
        sum(1 for r in wrong if r.cleaned in r.example.shortcut_predictions.values())
        / len(wrong)
    ) if wrong else None

    return {
        "overall_accuracy": _accuracy(results),
        "accuracy_by_num_topics": by_num_topics,
        "shortcut_selection_rate": shortcut_selection_rate,
        "shortcut_explained_wrong_rate": shortcut_explained_wrong_rate,
        "num_examples": len(results),
    }
