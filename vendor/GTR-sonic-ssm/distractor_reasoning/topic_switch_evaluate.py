"""Held-out evaluation for the Topic-Switch generator.

Mirrors the design doc's own "Detection" section directly: switch
correctness, contamination rate, and resume accuracy are evaluated as
separate rates across probed turns, not folded into one per-example
accuracy number the way every other generator in this package works --
Topic-Switch's trait is fundamentally about behavior across a sequence of
turns, not a single answer.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .topic_switch import TopicSwitchExample, generate_topic_switch_example

PredictFn = Callable[[str], str]


def generate_topic_switch_eval_examples(
    n: int,
    *,
    seed: int = 12_345,
    level: Optional[int] = None,
    level_range: tuple[int, int] = (1, 3),
) -> list[TopicSwitchExample]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        chosen_level = level if level is not None else rng.randint(*level_range)
        examples.append(generate_topic_switch_example(rng, chosen_level, split="eval"))
    return examples


def _clean(text: str) -> str:
    t = text.strip()
    for prefix in ("Answer:", "A:", "Assistant:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
            break
    if t:
        t = t.splitlines()[0].strip()
    return t


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


@dataclass
class ProbeResult:
    example: TopicSwitchExample
    label: str
    raw: str
    cleaned: str
    gold_text: str
    shortcut_predictions: dict[str, str]

    @property
    def correct(self) -> bool:
        return self.cleaned == self.gold_text


def evaluate_topic_switch_model(
    predict_fn: PredictFn,
    examples: list[TopicSwitchExample],
) -> dict[str, object]:
    results: list[ProbeResult] = []
    for ex in examples:
        for probe in ex.probes:
            raw = predict_fn(probe.prompt)
            results.append(ProbeResult(
                example=ex, label=probe.label, raw=raw, cleaned=_clean(raw),
                gold_text=probe.gold_text, shortcut_predictions=probe.shortcut_predictions,
            ))

    topic_b_results = [r for r in results if r.label == "topic_b"]
    resume_results = [r for r in results if r.label == "resume"]

    def _rate(rs: list[ProbeResult]) -> Optional[float]:
        if not rs:
            return None
        return sum(1 for r in rs if r.correct) / len(rs)

    contaminated = sum(
        1 for r in topic_b_results if _contains_word(r.cleaned, r.example.task_a_subject)
    )

    def _shortcut_rate(rs: list[ProbeResult], name: str) -> Optional[float]:
        wrong = [r for r in rs if not r.correct]
        applicable = [r for r in wrong if name in r.shortcut_predictions]
        if not applicable:
            return None
        matched = sum(1 for r in applicable if r.cleaned == r.shortcut_predictions[name])
        return matched / len(applicable)

    return {
        "switch_correctness": _rate(topic_b_results),
        "resume_accuracy": _rate(resume_results),
        "contamination_rate": (contaminated / len(topic_b_results)) if topic_b_results else None,
        "previous_task_persistence_rate": _shortcut_rate(topic_b_results, "previous_task_persistence"),
        "context_reset_rate": _shortcut_rate(resume_results, "context_reset"),
        "num_examples": len(examples),
        "num_topic_b_probes": len(topic_b_results),
        "num_resume_probes": len(resume_results),
    }
