"""Acceptance metrics for the recorded tabletop benchmark."""

from __future__ import annotations

import re
from dataclasses import dataclass

from manifoldflow.world_types import ZONES

# Natural-language phrasing ("upper left", not "upper_left") for the
# closed-world candidate sentences and the regex below; ZONES itself stays
# the underscore-joined form WorldStateEncoder/WorldRecord.zone use.
_ZONE_PHRASES = tuple(zone.replace("_", " ") for zone in ZONES)

WORLD_FACT_CANDIDATES = (
    *(f"The car is visible in the {phrase} zone." for phrase in _ZONE_PHRASES),
    "The car is hidden behind the box.",
    "The car's current location is unknown.",
    "Yes.",
    "No.",
    *(f"The ball is visible in the {phrase} zone." for phrase in _ZONE_PHRASES),
    "The ball is under the red cup.",
    "The ball is under the blue cup.",
    "The ball is under the yellow cup.",
    "There is not enough evidence to identify a cup.",
    "There is not enough evidence to determine that.",
    *(f"The red cup is in the {phrase} zone." for phrase in _ZONE_PHRASES),
    *(f"The blue cup is in the {phrase} zone." for phrase in _ZONE_PHRASES),
    *(f"The yellow cup is in the {phrase} zone." for phrase in _ZONE_PHRASES),
)


def world_fact_match(expected: str, prediction: str) -> bool:
    """Score the grounded fact while allowing harmless answer paraphrases."""
    expected_text = _normalize(expected)
    prediction_text = _normalize(prediction)
    if not prediction_text:
        return False
    if "not enough evidence" in expected_text:
        return any(
            phrase in prediction_text
            for phrase in (
                "not enough evidence",
                "no evidence",
                "cannot determine",
                "can't determine",
                "unable to determine",
            )
        )
    if expected_text in {"yes", "no"}:
        first_word = prediction_text.split(maxsplit=1)[0]
        return first_word == expected_text
    color = re.search(r"under the (red|blue|yellow) cup", expected_text)
    if color:
        return color.group(1) in prediction_text and "cup" in prediction_text
    zone = re.search(
        rf"(?:in|into) the ({'|'.join(_ZONE_PHRASES)}) zone", expected_text
    )
    if zone:
        return zone.group(1) in prediction_text
    if "box" in expected_text and "car" in expected_text:
        return "box" in prediction_text and "car" in prediction_text
    return expected_text in prediction_text


def answer_candidates_for_question(
    question: str, candidates: tuple[str, ...]
) -> tuple[str, ...]:
    """Restrict closed-world decoding by question intent, never by its answer."""
    text = _normalize(question)
    if any(phrase in text for phrase in ("visible", "in view", "camera")):
        selected = [value for value in candidates if _normalize(value) in {"yes", "no"}]
    elif any(
        phrase in text
        for phrase in ("contains the ball", "has the ball", "colored cup")
    ):
        selected = [
            value
            for value in candidates
            if "ball is under" in _normalize(value)
            or "identify a cup" in _normalize(value)
        ]
    elif any(phrase in text for phrase in ("hiding", "occludes", "concealed")):
        selected = [
            value
            for value in candidates
            if "hidden behind" in _normalize(value)
            or "determine that" in _normalize(value)
        ]
    elif "ball" in text:
        # Mirror the "car" branch below: a plain "where is the ball" question
        # cannot assume the ball is currently visible, so the candidate set
        # must include containment and evidence-insufficient answers too, not
        # just the visible-zone facts. Restricting to "visible" candidates
        # only made a confident wrong zone the only possible output whenever
        # the ball was actually contained or its location was unknown.
        selected = [
            value
            for value in candidates
            if _normalize(value).startswith("the ball")
            or "identify a cup" in _normalize(value)
        ]
    elif "car" in text:
        selected = [
            value for value in candidates if _normalize(value).startswith("the car")
        ]
    else:
        color = next((item for item in ("red", "blue", "yellow") if item in text), None)
        selected = [
            value
            for value in candidates
            if color is not None and _normalize(value).startswith(f"the {color} cup is")
        ]
    return tuple(selected) if selected else candidates


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", value.lower()).split())


@dataclass(frozen=True)
class WorldEvaluation:
    detector_map50: float
    visible_identity_accuracy: float
    reidentification_accuracy: float
    car_answer_accuracy: float
    cup_answer_accuracy: float
    unknown_precision: float
    retrieval_recall_at_4: float
    memory_answer_accuracy: float
    shuffled_memory_drop: float
    stale_override_accuracy: float

    def acceptance_failures(self) -> list[str]:
        thresholds = {
            "detector_map50": (self.detector_map50, 0.90),
            "visible_identity_accuracy": (self.visible_identity_accuracy, 0.95),
            "reidentification_accuracy": (self.reidentification_accuracy, 0.90),
            "car_answer_accuracy": (self.car_answer_accuracy, 0.90),
            "cup_answer_accuracy": (self.cup_answer_accuracy, 0.85),
            "unknown_precision": (self.unknown_precision, 0.90),
            "retrieval_recall_at_4": (self.retrieval_recall_at_4, 0.95),
            "memory_answer_accuracy": (self.memory_answer_accuracy, 0.85),
            "shuffled_memory_drop": (self.shuffled_memory_drop, 0.40),
            "stale_override_accuracy": (self.stale_override_accuracy, 0.90),
        }
        return [
            name for name, (value, threshold) in thresholds.items() if value < threshold
        ]
