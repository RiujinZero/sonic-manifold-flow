"""The ten-level difficulty curriculum.

Each level is a preset bundle of generation knobs, roughly in the order
described in the module spec: exact retrieval first, then missing-fact
detection, then progressively more/harder distractors, then category
generalization and aggregation, then the explicit-zero/reported-fact
distinctions that require the model to *not* manufacture an answer.
Levels are cumulative in spirit (each adds one new axis) but every knob
stays independently overridable, since real curricula usually want to mix
levels rather than jump monotonically through them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALL_DISTRACTOR_KINDS: tuple[str, ...] = (
    "subject_relation", "relation_object", "subject_object",
    "subject_only", "object_only", "category_confusable",
)


@dataclass(frozen=True)
class DifficultyConfig:
    level: int
    min_distractors: int = 0
    max_distractors: int = 0
    distractor_kinds: tuple[str, ...] = ()
    allow_category_questions: bool = False
    allow_aggregation: bool = False
    allow_zero_facts: bool = False
    allow_reported_facts: bool = False
    multi_entity: bool = False
    min_filler_facts: int = 0
    max_filler_facts: int = 1
    # Probability the generated example is answerable. Balanced near 0.5
    # throughout so "always answer UNKNOWN" (or the reverse) is never a
    # viable shortcut, per the spec's negative-example-batch requirement.
    answerable_probability: float = 0.5

    @property
    def allow_distractors(self) -> bool:
        return self.max_distractors > 0


DIFFICULTY_LEVELS: dict[int, DifficultyConfig] = {
    1: DifficultyConfig(
        level=1, answerable_probability=1.0,
    ),
    2: DifficultyConfig(
        level=2, answerable_probability=0.5,
    ),
    3: DifficultyConfig(
        level=3, min_distractors=1, max_distractors=1,
        distractor_kinds=("subject_relation", "relation_object", "subject_object"),
        min_filler_facts=0, max_filler_facts=2,
    ),
    4: DifficultyConfig(
        level=4, min_distractors=2, max_distractors=4,
        distractor_kinds=ALL_DISTRACTOR_KINDS[:5],
        min_filler_facts=1, max_filler_facts=3,
    ),
    5: DifficultyConfig(
        level=5, min_distractors=1, max_distractors=3,
        distractor_kinds=ALL_DISTRACTOR_KINDS[:5],
        allow_category_questions=True,
        min_filler_facts=1, max_filler_facts=3,
    ),
    6: DifficultyConfig(
        level=6, min_distractors=1, max_distractors=3,
        distractor_kinds=ALL_DISTRACTOR_KINDS[:5],
        allow_category_questions=True, allow_aggregation=True,
        min_filler_facts=1, max_filler_facts=3,
    ),
    7: DifficultyConfig(
        level=7, min_distractors=1, max_distractors=3,
        distractor_kinds=ALL_DISTRACTOR_KINDS[:5],
        allow_category_questions=True, allow_aggregation=True,
        allow_zero_facts=True,
        min_filler_facts=1, max_filler_facts=3,
    ),
    8: DifficultyConfig(
        level=8, min_distractors=2, max_distractors=4,
        distractor_kinds=ALL_DISTRACTOR_KINDS,
        allow_category_questions=True, allow_aggregation=True,
        allow_zero_facts=True,
        min_filler_facts=1, max_filler_facts=4,
    ),
    9: DifficultyConfig(
        level=9, min_distractors=2, max_distractors=4,
        distractor_kinds=ALL_DISTRACTOR_KINDS,
        allow_category_questions=True, allow_aggregation=True,
        allow_zero_facts=True, allow_reported_facts=True,
        min_filler_facts=1, max_filler_facts=4,
    ),
    10: DifficultyConfig(
        level=10, min_distractors=2, max_distractors=6,
        distractor_kinds=ALL_DISTRACTOR_KINDS,
        allow_category_questions=True, allow_aggregation=True,
        allow_zero_facts=True, allow_reported_facts=True,
        multi_entity=True,
        min_filler_facts=2, max_filler_facts=6,
    ),
}


def config_for_level(level: int) -> DifficultyConfig:
    if level not in DIFFICULTY_LEVELS:
        raise ValueError(f"difficulty level must be in 1..10, got {level}")
    return DIFFICULTY_LEVELS[level]
