"""Counterfactual Generator: verify that a model's answer is sensitive to
the *exact* facts given, not broad prompt similarity, by generating a
pair of near-identical examples where exactly one controlled fact changes
and the correct answer flips as a direct, deterministic consequence.

    Version A: Kevin has 8 cats. Maria has 4 parrots.
               Q: How many parrots does Kevin have?
               A: I don't have enough information to know how many
                  parrots Kevin has.
    Version B: Kevin has 8 cats. Kevin has 4 parrots.
               Q: How many parrots does Kevin have?
               A: Kevin has 4 parrots.

Only the *subject* of the second fact changed (Maria -> Kevin), but the
target flips entirely -- the design doc's own worked example. Per the
doc, this generator wraps another generator rather than building new
symbolic machinery of its own; it wraps the Binding generator here (its
Fact/World/Query shape is the cleanest single-axis perturbation target --
wrapping the others is a natural extension, not built in this pass).

Detection is fundamentally about *pair* consistency, not per-example
accuracy: a model that answers Version A correctly by luck (e.g. it just
always guesses UNKNOWN) but fails Version B has not learned the actual
subject+relation+noun binding this whole package exists to teach -- see
``evaluate_counterfactual_model``'s ``pair_consistency_rate`` and
``inconsistent_rate``.

Five perturbation axes, each mapped onto an existing Binding mechanism.
The design doc lists eight; three are deliberately not separate axes
here:

    ownership : HAVING_RELATIONS are already semantically interchangeable
                for query resolution (see query.py), so swapping among
                them can never flip an answer -- only a having-vs-
                observation swap is diagnostic, which is the `relation`
                axis below.
    negation  : a stated quantity of zero *is* this package's model of
                negation ("Kevin has no parrots."), so it collapses into
                the `quantity` axis.
    time      : temporal versioning is already State-Update's and
                Correction's whole trait; duplicating it here would just
                re-test the same mechanism under a different name.

    subject       : which name the second fact is about
    relation      : having vs. observation relation
    noun          : matching vs. non-matching noun
    quantity      : nonzero vs. explicit zero
    belief_status : directly asserted vs. reported (hearsay)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Literal, Union

from .query import UNKNOWN, Query, resolve_query
from .render import render_example
from .vocab import HAVING_RELATIONS, OBSERVE_RELATIONS, names_for, nouns_for
from .world import Fact, World

Answer = Union[int, Literal["UNKNOWN"]]
AxisBuilder = Callable[[random.Random, dict, tuple], tuple[World, World, Query]]


def _pair_subject(rng: random.Random, nouns: dict, names: tuple) -> tuple[World, World, Query]:
    subject, other_subject = rng.sample(names, k=2)
    target_noun, filler_noun = rng.sample(list(nouns.keys()), k=2)
    query = Query(subject, target_noun, False)

    filler = Fact(subject, rng.choice(HAVING_RELATIONS), filler_noun, rng.randint(1, 9))
    world_a = World([filler, Fact(other_subject, rng.choice(HAVING_RELATIONS), target_noun, rng.randint(1, 9))])
    world_b = World([filler, Fact(subject, rng.choice(HAVING_RELATIONS), target_noun, rng.randint(1, 9))])
    return world_a, world_b, query


def _pair_relation(rng: random.Random, nouns: dict, names: tuple) -> tuple[World, World, Query]:
    subject = rng.choice(names)
    target_noun = rng.choice(list(nouns.keys()))
    quantity = rng.randint(1, 9)
    query = Query(subject, target_noun, False)

    world_a = World([Fact(subject, rng.choice(OBSERVE_RELATIONS), target_noun, quantity)])
    world_b = World([Fact(subject, rng.choice(HAVING_RELATIONS), target_noun, quantity)])
    return world_a, world_b, query


def _pair_noun(rng: random.Random, nouns: dict, names: tuple) -> tuple[World, World, Query]:
    subject = rng.choice(names)
    target_noun, other_noun = rng.sample(list(nouns.keys()), k=2)
    quantity = rng.randint(1, 9)
    query = Query(subject, target_noun, False)

    world_a = World([Fact(subject, rng.choice(HAVING_RELATIONS), other_noun, quantity)])
    world_b = World([Fact(subject, rng.choice(HAVING_RELATIONS), target_noun, quantity)])
    return world_a, world_b, query


def _pair_quantity(rng: random.Random, nouns: dict, names: tuple) -> tuple[World, World, Query]:
    subject = rng.choice(names)
    target_noun = rng.choice(list(nouns.keys()))
    relation = rng.choice(HAVING_RELATIONS)
    quantity = rng.randint(1, 9)
    query = Query(subject, target_noun, False)

    world_a = World([Fact(subject, relation, target_noun, quantity)])
    world_b = World([Fact(subject, relation, target_noun, 0)])
    return world_a, world_b, query


def _pair_belief_status(rng: random.Random, nouns: dict, names: tuple) -> tuple[World, World, Query]:
    subject, speaker = rng.sample(names, k=2)
    target_noun = rng.choice(list(nouns.keys()))
    relation = rng.choice(HAVING_RELATIONS)
    quantity = rng.randint(1, 9)
    query = Query(subject, target_noun, False)

    world_a = World([Fact(subject, relation, target_noun, quantity, reported_by=speaker)])
    world_b = World([Fact(subject, relation, target_noun, quantity)])
    return world_a, world_b, query


AXIS_BUILDERS: dict[str, AxisBuilder] = {
    "subject": _pair_subject,
    "relation": _pair_relation,
    "noun": _pair_noun,
    "quantity": _pair_quantity,
    "belief_status": _pair_belief_status,
}
ALL_AXES: tuple[str, ...] = tuple(AXIS_BUILDERS.keys())


# -- Difficulty curriculum: one axis at a time, then a mixed level --------

COUNTERFACTUAL_DIFFICULTY_LEVELS: dict[int, tuple[str, ...]] = {
    1: ("subject",),
    2: ("relation",),
    3: ("noun",),
    4: ("quantity",),
    5: ("belief_status",),
    6: ALL_AXES,
}


def counterfactual_config_for_level(level: int) -> tuple[str, ...]:
    if level not in COUNTERFACTUAL_DIFFICULTY_LEVELS:
        raise ValueError(f"counterfactual difficulty level must be in 1..6, got {level}")
    return COUNTERFACTUAL_DIFFICULTY_LEVELS[level]


# -- Example assembly ---------------------------------------------------

@dataclass
class CounterfactualExample:
    axis: str
    difficulty: int
    prompt_a: str
    answer_text_a: str
    answer_value_a: Answer
    prompt_b: str
    answer_text_b: str
    answer_value_b: Answer


def generate_counterfactual_example(
    rng: random.Random, level: int, split: str = "train", style: str = "qa",
) -> CounterfactualExample:
    axes = counterfactual_config_for_level(level)
    axis = rng.choice(axes)
    nouns = nouns_for(split)
    names = names_for(split)

    world_a, world_b, query = AXIS_BUILDERS[axis](rng, nouns, names)
    answer_a = resolve_query(query, world_a, nouns)
    answer_b = resolve_query(query, world_b, nouns)

    prompt_a, answer_text_a = render_example(world_a.facts, query, answer_a, rng, split, nouns, style=style)
    prompt_b, answer_text_b = render_example(world_b.facts, query, answer_b, rng, split, nouns, style=style)

    return CounterfactualExample(
        axis=axis,
        difficulty=level,
        prompt_a=prompt_a, answer_text_a=answer_text_a, answer_value_a=answer_a,
        prompt_b=prompt_b, answer_text_b=answer_text_b, answer_value_b=answer_b,
    )
