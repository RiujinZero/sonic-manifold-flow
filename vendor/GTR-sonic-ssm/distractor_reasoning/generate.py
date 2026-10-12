"""Top-level example generator: assembles a symbolic world and query per a
difficulty level, resolves the gold answer, and renders to text.

This is the only place that decides what goes into an example; everything
downstream (``render.py``) just expresses that decision as text, and the
internal metadata fields on ``Example`` (distractor kinds, fact counts,
zero/unknown/aggregation flags) exist purely for evaluation bucketing --
``render_example`` never sees them and they never appear in the rendered
text.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional, Union

from .difficulty import DifficultyConfig, config_for_level
from .distractors import make_distractor
from .query import UNKNOWN, Query, resolve_query
from .render import render_example
from .shortcuts import shortcut_predictions as _shortcut_predictions
from .vocab import (
    CATEGORY_PLURALS,
    HAVING_RELATIONS,
    names_for,
    nouns_for,
)
from .world import Fact, World


@dataclass
class Example:
    prompt: str
    answer_text: str
    answer_value: Union[int, Literal["UNKNOWN"]]
    difficulty: int
    num_facts: int
    num_distractors: int
    distractor_kinds: tuple[str, ...]
    is_category_question: bool
    is_zero_case: bool
    is_unknown_case: bool
    is_aggregation: bool
    has_reported_fact: bool
    multi_entity: bool
    world: World
    query: Query
    # Internal-only, like distractor_kinds: the value each naive
    # partial-match/aggregation shortcut would predict for this example.
    # Never rendered into prompt/answer text -- see shortcuts.py.
    shortcut_predictions: dict[str, int]


def _pick_target(
    config: DifficultyConfig, nouns: dict, rng: random.Random,
) -> tuple[str, bool]:
    if config.allow_category_questions and rng.random() < 0.35:
        return rng.choice(list(CATEGORY_PLURALS.keys())), True
    return rng.choice(list(nouns.keys())), False


def _member_noun(target: str, is_category: bool, nouns: dict, rng: random.Random) -> str:
    if not is_category:
        return target
    members = [n for n, info in nouns.items() if target in info.categories]
    return rng.choice(members) if members else rng.choice(list(nouns))


def _build_direct_facts(
    subject: str, target: str, is_category: bool, outcome: str,
    config: DifficultyConfig, nouns: dict, rng: random.Random,
) -> list[Fact]:
    """``outcome`` is one of "nonzero", "zero", "unknown"."""

    if outcome == "unknown":
        return []
    if outcome == "zero":
        noun = _member_noun(target, is_category, nouns, rng)
        return [Fact(subject, rng.choice(HAVING_RELATIONS), noun, 0)]
    # nonzero
    if is_category and config.allow_aggregation and rng.random() < 0.6:
        members = [n for n, info in nouns.items() if target in info.categories]
        count = min(len(members), rng.randint(2, 3))
        chosen = rng.sample(members, k=count) if count > 0 else []
        if not chosen:
            chosen = [_member_noun(target, is_category, nouns, rng)]
        return [
            Fact(subject, rng.choice(HAVING_RELATIONS), n, rng.randint(1, 9))
            for n in chosen
        ]
    noun = _member_noun(target, is_category, nouns, rng)
    return [Fact(subject, rng.choice(HAVING_RELATIONS), noun, rng.randint(1, 9))]


def _build_distractors(
    subject: str, target: str, is_category: bool, config: DifficultyConfig,
    nouns: dict, names: tuple[str, ...], rng: random.Random,
) -> tuple[list[Fact], tuple[str, ...]]:
    if not config.allow_distractors:
        return [], ()
    count = rng.randint(config.min_distractors, config.max_distractors)
    kinds_used: list[str] = []
    facts: list[Fact] = []
    available_kinds = list(config.distractor_kinds)
    for _ in range(count):
        if not available_kinds:
            break
        kind = rng.choice(available_kinds)
        fact = make_distractor(kind, subject, target, is_category, rng, nouns, names)
        if fact is None:
            continue
        facts.append(fact)
        kinds_used.append(kind)
    return facts, tuple(kinds_used)


def _build_filler_facts(
    subject: str, config: DifficultyConfig, names: tuple[str, ...],
    nouns: dict, rng: random.Random,
) -> list[Fact]:
    n = rng.randint(config.min_filler_facts, config.max_filler_facts)
    other_names = [name for name in names if name != subject]
    if not other_names or n == 0:
        return []
    pool_size = len(other_names) if config.multi_entity else min(2, len(other_names))
    fillers = []
    for _ in range(n):
        other_subject = rng.choice(other_names[:pool_size] or other_names)
        relation = rng.choice(HAVING_RELATIONS + ("photographed", "counted", "saw", "spotted"))
        noun = rng.choice(list(nouns.keys()))
        fillers.append(Fact(other_subject, relation, noun, rng.randint(1, 12)))
    return fillers


def _maybe_reported_fact(
    subject: str, target: str, is_category: bool, config: DifficultyConfig,
    nouns: dict, names: tuple[str, ...], rng: random.Random,
) -> tuple[Optional[Fact], bool]:
    """Returns (fact_or_None, disambiguates) -- disambiguates is True when
    the reported fact targets the query's own subject+noun/category, which
    should make the question use the "actually have" phrasing."""

    if not config.allow_reported_facts or rng.random() >= 0.5:
        return None, False
    speakers = [n for n in names if n != subject]
    if not speakers:
        return None, False
    speaker = rng.choice(speakers)
    deceptive = rng.random() < 0.5
    if deceptive:
        noun = _member_noun(target, is_category, nouns, rng)
    else:
        noun = rng.choice(list(nouns.keys()))
    quantity = rng.randint(1, 12)
    fact = Fact(subject, rng.choice(HAVING_RELATIONS), noun, quantity, reported_by=speaker)
    return fact, deceptive


def generate_example(
    rng: random.Random,
    level: int,
    split: str = "train",
    style: str = "qa",
) -> Example:
    config = config_for_level(level)
    nouns = nouns_for(split)
    names = names_for(split)

    subject = rng.choice(names)
    target, is_category = _pick_target(config, nouns, rng)

    if rng.random() < config.answerable_probability:
        outcome = "zero" if (config.allow_zero_facts and rng.random() < 0.25) else "nonzero"
    else:
        outcome = "unknown"

    direct_facts = _build_direct_facts(subject, target, is_category, outcome, config, nouns, rng)
    distractor_facts, distractor_kinds = _build_distractors(
        subject, target, is_category, config, nouns, names, rng
    )
    filler_facts = _build_filler_facts(subject, config, names, nouns, rng)
    reported_fact, disambiguates = _maybe_reported_fact(
        subject, target, is_category, config, nouns, names, rng
    )

    all_facts = [*direct_facts, *distractor_facts, *filler_facts]
    if reported_fact is not None:
        all_facts.append(reported_fact)

    query = Query(subject=subject, target=target, is_category=is_category,
                  disambiguates_reported=disambiguates)
    world = World(all_facts)
    answer = resolve_query(query, world, nouns)

    prompt, answer_text = render_example(all_facts, query, answer, rng, split, nouns, style=style)

    return Example(
        prompt=prompt,
        answer_text=answer_text,
        answer_value=answer,
        difficulty=level,
        num_facts=len(all_facts),
        num_distractors=len(distractor_facts),
        distractor_kinds=distractor_kinds,
        is_category_question=is_category,
        is_zero_case=(answer == 0),
        is_unknown_case=(answer == UNKNOWN),
        is_aggregation=(len(direct_facts) > 1),
        has_reported_fact=reported_fact is not None,
        multi_entity=config.multi_entity,
        world=world,
        query=query,
        shortcut_predictions=_shortcut_predictions(world, query, nouns),
    )
