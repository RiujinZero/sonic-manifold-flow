"""Labeled hard-negative distractor generators.

Each function builds one ``Fact`` that shares exactly the surface features
its name describes with the query -- and is *guaranteed*, by construction,
never to satisfy ``query.matches`` -- so a model that binds subject +
relation + object correctly can never be fooled by it, while a model that
pattern-matches on partial overlap can. The kind label is returned
alongside the fact for internal bookkeeping/evaluation only; it must never
leak into rendered text (``render.py`` never sees it).

    subject_relation : same subject, a having-relation, a different noun
                        ("Kevin has 6 cats" when asked about parrots)
    relation_object   : a having-relation, the same noun, a different subject
                        ("Maria has 8 parrots")
    subject_object    : same subject, same noun, a non-having relation
                        ("Kevin photographed 5 parrots")
    subject_only      : same subject, a having-relation, an unrelated noun
                        from a disjoint category branch ("Kevin owns 3 bicycles")
    object_only       : a different subject, a non-having relation, the same noun
                        ("Lisa counted 11 parrots")
    category_confusable : a having-relation fact whose noun sits in a
                        *different* category branch than the queried
                        category, testing that category aggregation
                        doesn't over-generalize (only meaningful for
                        category queries)
"""

from __future__ import annotations

import random
from typing import Optional

from .vocab import HAVING_RELATIONS, Noun, OBSERVE_RELATIONS
from .world import Fact

DEFAULT_QUANTITY_RANGE = (1, 12)


def _excluded_nouns(target: str, is_category: bool, nouns: dict[str, Noun]) -> set[str]:
    """Noun keys that would accidentally satisfy the query if used."""

    if is_category:
        return {n for n, info in nouns.items() if target in info.categories}
    return {target}


def _quantity(rng: random.Random) -> int:
    return rng.randint(*DEFAULT_QUANTITY_RANGE)


def _member_of_category(target: str, nouns: dict[str, Noun], rng: random.Random) -> str:
    members = [n for n, info in nouns.items() if target in info.categories]
    return rng.choice(members) if members else rng.choice(list(nouns))


def make_subject_relation(
    subject: str, target: str, is_category: bool, rng: random.Random,
    nouns: dict[str, Noun], names: tuple[str, ...],
) -> Fact:
    excluded = _excluded_nouns(target, is_category, nouns)
    candidates = [n for n in nouns if n not in excluded]
    other_noun = rng.choice(candidates)
    return Fact(subject, rng.choice(HAVING_RELATIONS), other_noun, _quantity(rng))


def make_relation_object(
    subject: str, target: str, is_category: bool, rng: random.Random,
    nouns: dict[str, Noun], names: tuple[str, ...],
) -> Fact:
    other_subject = rng.choice([n for n in names if n != subject])
    noun = target if not is_category else _member_of_category(target, nouns, rng)
    return Fact(other_subject, rng.choice(HAVING_RELATIONS), noun, _quantity(rng))


def make_subject_object(
    subject: str, target: str, is_category: bool, rng: random.Random,
    nouns: dict[str, Noun], names: tuple[str, ...],
) -> Fact:
    # Non-having relation, so re-using the exact target noun is always
    # safe -- it can never satisfy the query regardless of noun identity.
    noun = target if not is_category else _member_of_category(target, nouns, rng)
    return Fact(subject, rng.choice(OBSERVE_RELATIONS), noun, _quantity(rng))


def make_subject_only(
    subject: str, target: str, is_category: bool, rng: random.Random,
    nouns: dict[str, Noun], names: tuple[str, ...],
) -> Fact:
    target_categories = (
        {target} if is_category else set(nouns[target].categories)
    )
    candidates = [
        n for n, info in nouns.items()
        if not (set(info.categories) & target_categories) and n != target
    ]
    if not candidates:
        candidates = [n for n in nouns if n != target]
    other_noun = rng.choice(candidates)
    return Fact(subject, rng.choice(HAVING_RELATIONS), other_noun, _quantity(rng))


def make_object_only(
    subject: str, target: str, is_category: bool, rng: random.Random,
    nouns: dict[str, Noun], names: tuple[str, ...],
) -> Fact:
    other_subject = rng.choice([n for n in names if n != subject])
    noun = target if not is_category else _member_of_category(target, nouns, rng)
    return Fact(other_subject, rng.choice(OBSERVE_RELATIONS), noun, _quantity(rng))


def make_category_confusable(
    subject: str, target: str, is_category: bool, rng: random.Random,
    nouns: dict[str, Noun], names: tuple[str, ...],
) -> Optional[Fact]:
    if not is_category:
        return None
    candidates = [n for n, info in nouns.items() if target not in info.categories]
    if not candidates:
        return None
    other_noun = rng.choice(candidates)
    other_subject = rng.choice(names) if rng.random() < 0.5 else subject
    return Fact(other_subject, rng.choice(HAVING_RELATIONS), other_noun, _quantity(rng))


DISTRACTOR_MAKERS = {
    "subject_relation": make_subject_relation,
    "relation_object": make_relation_object,
    "subject_object": make_subject_object,
    "subject_only": make_subject_only,
    "object_only": make_object_only,
    "category_confusable": make_category_confusable,
}


def make_distractor(
    kind: str, subject: str, target: str, is_category: bool,
    rng: random.Random, nouns: dict[str, Noun], names: tuple[str, ...],
) -> Optional[Fact]:
    return DISTRACTOR_MAKERS[kind](subject, target, is_category, rng, nouns, names)
