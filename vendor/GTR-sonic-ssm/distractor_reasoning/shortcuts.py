"""Naive partial-match / aggregation shortcuts a model could exploit
instead of real subject+relation+object binding, plus the value each one
would predict for a given example -- so evaluation can tell *which*
shortcut a wrong answer matches, not just that it's wrong.

Each shortcut scans every fact in the world (direct, distractor, filler,
and reported alike -- a model naively pattern-matching on surface text
can't tell those apart) for the first fact matching a *partial* subset of
(subject, relation-is-having, noun), ignoring one or more of those slots
entirely. ``subject_aggregate`` additionally sums every matching fact
instead of taking the first, modeling a "add up anything that looks
associated with the subject" cheat.

Facts are scanned in construction order (direct facts first, then
distractors, filler, and any reported fact -- see ``generate.py``), so on
answerable examples a shortcut can legitimately land on the same value as
the correct answer; that's expected, not a bug -- it just means this
particular shortcut doesn't separate from correct behavior on that
example. The diagnostic value is concentrated on UNKNOWN/wrong cases,
exactly as in the worked example this module implements:

    Kevin has 20 cats. Maria has 7 parrots. Kevin photographed 9 parrots.
    Q: How many parrots does Kevin have?  ->  UNKNOWN (no direct match)
        subject_relation: 20   (ignores noun -- Kevin's only having-fact)
        subject_object:    9   (ignores relation -- Kevin+parrots)
        relation_object:   7   (ignores subject -- Maria's having+parrots)
        subject_aggregate: 29  (20 + 9, everything Kevin-associated)

This is internal bookkeeping only, exactly like ``distractor_kinds`` --
``Example.shortcut_predictions`` must never be rendered into prompt/answer
text (nothing in ``render.py`` touches it).
"""

from __future__ import annotations

from typing import Optional

from .query import Query
from .vocab import Noun
from .world import Fact, World


def _noun_matches(fact: Fact, query: Query, nouns: dict[str, Noun]) -> bool:
    if query.is_category:
        info = nouns.get(fact.noun)
        return info is not None and query.target in info.categories
    return fact.noun == query.target


def _first_quantity(facts: list[Fact]) -> Optional[int]:
    return facts[0].quantity if facts else None


def subject_relation(world: World, query: Query, nouns: dict[str, Noun]) -> Optional[int]:
    """Ignores the noun: any having-relation fact for this subject."""

    matches = [f for f in world.facts if f.subject == query.subject and f.is_having]
    return _first_quantity(matches)


def relation_object(world: World, query: Query, nouns: dict[str, Noun]) -> Optional[int]:
    """Ignores the subject: any having-relation fact for this noun."""

    matches = [f for f in world.facts if f.is_having and _noun_matches(f, query, nouns)]
    return _first_quantity(matches)


def subject_object(world: World, query: Query, nouns: dict[str, Noun]) -> Optional[int]:
    """Ignores the relation: any fact (having or not) for this subject+noun."""

    matches = [f for f in world.facts if f.subject == query.subject and _noun_matches(f, query, nouns)]
    return _first_quantity(matches)


def subject_only(world: World, query: Query, nouns: dict[str, Noun]) -> Optional[int]:
    """Ignores relation and noun: any fact at all for this subject."""

    matches = [f for f in world.facts if f.subject == query.subject]
    return _first_quantity(matches)


def object_only(world: World, query: Query, nouns: dict[str, Noun]) -> Optional[int]:
    """Ignores subject and relation: any fact at all for this noun."""

    matches = [f for f in world.facts if _noun_matches(f, query, nouns)]
    return _first_quantity(matches)


def subject_aggregate(world: World, query: Query, nouns: dict[str, Noun]) -> Optional[int]:
    """Sums every fact for this subject, regardless of relation or noun."""

    matches = [f for f in world.facts if f.subject == query.subject]
    if not matches:
        return None
    return sum(f.quantity for f in matches)


SHORTCUTS = {
    "subject_relation": subject_relation,
    "relation_object": relation_object,
    "subject_object": subject_object,
    "subject_only": subject_only,
    "object_only": object_only,
    "subject_aggregate": subject_aggregate,
}

SHORTCUT_NAMES: tuple[str, ...] = tuple(SHORTCUTS.keys())


def shortcut_predictions(world: World, query: Query, nouns: dict[str, Noun]) -> dict[str, int]:
    """The value each naive shortcut would predict for this example,
    omitting shortcuts with no candidate fact to guess from."""

    predictions: dict[str, int] = {}
    for name, fn in SHORTCUTS.items():
        value = fn(world, query, nouns)
        if value is not None:
            predictions[name] = value
    return predictions
