"""Naive shortcuts a model could exploit on the Ontology task instead of
genuinely following the supplied taxonomy, mirroring ``shortcuts.py``'s
role for the Binding generator.

    lexical_matching:      only counts facts whose text literally contains
                            the queried category word -- since facts are
                            always phrased in terms of a *leaf* noun, this
                            never finds anything for a category query and
                            always predicts UNKNOWN, correct answer or not.
    subject_aggregate:      sums every number associated with the subject,
                            ignoring the taxonomy entirely.
    over_generalize_category: matches one level broader than the query
                            actually asks for (siblings under the same
                            parent for a leaf query; the whole grandparent
                            branch for a parent query) -- the "everything
                            in the neighborhood counts" failure mode.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from .ontology import Taxonomy
from .query import UNKNOWN, Query
from .vocab import Noun
from .world import World

Prediction = Union[int, Literal["UNKNOWN"]]


def lexical_matching(
    world: World, query: Query, taxonomy: Taxonomy, nouns: dict[str, Noun],
) -> Optional[Prediction]:
    if not query.is_category:
        return None
    return UNKNOWN


def subject_aggregate(
    world: World, query: Query, taxonomy: Taxonomy, nouns: dict[str, Noun],
) -> Optional[Prediction]:
    matches = [f.quantity for f in world.facts if f.subject == query.subject]
    if not matches:
        return None
    return sum(matches)


def _broader_match_set(query: Query, taxonomy: Taxonomy) -> Optional[set[str]]:
    target = query.target
    if target in taxonomy.grandparents:
        return None  # already the broadest level in a 2-level taxonomy
    if target in taxonomy.parents:
        grandparent = taxonomy.parent_to_grandparent.get(target)
        if grandparent is None:
            return None  # cross-cutting parent has no further ancestor
        return {leaf for leaf in taxonomy.leaves if grandparent in taxonomy.ancestors_of(leaf)}
    # leaf-level query: broaden to siblings sharing any of its parents
    parents = taxonomy.leaf_to_parents.get(target, ())
    siblings: set[str] = set()
    for parent in parents:
        siblings |= {leaf for leaf in taxonomy.leaves if parent in taxonomy.leaf_to_parents[leaf]}
    return siblings or None


def over_generalize_category(
    world: World, query: Query, taxonomy: Taxonomy, nouns: dict[str, Noun],
) -> Optional[Prediction]:
    broader = _broader_match_set(query, taxonomy)
    if not broader:
        return None
    matches = [
        f.quantity for f in world.facts
        if f.subject == query.subject and f.is_having and f.noun in broader
    ]
    if not matches:
        return None
    return sum(matches)


ONTOLOGY_SHORTCUTS = {
    "lexical_matching": lexical_matching,
    "subject_aggregate": subject_aggregate,
    "over_generalize_category": over_generalize_category,
}

ONTOLOGY_SHORTCUT_NAMES: tuple[str, ...] = tuple(ONTOLOGY_SHORTCUTS.keys())


def ontology_shortcut_predictions(
    world: World, query: Query, taxonomy: Taxonomy, nouns: dict[str, Noun],
) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    for name, fn in ONTOLOGY_SHORTCUTS.items():
        value = fn(world, query, taxonomy, nouns)
        if value is not None:
            predictions[name] = value
    return predictions
