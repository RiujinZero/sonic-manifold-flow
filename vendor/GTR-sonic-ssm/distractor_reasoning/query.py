"""Query representation and the single rule that resolves any query
against a ``World``: sum every directly-asserted having-fact that matches
the subject and (noun or category); if nothing matches, the answer is
UNKNOWN.

That one rule uniformly covers exact retrieval (a single matching fact),
explicit zero (a matching fact whose quantity is literally 0), aggregation
across multiple facts or an entire category (several matching facts
summed), and true unanswerability (no matching facts at all) -- there is
deliberately no special-cased branch for any of those, since they are all
the same operation at different fact counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from .vocab import Noun
from .world import Fact, World

UNKNOWN: Literal["UNKNOWN"] = "UNKNOWN"


@dataclass(frozen=True)
class Query:
    subject: str
    target: str  # a noun key (e.g. "parrot") or a category name (e.g. "bird")
    is_category: bool
    # True for the "actually/really have" phrasing used to disambiguate
    # from a co-present reported (hearsay) fact about the same subject.
    disambiguates_reported: bool = False

    def matches(self, fact: Fact, nouns: dict[str, Noun]) -> bool:
        if fact.subject != self.subject:
            return False
        if self.is_category:
            noun_info = nouns.get(fact.noun)
            return noun_info is not None and self.target in noun_info.categories
        return fact.noun == self.target


def resolve_query(
    query: Query, world: World, nouns: dict[str, Noun]
) -> Union[int, Literal["UNKNOWN"]]:
    """Return the gold answer: a non-negative int, or the literal string
    ``"UNKNOWN"`` when no directly-asserted fact resolves the query."""

    matching = [f for f in world.direct_having_facts() if query.matches(f, nouns)]
    if not matching:
        return UNKNOWN
    return sum(f.quantity for f in matching)
