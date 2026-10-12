"""The symbolic world: facts as structured (subject, relation, object,
quantity) tuples, generated and reasoned over *before* any natural-language
text is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .vocab import HAVING_RELATIONS


@dataclass(frozen=True)
class Fact:
    """One (subject, relation, noun, quantity) binding.

    ``reported_by`` marks the fact as hearsay (e.g. "Lisa says Kevin owns
    4 rabbits") rather than a directly asserted fact -- reported facts are
    excluded from ordinary query resolution unless a question explicitly
    asks what was reported, matching the spec's requirement that the model
    not manufacture facts from unreliable attribution.
    """

    subject: str
    relation: str
    noun: str
    quantity: int
    reported_by: Optional[str] = None

    @property
    def is_having(self) -> bool:
        return self.relation in HAVING_RELATIONS

    @property
    def is_reported(self) -> bool:
        return self.reported_by is not None


@dataclass
class World:
    """An unordered collection of facts. Order is only decided at render
    time (``render.py``), keeping the symbolic world itself order-free."""

    facts: list[Fact]

    def direct_having_facts(self) -> list[Fact]:
        """Facts that are directly asserted (not hearsay) and establish
        possession -- the only facts ordinary queries resolve against."""

        return [f for f in self.facts if f.is_having and not f.is_reported]
