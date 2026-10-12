"""Naive shortcuts a model could exploit on the Instruction-Selection task
instead of genuinely reading which operation is being requested, mirroring
``shortcuts.py``'s role for the Binding generator.

Applicability is operation-dependent -- a shortcut that "collapses to
retrieve" is meaningless when the real operation already IS retrieve --
so each predictor returns ``None`` outside the operation(s) it's
diagnostic for, the same convention ``ontology_shortcuts.py``'s
``lexical_matching`` uses.
"""

from __future__ import annotations

from typing import Optional

from .instruction_selection import InstructionAnswer, Scenario
from .query import Query, resolve_query


def _retrieve_value(scenario: Scenario) -> InstructionAnswer:
    return resolve_query(Query(scenario.subjects[0], scenario.primary_noun, False), scenario.world, scenario.nouns)


def _identify_holder_value(scenario: Scenario) -> tuple[str, ...]:
    return tuple(sorted({
        f.subject for f in scenario.world.direct_having_facts() if f.noun == scenario.secondary_noun
    }))


def _classify_observed_value(scenario: Scenario) -> tuple[str, ...]:
    subject = scenario.subjects[0]
    return tuple(sorted({
        f.noun for f in scenario.world.facts
        if f.subject == subject and f.relation == scenario.observe_relation and not f.is_having
    }))


def word_problem_reflex(operation: str, scenario: Scenario, instruction: str) -> Optional[InstructionAnswer]:
    """"Sees numbers, performs arithmetic": sums every quantity visible in
    the world regardless of what was actually asked. Only diagnostic for
    operations whose correct answer isn't already a plain number."""

    if operation not in ("identify_holder", "classify_observed", "repeat", "reject_unsupported"):
        return None
    return sum(f.quantity for f in scenario.world.facts)


def dominant_task_collapse(operation: str, scenario: Scenario, instruction: str) -> Optional[InstructionAnswer]:
    """Falls back to the single most-drilled operation (retrieve) instead
    of the QA-shaped operation actually requested."""

    if operation not in ("aggregate", "identify_holder", "classify_observed"):
        return None
    return _retrieve_value(scenario)


def default_qa_behavior(operation: str, scenario: Scenario, instruction: str) -> Optional[InstructionAnswer]:
    """Answers a factual "how many" question regardless of the
    instruction -- diagnostic specifically for the two operations that
    aren't shaped like a QA question at all (repeat, reject)."""

    if operation not in ("repeat", "reject_unsupported"):
        return None
    return _retrieve_value(scenario)


def keyword_routing(operation: str, scenario: Scenario, instruction: str) -> Optional[InstructionAnswer]:
    """Confuses the two "wh"-shaped operations with each other purely by
    surface phrasing (Who has X? / What did X do?)."""

    if operation == "identify_holder":
        return _classify_observed_value(scenario)
    if operation == "classify_observed":
        return _identify_holder_value(scenario)
    return None


INSTRUCTION_SHORTCUTS = {
    "word_problem_reflex": word_problem_reflex,
    "dominant_task_collapse": dominant_task_collapse,
    "default_qa_behavior": default_qa_behavior,
    "keyword_routing": keyword_routing,
}

INSTRUCTION_SHORTCUT_NAMES: tuple[str, ...] = tuple(INSTRUCTION_SHORTCUTS.keys())


def instruction_shortcut_predictions(
    operation: str, scenario: Scenario, instruction: str,
) -> dict[str, InstructionAnswer]:
    predictions: dict[str, InstructionAnswer] = {}
    for name, fn in INSTRUCTION_SHORTCUTS.items():
        value = fn(operation, scenario, instruction)
        if value is not None:
            predictions[name] = value
    return predictions
