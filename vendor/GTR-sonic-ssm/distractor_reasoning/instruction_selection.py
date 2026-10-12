"""Instruction-Selection Generator: determine what operation is actually
being requested -- retrieval, category aggregation, identifying who holds
something, classifying an observation, verbatim repetition, or recognizing
an unanswerable question -- before attempting to answer, rather than
defaulting to "extract a number" for every prompt.

    Kevin has 5 parrots.
    Maria has 8 cats.
    Kevin photographed 3 cats.

    How many parrots does Kevin have?           -> retrieve            -> 5
    How many animals does Kevin have?            -> aggregate           -> 5
    Who has cats?                                -> identify_holder     -> Maria
    What did Kevin photograph?                   -> classify_observed   -> cats
    Repeat the first sentence.                    -> repeat              -> "Kevin has 5 parrots."
    Is there enough information to know how many
    lizards Kevin owns?                          -> reject_unsupported  -> No

Every example reuses the *same shape* of shared world (two subjects, one
having-fact each, one observation) and asks one randomly chosen operation
about it -- this is deliberately "reuse nearly identical contexts with
different instructions," the design doc's own detection technique: a
model that has only learned to answer "how many X" questions has no
surface cue left to fall back on once the instruction changes but the
context doesn't.

Answer *types* genuinely differ by operation (a count, a name, a noun, a
verbatim sentence, a yes/no) -- unlike every other generator in this
package, there is no single numeric-or-UNKNOWN contract here, so parsing
and shortcut prediction are both dispatched per operation.

Out of scope for this pass (see writing-block.md's full operation list):
compare, summarize, and transform/ignore-selected-information -- retrieve,
aggregate, identify_holder, classify_observed, repeat, and
reject_unsupported already exercise all four named cheating strategies and
match the design doc's own worked example almost exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional, Union

from .query import UNKNOWN, Query, resolve_query
from .render import render_fact
from .vocab import HAVING_RELATIONS, OBSERVE_RELATIONS, Noun, names_for, nouns_for
from .world import Fact, World

OPERATIONS: tuple[str, ...] = (
    "retrieve", "aggregate", "identify_holder", "classify_observed",
    "reject_unsupported", "repeat",
)

# "What did X {verb}?" needs the base/infinitive form -- OBSERVE_RELATIONS
# itself is already past tense (used directly in fact sentences), so
# "What did Kevin photographed?" would double up the tense.
OBSERVE_RELATIONS_BASE: dict[str, str] = {
    "photographed": "photograph", "counted": "count", "saw": "see", "spotted": "spot",
}

InstructionAnswer = Union[int, Literal["UNKNOWN", "YES", "NO"], str, tuple[str, ...]]


@dataclass(frozen=True)
class Scenario:
    world: World
    nouns: dict[str, Noun]
    names: tuple[str, ...]
    subjects: tuple[str, str]
    primary_noun: str      # subjects[0]'s having-fact noun
    secondary_noun: str    # subjects[1]'s having-fact noun; also what subjects[0] observed
    unheld_noun: str       # a noun nobody has any having-fact for
    sentences: list[str]   # rendered fact sentences, in the exact order shown
    observe_relation: str


def _build_scenario(rng: random.Random, split: str, extra_aggregation_fact: bool) -> Scenario:
    nouns = nouns_for(split)
    names = names_for(split)
    subjects = tuple(rng.sample(names, k=2))
    noun_pool = list(nouns.keys())
    rng.shuffle(noun_pool)
    primary_noun, secondary_noun, unheld_noun = noun_pool[0], noun_pool[1], noun_pool[2]
    observe_relation = rng.choice(OBSERVE_RELATIONS)

    facts = [
        Fact(subjects[0], rng.choice(HAVING_RELATIONS), primary_noun, rng.randint(1, 9)),
        Fact(subjects[1], rng.choice(HAVING_RELATIONS), secondary_noun, rng.randint(1, 9)),
        Fact(subjects[0], observe_relation, secondary_noun, rng.randint(1, 9)),
    ]
    if extra_aggregation_fact:
        same_category = [
            n for n, info in nouns.items()
            if n != primary_noun and set(info.categories) & set(nouns[primary_noun].categories)
        ]
        if same_category:
            facts.append(Fact(subjects[0], rng.choice(HAVING_RELATIONS), rng.choice(same_category), rng.randint(1, 9)))

    sentences = [render_fact(f, rng, split, nouns) for f in facts]
    return Scenario(
        world=World(facts), nouns=nouns, names=names, subjects=subjects,
        primary_noun=primary_noun, secondary_noun=secondary_noun, unheld_noun=unheld_noun,
        sentences=sentences, observe_relation=observe_relation,
    )


def _category_containing(scenario: Scenario, noun: str, rng: random.Random) -> Optional[str]:
    categories = scenario.nouns[noun].categories
    return rng.choice(categories) if categories else None


# -- Operations: each returns (instruction_text, answer_value, answer_text) --

def _op_retrieve(scenario: Scenario, rng: random.Random, split: str) -> tuple[str, InstructionAnswer, str]:
    subject, noun = scenario.subjects[0], scenario.primary_noun
    noun_info = scenario.nouns[noun]
    instruction = f"How many {noun_info.plural} does {subject} have?"
    value = resolve_query(Query(subject, noun, False), scenario.world, scenario.nouns)
    phrase = noun_info.plural if value != 1 else noun_info.singular
    answer = f"{subject} has {value} {phrase}." if value != UNKNOWN else (
        f"I don't have enough information to know how many {noun_info.plural} {subject} has."
    )
    return instruction, value, answer


def _op_aggregate(scenario: Scenario, rng: random.Random, split: str) -> tuple[str, InstructionAnswer, str]:
    subject, noun = scenario.subjects[0], scenario.primary_noun
    category = _category_containing(scenario, noun, rng)
    if category is None:
        return _op_retrieve(scenario, rng, split)
    from .vocab import CATEGORY_PLURALS
    plural = CATEGORY_PLURALS[category]
    instruction = f"How many {plural} does {subject} have?"
    value = resolve_query(Query(subject, category, True), scenario.world, scenario.nouns)
    answer = f"{subject} has {value} {plural}." if value != UNKNOWN else (
        f"I don't have enough information to know how many {plural} {subject} has."
    )
    return instruction, value, answer


def _op_identify_holder(scenario: Scenario, rng: random.Random, split: str) -> tuple[str, InstructionAnswer, str]:
    noun_info = scenario.nouns[scenario.secondary_noun]
    instruction = f"Who has {noun_info.plural}?"
    holders = tuple(sorted({
        f.subject for f in scenario.world.direct_having_facts() if f.noun == scenario.secondary_noun
    }))
    if not holders:
        return instruction, (), f"No one has {noun_info.plural}."
    verb = "has" if len(holders) == 1 else "have"
    answer = f"{' and '.join(holders)} {verb} {noun_info.plural}."
    return instruction, holders, answer


def _op_classify_observed(scenario: Scenario, rng: random.Random, split: str) -> tuple[str, InstructionAnswer, str]:
    subject = scenario.subjects[0]
    base_verb = OBSERVE_RELATIONS_BASE[scenario.observe_relation]
    instruction = f"What did {subject} {base_verb}?"
    observed = tuple(sorted({
        f.noun for f in scenario.world.facts
        if f.subject == subject and f.relation == scenario.observe_relation and not f.is_having
    }))
    if not observed:
        return instruction, (), f"{subject} didn't {scenario.observe_relation} anything mentioned here."
    phrases = [scenario.nouns[n].plural for n in observed]
    answer = f"{subject} {scenario.observe_relation} {' and '.join(phrases)}."
    return instruction, observed, answer


def _op_reject_unsupported(scenario: Scenario, rng: random.Random, split: str) -> tuple[str, InstructionAnswer, str]:
    answerable = rng.random() < 0.5
    subject = scenario.subjects[0]
    noun = scenario.primary_noun if answerable else scenario.unheld_noun
    noun_info = scenario.nouns[noun]
    instruction = f"Is there enough information to know how many {noun_info.plural} {subject} owns?"
    value: Literal["YES", "NO"] = "YES" if answerable else "NO"
    answer = (
        f"Yes, there's enough information to know how many {noun_info.plural} {subject} owns."
        if answerable else
        f"No, there isn't enough information to know how many {noun_info.plural} {subject} owns."
    )
    return instruction, value, answer


def _op_repeat(scenario: Scenario, rng: random.Random, split: str) -> tuple[str, InstructionAnswer, str]:
    instruction = "Repeat the first sentence."
    target = scenario.sentences[0]
    return instruction, target, target


OPERATION_HANDLERS = {
    "retrieve": _op_retrieve,
    "aggregate": _op_aggregate,
    "identify_holder": _op_identify_holder,
    "classify_observed": _op_classify_observed,
    "reject_unsupported": _op_reject_unsupported,
    "repeat": _op_repeat,
}


# -- Difficulty curriculum -------------------------------------------------

@dataclass(frozen=True)
class InstructionDifficultyConfig:
    level: int
    operations: tuple[str, ...]
    extra_aggregation_fact_probability: float = 0.0


INSTRUCTION_DIFFICULTY_LEVELS: dict[int, InstructionDifficultyConfig] = {
    1: InstructionDifficultyConfig(level=1, operations=("retrieve",)),
    2: InstructionDifficultyConfig(level=2, operations=("retrieve", "identify_holder", "classify_observed")),
    3: InstructionDifficultyConfig(
        level=3, operations=("retrieve", "aggregate", "identify_holder", "classify_observed"),
        extra_aggregation_fact_probability=0.5,
    ),
    4: InstructionDifficultyConfig(
        level=4, operations=OPERATIONS, extra_aggregation_fact_probability=0.5,
    ),
}


def instruction_config_for_level(level: int) -> InstructionDifficultyConfig:
    if level not in INSTRUCTION_DIFFICULTY_LEVELS:
        raise ValueError(f"instruction-selection difficulty level must be in 1..4, got {level}")
    return INSTRUCTION_DIFFICULTY_LEVELS[level]


# -- Example assembly ---------------------------------------------------

@dataclass
class InstructionExample:
    prompt: str
    answer_text: str
    answer_value: InstructionAnswer
    difficulty: int
    operation: str
    scenario: Scenario
    shortcut_predictions: dict[str, InstructionAnswer]


def generate_instruction_example(
    rng: random.Random, level: int, split: str = "train", style: str = "qa",
) -> InstructionExample:
    config = instruction_config_for_level(level)
    scenario = _build_scenario(rng, split, rng.random() < config.extra_aggregation_fact_probability)
    operation = rng.choice(config.operations)
    instruction, answer_value, answer_text = OPERATION_HANDLERS[operation](scenario, rng, split)

    context = "\n".join(scenario.sentences)
    instruction_line = f"Q: {instruction}" if style == "qa" else instruction
    prompt = f"{context}\n{instruction_line}"
    answer_prefixed = f"A: {answer_text}" if style == "qa" else f"Answer: {answer_text}"

    from .instruction_selection_shortcuts import instruction_shortcut_predictions

    return InstructionExample(
        prompt=prompt,
        answer_text=answer_prefixed,
        answer_value=answer_value,
        difficulty=level,
        operation=operation,
        scenario=scenario,
        shortcut_predictions=instruction_shortcut_predictions(operation, scenario, instruction),
    )
