"""Correction Generator: update a tracked value when a later statement
explicitly corrects an earlier one, rather than either freezing on the
first value or blindly trusting whatever number appeared most recently.

    Kevin owns 7 parrots.
    Correction: Kevin owns 4 parrots.
    Maria owns 9 parrots.

    Q: How many parrots does Kevin own?
    A: Kevin owns 4 parrots.

The decoy final line (Maria's, unrelated to the query) is placed
*after* the real correction specifically so a model that just grabs "the
last number mentioned anywhere" gets it wrong -- this is the exact
construction the design doc's Detection section asks for.

Every statement -- whether phrased as a plain fact, an explicit
"Correction:"/"Actually,"/"Wait," restatement, or a pronoun-referring one
("they own 4 parrots" instead of repeating the subject's name) -- resolves
the same way: the *last* statement touching a given (subject, noun) pair
wins. A correction is not semantically special beyond being a later
statement about the same pair; the phrasing exists only to make the human-
readable text natural, matching how people actually correct themselves in
conversation. Observation-style mentions (someone merely *seeing* a
number near the subject) never count, mirroring the having/observe
relation split used throughout this package.

Out of scope for this pass (see writing-block.md's full type list):
spelling/name corrections -- those correct a string attribute, not a
tracked quantity, and don't fit this module's numeric-answer contract.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional, Union

from .query import UNKNOWN, Query
from .vocab import HAVING_RELATIONS, Noun, names_for, nouns_for

HAVING_RELATIONS_PLURAL: dict[str, str] = {"has": "have", "owns": "own", "keeps": "keep"}

StatementKind = Literal["initial", "correction", "decoy_fact", "decoy_correction", "decoy_observation"]


@dataclass(frozen=True)
class Statement:
    subject: str
    noun: str
    quantity: int
    kind: StatementKind
    pronoun: bool = False


def resolve_correction_query(query: Query, statements: list[Statement]) -> Union[int, Literal["UNKNOWN"]]:
    """The last statement touching (query.subject, query.target) wins;
    observation-style mentions never count. UNKNOWN if the pair is never
    touched by anything but an observation."""

    key = (query.subject, query.target)
    value: Optional[int] = None
    for s in statements:
        if s.kind == "decoy_observation":
            continue
        if (s.subject, s.noun) == key:
            value = s.quantity
    return value if value is not None else UNKNOWN


# -- Rendering --------------------------------------------------------------

CORRECTION_TEMPLATES: tuple[str, ...] = (
    "Correction: {subject} {relation} {quantity} {noun_phrase}.",
    "Actually, {subject} {relation} {quantity} {noun_phrase}.",
    "Wait, {subject} {relation} {quantity} {noun_phrase}.",
)
INITIAL_TEMPLATES: tuple[str, ...] = (
    "{subject} {relation} {quantity} {noun_phrase}.",
)
OBSERVATION_TEMPLATES: tuple[str, ...] = (
    "Someone reported seeing {quantity} {noun_phrase} near {subject}.",
)
QUESTION_TEMPLATES: tuple[str, ...] = (
    "How many {noun_phrase} does {subject} {relation_base}?",
)
ANSWER_TEMPLATES: tuple[str, ...] = (
    "{subject} {relation} {quantity} {noun_phrase}.",
)
ANSWER_UNKNOWN_TEMPLATES: tuple[str, ...] = (
    "I don't have enough information to know how many {noun_phrase} {subject} {relation_base}.",
)


def _noun_phrase(noun_info: Noun, quantity: int) -> str:
    return noun_info.singular if quantity == 1 else noun_info.plural


def render_statement(
    s: Statement, rng: random.Random, nouns: dict[str, Noun], relation: str,
) -> str:
    noun_info = nouns[s.noun]
    phrase = _noun_phrase(noun_info, s.quantity)
    subject_text = "They" if s.pronoun else s.subject
    if s.kind == "decoy_observation":
        return rng.choice(OBSERVATION_TEMPLATES).format(
            subject=s.subject, quantity=s.quantity, noun_phrase=phrase,
        )
    rel = HAVING_RELATIONS_PLURAL[relation] if s.pronoun else relation
    if s.kind in ("correction", "decoy_correction"):
        template = rng.choice(CORRECTION_TEMPLATES)
    else:
        template = rng.choice(INITIAL_TEMPLATES)
    return template.format(subject=subject_text, relation=rel, quantity=s.quantity, noun_phrase=phrase)


def render_correction_example(
    statements: list[Statement], query: Query, answer: Union[int, Literal["UNKNOWN"]],
    rng: random.Random, nouns: dict[str, Noun], relation: str, style: str = "qa",
) -> tuple[str, str]:
    context = "\n".join(render_statement(s, rng, nouns, relation) for s in statements)
    noun_info = nouns[query.target]
    question = rng.choice(QUESTION_TEMPLATES).format(
        subject=query.subject, noun_phrase=noun_info.plural, relation_base=HAVING_RELATIONS_PLURAL[relation],
    )
    question_line = f"Q: {question}" if style == "qa" else question
    prompt = f"{context}\n{question_line}"

    if answer == UNKNOWN:
        text = rng.choice(ANSWER_UNKNOWN_TEMPLATES).format(
            subject=query.subject, noun_phrase=noun_info.plural,
            relation_base=HAVING_RELATIONS_PLURAL[relation],
        )
    else:
        phrase = _noun_phrase(noun_info, answer)
        text = rng.choice(ANSWER_TEMPLATES).format(
            subject=query.subject, relation=relation, quantity=answer, noun_phrase=phrase,
        )
    answer_text = f"A: {text}" if style == "qa" else f"Answer: {text}"
    return prompt, answer_text


# -- Decoys ---------------------------------------------------------------

def _quantity(rng: random.Random, avoid: Optional[int] = None) -> int:
    value = rng.randint(1, 12)
    while value == avoid:
        value = rng.randint(1, 12)
    return value


def make_decoy_fact(
    subject: str, noun: str, nouns: dict[str, Noun], names: tuple[str, ...], rng: random.Random,
) -> Statement:
    """A plain new fact about someone else entirely -- the exact
    construction that defeats a subject-blind "last number" shortcut."""

    other_subject = rng.choice([n for n in names if n != subject])
    other_noun = rng.choice(list(nouns.keys()))
    return Statement(other_subject, other_noun, _quantity(rng), "decoy_fact")


def make_decoy_correction(
    subject: str, noun: str, nouns: dict[str, Noun], names: tuple[str, ...], rng: random.Random,
) -> Statement:
    """A *correction*-phrased statement, but about someone else -- defeats
    "grab the last thing phrased as a correction" regardless of subject."""

    other_subject = rng.choice([n for n in names if n != subject])
    other_noun = rng.choice(list(nouns.keys()))
    return Statement(other_subject, other_noun, _quantity(rng), "decoy_correction")


def make_decoy_same_subject_other_noun(
    subject: str, noun: str, nouns: dict[str, Noun], names: tuple[str, ...], rng: random.Random,
) -> Statement:
    """Same subject, different noun -- tests that the correction is bound
    to the right noun, not just the right person."""

    other_noun = rng.choice([n for n in nouns if n != noun])
    kind = rng.choice(("decoy_fact", "decoy_correction"))
    return Statement(subject, other_noun, _quantity(rng), kind)


def make_decoy_observation(
    subject: str, noun: str, nouns: dict[str, Noun], names: tuple[str, ...], rng: random.Random,
) -> Statement:
    """Mentions the real (subject, noun) again with a different number,
    but as a mere sighting -- must never override the tracked value."""

    return Statement(subject, noun, _quantity(rng), "decoy_observation")


DECOY_MAKERS = {
    "decoy_fact": make_decoy_fact,
    "decoy_correction": make_decoy_correction,
    "decoy_same_subject_other_noun": make_decoy_same_subject_other_noun,
    "decoy_observation": make_decoy_observation,
}
ALL_DECOY_KINDS: tuple[str, ...] = tuple(DECOY_MAKERS.keys())


# -- Difficulty curriculum -------------------------------------------------

@dataclass(frozen=True)
class CorrectionDifficultyConfig:
    level: int
    min_corrections: int = 1
    max_corrections: int = 1
    correction_probability: float = 1.0
    allow_pronoun: bool = False
    decoy_kinds: tuple[str, ...] = ()
    min_decoys: int = 0
    max_decoys: int = 0
    answerable_probability: float = 1.0

    @property
    def allow_decoys(self) -> bool:
        return self.max_decoys > 0


CORRECTION_DIFFICULTY_LEVELS: dict[int, CorrectionDifficultyConfig] = {
    1: CorrectionDifficultyConfig(level=1, min_corrections=1, max_corrections=1),
    2: CorrectionDifficultyConfig(
        level=2, min_corrections=0, max_corrections=1, correction_probability=0.6,
    ),
    3: CorrectionDifficultyConfig(
        level=3, min_corrections=1, max_corrections=1,
        decoy_kinds=("decoy_fact",), min_decoys=1, max_decoys=2,
    ),
    4: CorrectionDifficultyConfig(
        level=4, min_corrections=1, max_corrections=2, allow_pronoun=True,
        decoy_kinds=("decoy_fact", "decoy_same_subject_other_noun"),
        min_decoys=1, max_decoys=2,
    ),
    5: CorrectionDifficultyConfig(
        level=5, min_corrections=1, max_corrections=3, allow_pronoun=True,
        decoy_kinds=ALL_DECOY_KINDS, min_decoys=2, max_decoys=3,
    ),
    6: CorrectionDifficultyConfig(
        level=6, min_corrections=0, max_corrections=3, correction_probability=0.7,
        allow_pronoun=True, decoy_kinds=ALL_DECOY_KINDS, min_decoys=2, max_decoys=4,
        answerable_probability=0.6,
    ),
}


def correction_config_for_level(level: int) -> CorrectionDifficultyConfig:
    if level not in CORRECTION_DIFFICULTY_LEVELS:
        raise ValueError(f"correction difficulty level must be in 1..6, got {level}")
    return CORRECTION_DIFFICULTY_LEVELS[level]


# -- Example assembly ---------------------------------------------------

@dataclass
class CorrectionExample:
    prompt: str
    answer_text: str
    answer_value: Union[int, Literal["UNKNOWN"]]
    difficulty: int
    num_corrections: int
    num_decoys: int
    decoy_kinds: tuple[str, ...]
    has_pronoun_correction: bool
    is_unknown_case: bool
    statements: list[Statement]
    query: Query
    shortcut_predictions: dict[str, int]


def generate_correction_example(
    rng: random.Random, level: int, split: str = "train", style: str = "qa",
) -> CorrectionExample:
    config = correction_config_for_level(level)
    nouns = nouns_for(split)
    names = names_for(split)
    relation = rng.choice(HAVING_RELATIONS)

    subject = rng.choice(names)
    noun = rng.choice(list(nouns.keys()))

    initial_value = rng.randint(1, 12)
    chain: list[Statement] = [Statement(subject, noun, initial_value, "initial")]
    current_value = initial_value
    has_pronoun = False

    if rng.random() < config.correction_probability:
        num_corrections = rng.randint(config.min_corrections, config.max_corrections)
        for _ in range(num_corrections):
            new_value = _quantity(rng, avoid=current_value)
            pronoun = config.allow_pronoun and rng.random() < 0.5
            has_pronoun = has_pronoun or pronoun
            chain.append(Statement(subject, noun, new_value, "correction", pronoun=pronoun))
            current_value = new_value

    decoys: list[Statement] = []
    kinds_used: list[str] = []
    if config.allow_decoys:
        count = rng.randint(config.min_decoys, config.max_decoys)
        available = list(config.decoy_kinds)
        for _ in range(count):
            if not available:
                break
            kind = rng.choice(available)
            decoys.append(DECOY_MAKERS[kind](subject, noun, nouns, names, rng))
            kinds_used.append(kind)

    # Decoys always come after the real chain -- this is the specific
    # construction that defeats a subject-blind "last number wins"
    # shortcut (see module docstring / design doc Detection section).
    statements = chain + decoys

    unknown_case = rng.random() >= config.answerable_probability
    if unknown_case:
        # Pick any (subject, noun) pair nothing in the passage touches.
        used_pairs = {(s.subject, s.noun) for s in statements}
        shuffled_names = list(names)
        rng.shuffle(shuffled_names)
        shuffled_nouns = list(nouns.keys())
        rng.shuffle(shuffled_nouns)
        query_subject, query_noun = next(
            (cand_subject, cand_noun)
            for cand_subject in shuffled_names
            for cand_noun in shuffled_nouns
            if (cand_subject, cand_noun) not in used_pairs
        )
        query = Query(subject=query_subject, target=query_noun, is_category=False)
    else:
        query = Query(subject=subject, target=noun, is_category=False)

    answer = resolve_correction_query(query, statements)

    prompt, answer_text = render_correction_example(
        statements, query, answer, rng, nouns, relation, style=style,
    )

    from .correction_shortcuts import correction_shortcut_predictions

    return CorrectionExample(
        prompt=prompt,
        answer_text=answer_text,
        answer_value=answer,
        difficulty=level,
        num_corrections=len(chain) - 1,
        num_decoys=len(decoys),
        decoy_kinds=tuple(kinds_used),
        has_pronoun_correction=has_pronoun,
        is_unknown_case=(answer == UNKNOWN),
        statements=statements,
        query=query,
        shortcut_predictions=correction_shortcut_predictions(statements, query),
    )
