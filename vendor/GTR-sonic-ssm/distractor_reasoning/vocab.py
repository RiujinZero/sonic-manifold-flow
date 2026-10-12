"""Vocabulary pools: names, nouns (with a category taxonomy), relations,
and sentence/question templates.

Everything here is split into a ``train`` and an ``eval`` slice so the
held-out evaluation stream (``evaluate.py``) can use names, nouns, and
phrasings the training generator never produces -- surface-level
memorization of *any* specific wording can't help on held-out examples,
only the underlying binding/aggregation behavior can.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Noun:
    singular: str
    plural: str
    # Ordered specific -> general; "pet" is a cross-cutting category, not
    # a taxonomic rank, so it's included explicitly only for nouns that
    # are actually typical pets (a sparrow is a bird but not a pet).
    categories: tuple[str, ...]


# -- Names ------------------------------------------------------------------
# Held-out names never appear in training-generated examples.
TRAIN_NAMES: tuple[str, ...] = (
    "Kevin", "Maria", "Lisa", "James", "Anna", "Robert", "Sofia", "David",
    "Elena", "Marcus", "Priya", "Tom", "Nina", "Carlos", "Grace", "Omar",
    "Julia", "Felix", "Ines", "Victor",
)
EVAL_NAMES: tuple[str, ...] = (
    "Wendy", "Hassan", "Freya", "Diego", "Mei", "Aaron", "Talia", "Bruno",
)

# -- Nouns and their category taxonomy --------------------------------------
# categories used across the pool: bird, mammal, reptile, pet, animal,
# vehicle, possession. "animal" is the top of the animal branch; "pet" cuts
# across bird/mammal/reptile for nouns people typically keep as pets.
TRAIN_NOUNS: dict[str, Noun] = {
    "parrot": Noun("parrot", "parrots", ("bird", "pet", "animal")),
    "cat": Noun("cat", "cats", ("mammal", "pet", "animal")),
    "dog": Noun("dog", "dogs", ("mammal", "pet", "animal")),
    "rabbit": Noun("rabbit", "rabbits", ("mammal", "pet", "animal")),
    "sparrow": Noun("sparrow", "sparrows", ("bird", "animal")),
    "pigeon": Noun("pigeon", "pigeons", ("bird", "animal")),
    "lizard": Noun("lizard", "lizards", ("reptile", "pet", "animal")),
    "bicycle": Noun("bicycle", "bicycles", ("vehicle", "possession")),
    "car": Noun("car", "cars", ("vehicle", "possession")),
    "umbrella": Noun("umbrella", "umbrellas", ("possession",)),
    "book": Noun("book", "books", ("possession",)),
    "backpack": Noun("backpack", "backpacks", ("possession",)),
}
EVAL_NOUNS: dict[str, Noun] = {
    "hamster": Noun("hamster", "hamsters", ("mammal", "pet", "animal")),
    "turtle": Noun("turtle", "turtles", ("reptile", "pet", "animal")),
    "falcon": Noun("falcon", "falcons", ("bird", "animal")),
    "watch": Noun("watch", "watches", ("possession",)),
    "guitar": Noun("guitar", "guitars", ("possession",)),
}

CATEGORY_PLURALS: dict[str, str] = {
    "bird": "birds", "mammal": "mammals", "reptile": "reptiles",
    "pet": "pets", "animal": "animals", "vehicle": "vehicles",
    "possession": "possessions",
}

# -- Relations ----------------------------------------------------------
# "Having" relations are semantically interchangeable for query resolution
# -- a question always asks "does X have", and any of has/owns/keeps
# answers it, regardless of which verb the context sentence used.
HAVING_RELATIONS: tuple[str, ...] = ("has", "owns", "keeps")
# Observation relations look similar (same subject/object slots) but never
# establish possession -- a fact stated with one of these must never count
# toward "how many does X have".
OBSERVE_RELATIONS: tuple[str, ...] = ("photographed", "counted", "saw", "spotted")

# -- Sentence templates -------------------------------------------------
TRAIN_HAVING_TEMPLATES: tuple[str, ...] = (
    "{subject} {relation} {quantity} {noun_phrase}.",
    "{subject} {relation} {quantity} {noun_phrase} in total.",
    "{subject} currently {relation} {quantity} {noun_phrase}.",
)
EVAL_HAVING_TEMPLATES: tuple[str, ...] = (
    "{subject} {relation} a total of {quantity} {noun_phrase}.",
)

TRAIN_ZERO_TEMPLATES: tuple[str, ...] = (
    "{subject} has no {noun_phrase}.",
    "{subject} owns no {noun_phrase}.",
    "{subject} has 0 {noun_phrase}.",
)
EVAL_ZERO_TEMPLATES: tuple[str, ...] = (
    "{subject} does not have any {noun_phrase}.",
)

TRAIN_OBSERVE_TEMPLATES: tuple[str, ...] = (
    "{subject} {relation} {quantity} {noun_phrase}.",
    "{subject} {relation} {quantity} {noun_phrase} yesterday.",
    "{subject} {relation} {quantity} {noun_phrase} last week.",
)
EVAL_OBSERVE_TEMPLATES: tuple[str, ...] = (
    "{subject} {relation} {quantity} {noun_phrase} during the trip.",
)

TRAIN_REPORTED_TEMPLATES: tuple[str, ...] = (
    "{speaker} says {subject} {relation} {quantity} {noun_phrase}.",
    "{speaker} claims {subject} {relation} {quantity} {noun_phrase}.",
)
EVAL_REPORTED_TEMPLATES: tuple[str, ...] = (
    "According to {speaker}, {subject} {relation} {quantity} {noun_phrase}.",
)

TRAIN_QUESTION_TEMPLATES: tuple[str, ...] = (
    "How many {noun_phrase} does {subject} have?",
    "How many {noun_phrase} does {subject} own?",
    "What number of {noun_phrase} does {subject} have?",
)
EVAL_QUESTION_TEMPLATES: tuple[str, ...] = (
    "How many {noun_phrase} does {subject} currently have?",
)
# Used only when a reported (hearsay) fact for the same subject/noun is
# present in the context, to disambiguate "what was claimed" from "what
# is actually true" -- see difficulty level 9.
TRAIN_QUESTION_ACTUAL_TEMPLATES: tuple[str, ...] = (
    "How many {noun_phrase} does {subject} actually have?",
)
EVAL_QUESTION_ACTUAL_TEMPLATES: tuple[str, ...] = (
    "How many {noun_phrase} does {subject} really have?",
)

# -- Answer templates -----------------------------------------------------
# Targets are full conversational sentences, not a bare number/UNKNOWN
# token, so a well-trained model produces something a person would
# actually say. The zero case still states the quantity as a literal "0"
# (rather than "no"/"none") purely so evaluation can extract it with the
# same digit regex used for every other answerable case -- it reads
# naturally either way ("Kevin has 0 parrots.").
TRAIN_ANSWER_TEMPLATES: tuple[str, ...] = (
    "{subject} has {quantity} {noun_phrase}.",
    "{subject} has {quantity} {noun_phrase} in total.",
    "That's {quantity} {noun_phrase} for {subject}.",
)
EVAL_ANSWER_TEMPLATES: tuple[str, ...] = (
    "{subject} currently has {quantity} {noun_phrase}.",
)

# Unknown-answer templates deliberately avoid the literal word "UNKNOWN"
# to stay conversational; evaluation instead looks for one of a small set
# of stable phrase indicators (see evaluate.py's UNKNOWN_INDICATORS).
TRAIN_ANSWER_UNKNOWN_TEMPLATES: tuple[str, ...] = (
    "I don't have enough information to know how many {noun_phrase} {subject} has.",
    "There's nothing here that tells me how many {noun_phrase} {subject} has.",
)
EVAL_ANSWER_UNKNOWN_TEMPLATES: tuple[str, ...] = (
    "I can't tell how many {noun_phrase} {subject} currently has from what's given.",
)


def answer_templates_for(split: str) -> tuple[str, ...]:
    return TRAIN_ANSWER_TEMPLATES if split == "train" else EVAL_ANSWER_TEMPLATES


def answer_unknown_templates_for(split: str) -> tuple[str, ...]:
    return (
        TRAIN_ANSWER_UNKNOWN_TEMPLATES if split == "train"
        else EVAL_ANSWER_UNKNOWN_TEMPLATES
    )


def nouns_for(split: str) -> dict[str, Noun]:
    if split == "train":
        return TRAIN_NOUNS
    if split == "eval":
        return dict(TRAIN_NOUNS) | EVAL_NOUNS
    raise ValueError(f"unknown split: {split!r}")


def names_for(split: str) -> tuple[str, ...]:
    if split == "train":
        return TRAIN_NAMES
    if split == "eval":
        return EVAL_NAMES
    raise ValueError(f"unknown split: {split!r}")


def having_templates_for(split: str) -> tuple[str, ...]:
    return TRAIN_HAVING_TEMPLATES if split == "train" else EVAL_HAVING_TEMPLATES


def zero_templates_for(split: str) -> tuple[str, ...]:
    return TRAIN_ZERO_TEMPLATES if split == "train" else EVAL_ZERO_TEMPLATES


def observe_templates_for(split: str) -> tuple[str, ...]:
    return TRAIN_OBSERVE_TEMPLATES if split == "train" else EVAL_OBSERVE_TEMPLATES


def reported_templates_for(split: str) -> tuple[str, ...]:
    return TRAIN_REPORTED_TEMPLATES if split == "train" else EVAL_REPORTED_TEMPLATES


def question_templates_for(split: str) -> tuple[str, ...]:
    return TRAIN_QUESTION_TEMPLATES if split == "train" else EVAL_QUESTION_TEMPLATES


def question_actual_templates_for(split: str) -> tuple[str, ...]:
    return (
        TRAIN_QUESTION_ACTUAL_TEMPLATES if split == "train"
        else EVAL_QUESTION_ACTUAL_TEMPLATES
    )
