"""Ontology Generator: reason across a 2-level category hierarchy (leaf ->
parent -> grandparent, with multiple inheritance via one cross-cutting
category per taxonomy) that is explicitly *stated in the prompt*, using
entirely invented (nonsense) words -- so a correct answer can only come
from reading the supplied taxonomy in context, never from memorized
real-world category knowledge. A parrot is a bird whether or not a
sentence ever says so; a "zorbin" is only ever a "flarn" because the
prompt says so, and the held-out eval taxonomy uses different invented
words entirely, so surface memorization of the train taxonomy can't help.

    A zorbin is a kind of flarn.
    A flarn is a kind of trindle.
    Kevin has 3 zorbins.
    Kevin has 2 nexlins.
    Kevin photographed 9 dravors.

    Q: How many trindles does Kevin have?
    A: Kevin has 5 trindles.

(3 zorbins + 2 nexlins: zorbin -> flarn -> trindle and nexlin -> brellow
-> trindle, two different parents converging on the same grandparent --
the photographed dravor never establishes possession.)

Reuses ``world.Fact``/``World`` and ``query.Query``/``resolve_query``
completely unchanged: a noun's ``categories`` tuple already carries every
ancestor name regardless of depth, so category-match resolution needs no
new logic. ``distractors.make_distractor`` is reused too -- its six kinds
are generic over any ``dict[str, Noun]``, Ontology's synthetic taxonomy
included.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional, Union

from .difficulty import ALL_DISTRACTOR_KINDS
from .distractors import make_distractor
from .query import UNKNOWN, Query, resolve_query
from .render import render_answer, render_fact, render_question
from .vocab import HAVING_RELATIONS, Noun, names_for
from .world import Fact, World


@dataclass(frozen=True)
class Taxonomy:
    grandparents: tuple[str, ...]
    # parent -> grandparent, or None for a cross-cutting parent category
    # that doesn't roll up any further (mirrors "pet" in the Binding
    # generator's real-world taxonomy: spans multiple animal branches
    # without itself being a branch of "animal").
    parent_to_grandparent: dict[str, Optional[str]]
    leaf_to_parents: dict[str, tuple[str, ...]]

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(self.leaf_to_parents.keys())

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(self.parent_to_grandparent.keys())

    def ancestors_of(self, leaf: str) -> tuple[str, ...]:
        result: list[str] = []
        for parent in self.leaf_to_parents[leaf]:
            result.append(parent)
            grandparent = self.parent_to_grandparent.get(parent)
            if grandparent is not None:
                result.append(grandparent)
        return tuple(dict.fromkeys(result))


TRAIN_TAXONOMY = Taxonomy(
    grandparents=("trindle", "moklar", "vexoid"),
    parent_to_grandparent={
        "flarn": "trindle", "glimset": "trindle",
        "brellow": "moklar", "dackspur": "moklar",
        "quenith": "vexoid", "storbek": "vexoid",
        "glim": None,  # cross-cutting, one leaf from each grandparent branch
    },
    leaf_to_parents={
        "zorbin": ("flarn", "glim"),
        "plibbon": ("flarn",),
        "quilfex": ("glimset",),
        "dravor": ("glimset",),
        "nexlin": ("brellow", "glim"),
        "worbat": ("brellow",),
        "cravun": ("dackspur",),
        "sindle": ("dackspur",),
        "molvex": ("quenith", "glim"),
        "tarnip": ("quenith",),
        "yendral": ("storbek",),
        "prask": ("storbek",),
    },
)

EVAL_TAXONOMY = Taxonomy(
    grandparents=("quorlath", "brendak"),
    parent_to_grandparent={
        "velmish": "quorlath", "haffrin": "quorlath",
        "dolmeck": "brendak", "trewsic": "brendak",
        "florn": None,
    },
    leaf_to_parents={
        "sprundle": ("velmish", "florn"),
        "cavrin": ("velmish",),
        "molteg": ("haffrin",),
        "yispar": ("haffrin",),
        "brakoni": ("dolmeck", "florn"),
        "twindle": ("trewsic",),
    },
)


def taxonomy_for(split: str) -> Taxonomy:
    if split == "train":
        return TRAIN_TAXONOMY
    if split == "eval":
        return EVAL_TAXONOMY
    raise ValueError(f"unknown split: {split!r}")


def ontology_nouns_for(split: str) -> dict[str, Noun]:
    taxonomy = taxonomy_for(split)
    return {
        leaf: Noun(leaf, leaf + "s", taxonomy.ancestors_of(leaf))
        for leaf in taxonomy.leaves
    }


def category_plurals_for(split: str) -> dict[str, str]:
    taxonomy = taxonomy_for(split)
    names = set(taxonomy.parents) | set(taxonomy.grandparents)
    return {name: f"{name}s" for name in names}


def render_taxonomy_definitions(
    nouns_used: set[str], taxonomy: Taxonomy, rng: random.Random,
) -> list[str]:
    """One "A {child} is a kind of {parent}." sentence per taxonomy edge
    reachable from the nouns actually used in this example (not the whole
    taxonomy every time, to keep prompt length proportional)."""

    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for leaf in nouns_used:
        for parent in taxonomy.leaf_to_parents[leaf]:
            edge = (leaf, parent)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
            grandparent = taxonomy.parent_to_grandparent.get(parent)
            if grandparent is not None:
                edge2 = (parent, grandparent)
                if edge2 not in seen:
                    seen.add(edge2)
                    edges.append(edge2)
    rng.shuffle(edges)
    return [f"A {child} is a kind of {parent}." for child, parent in edges]


# -- Difficulty curriculum -------------------------------------------------

@dataclass(frozen=True)
class OntologyDifficultyConfig:
    level: int
    query_levels: tuple[str, ...] = ("leaf",)  # "leaf" | "parent" | "grandparent"
    min_distractors: int = 0
    max_distractors: int = 0
    distractor_kinds: tuple[str, ...] = ()
    min_filler_facts: int = 0
    max_filler_facts: int = 1
    answerable_probability: float = 1.0

    @property
    def allow_distractors(self) -> bool:
        return self.max_distractors > 0


ONTOLOGY_DIFFICULTY_LEVELS: dict[int, OntologyDifficultyConfig] = {
    1: OntologyDifficultyConfig(level=1, query_levels=("leaf",), answerable_probability=1.0),
    2: OntologyDifficultyConfig(level=2, query_levels=("leaf",), answerable_probability=0.5),
    3: OntologyDifficultyConfig(
        level=3, query_levels=("leaf", "parent"),
        min_filler_facts=0, max_filler_facts=2,
    ),
    4: OntologyDifficultyConfig(
        level=4, query_levels=("leaf", "parent"),
        min_distractors=1, max_distractors=2,
        distractor_kinds=("subject_relation", "relation_object", "subject_object"),
        min_filler_facts=1, max_filler_facts=3,
    ),
    5: OntologyDifficultyConfig(
        level=5, query_levels=("parent", "grandparent"),
        min_distractors=1, max_distractors=3,
        distractor_kinds=ALL_DISTRACTOR_KINDS,
        min_filler_facts=1, max_filler_facts=3,
        answerable_probability=0.7,
    ),
    6: OntologyDifficultyConfig(
        level=6, query_levels=("leaf", "parent", "grandparent"),
        min_distractors=2, max_distractors=4,
        distractor_kinds=ALL_DISTRACTOR_KINDS,
        min_filler_facts=1, max_filler_facts=4,
        answerable_probability=0.5,
    ),
}


def ontology_config_for_level(level: int) -> OntologyDifficultyConfig:
    if level not in ONTOLOGY_DIFFICULTY_LEVELS:
        raise ValueError(f"ontology difficulty level must be in 1..6, got {level}")
    return ONTOLOGY_DIFFICULTY_LEVELS[level]


# -- Example assembly ---------------------------------------------------

@dataclass
class OntologyExample:
    prompt: str
    answer_text: str
    answer_value: Union[int, Literal["UNKNOWN"]]
    difficulty: int
    query_level: str
    num_facts: int
    num_distractors: int
    distractor_kinds: tuple[str, ...]
    is_unknown_case: bool
    is_aggregation: bool
    world: World
    query: Query
    shortcut_predictions: dict[str, int]


def _pick_target(config: OntologyDifficultyConfig, taxonomy: Taxonomy, rng: random.Random) -> tuple[str, bool, str]:
    level = rng.choice(config.query_levels)
    if level == "leaf":
        return rng.choice(taxonomy.leaves), False, level
    if level == "parent":
        return rng.choice(taxonomy.parents), True, level
    return rng.choice(taxonomy.grandparents), True, level


def _leaves_under(taxonomy: Taxonomy, target: str, is_category: bool) -> list[str]:
    if not is_category:
        return [target]
    return [leaf for leaf in taxonomy.leaves if target in taxonomy.ancestors_of(leaf)]


def _build_direct_facts(
    subject: str, target: str, is_category: bool, outcome: str,
    taxonomy: Taxonomy, rng: random.Random,
) -> list[Fact]:
    if outcome == "unknown":
        return []
    candidates = _leaves_under(taxonomy, target, is_category)
    if outcome == "zero":
        leaf = rng.choice(candidates)
        return [Fact(subject, rng.choice(HAVING_RELATIONS), leaf, 0)]
    # nonzero -- for category targets with >=2 member leaves, prefer using
    # more than one (ideally from different parents) so the example
    # actually exercises hierarchical aggregation, not just retrieval.
    if is_category and len(candidates) >= 2 and rng.random() < 0.7:
        count = min(len(candidates), rng.randint(2, 3))
        chosen = rng.sample(candidates, k=count)
        return [Fact(subject, rng.choice(HAVING_RELATIONS), leaf, rng.randint(1, 9)) for leaf in chosen]
    leaf = rng.choice(candidates)
    return [Fact(subject, rng.choice(HAVING_RELATIONS), leaf, rng.randint(1, 9))]


def _build_filler_facts(
    subject: str, config: OntologyDifficultyConfig, names: tuple[str, ...],
    taxonomy: Taxonomy, rng: random.Random,
) -> list[Fact]:
    n = rng.randint(config.min_filler_facts, config.max_filler_facts)
    other_names = [name for name in names if name != subject]
    if not other_names or n == 0:
        return []
    fillers = []
    for _ in range(n):
        other_subject = rng.choice(other_names)
        relation = rng.choice(HAVING_RELATIONS + ("photographed", "counted", "saw", "spotted"))
        leaf = rng.choice(taxonomy.leaves)
        fillers.append(Fact(other_subject, relation, leaf, rng.randint(1, 12)))
    return fillers


def generate_ontology_example(
    rng: random.Random, level: int, split: str = "train", style: str = "qa",
) -> OntologyExample:
    config = ontology_config_for_level(level)
    taxonomy = taxonomy_for(split)
    nouns = ontology_nouns_for(split)
    category_plurals = category_plurals_for(split)
    names = names_for(split)

    subject = rng.choice(names)
    target, is_category, query_level = _pick_target(config, taxonomy, rng)

    if rng.random() < config.answerable_probability:
        outcome = "zero" if rng.random() < 0.15 else "nonzero"
    else:
        outcome = "unknown"

    direct_facts = _build_direct_facts(subject, target, is_category, outcome, taxonomy, rng)

    distractor_facts: list[Fact] = []
    kinds_used: list[str] = []
    if config.allow_distractors:
        count = rng.randint(config.min_distractors, config.max_distractors)
        available = list(config.distractor_kinds)
        for _ in range(count):
            if not available:
                break
            kind = rng.choice(available)
            fact = make_distractor(kind, subject, target, is_category, rng, nouns, names)
            if fact is None:
                continue
            distractor_facts.append(fact)
            kinds_used.append(kind)

    filler_facts = _build_filler_facts(subject, config, names, taxonomy, rng)

    all_facts = [*direct_facts, *distractor_facts, *filler_facts]
    query = Query(subject=subject, target=target, is_category=is_category)
    world = World(all_facts)
    answer = resolve_query(query, world, nouns)

    nouns_used = {f.noun for f in all_facts}
    if is_category:
        # The queried category's own edges must be stated even if no
        # fact happens to use one of its member leaves (an UNKNOWN
        # example still needs the taxonomy defined to be answerable-in-
        # principle from the text).
        nouns_used |= set(_leaves_under(taxonomy, target, True))
    definitions = render_taxonomy_definitions(nouns_used, taxonomy, rng)

    ordered_facts = list(all_facts)
    rng.shuffle(ordered_facts)
    fact_lines = [render_fact(f, rng, split, nouns) for f in ordered_facts]
    context_lines = definitions + fact_lines
    rng.shuffle(context_lines)
    context = "\n".join(context_lines)

    question = render_question(query, rng, split, nouns, category_plurals=category_plurals)
    question_line = f"Q: {question}" if style == "qa" else question
    prompt = f"{context}\n{question_line}"
    answer_text = render_answer(
        query, answer, rng, split, nouns, style=style, category_plurals=category_plurals,
    )

    from .ontology_shortcuts import ontology_shortcut_predictions

    return OntologyExample(
        prompt=prompt,
        answer_text=answer_text,
        answer_value=answer,
        difficulty=level,
        query_level=query_level,
        num_facts=len(all_facts),
        num_distractors=len(distractor_facts),
        distractor_kinds=tuple(kinds_used),
        is_unknown_case=(answer == UNKNOWN),
        is_aggregation=(len(direct_facts) > 1),
        world=world,
        query=query,
        shortcut_predictions=ontology_shortcut_predictions(world, query, taxonomy, nouns),
    )
