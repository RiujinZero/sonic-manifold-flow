"""Render the symbolic world/query/answer to natural-language text.

Rendering happens strictly *after* the symbolic world and the gold answer
have been decided (see ``generate.py``) -- text is never used to compute
anything, only to express what's already been decided.
"""

from __future__ import annotations

import random
from typing import Literal, Union

from .query import UNKNOWN, Query
from .vocab import (
    CATEGORY_PLURALS,
    Noun,
    answer_templates_for,
    answer_unknown_templates_for,
    having_templates_for,
    observe_templates_for,
    question_actual_templates_for,
    question_templates_for,
    reported_templates_for,
    zero_templates_for,
)
from .world import Fact


def _noun_phrase(noun: Noun, quantity: int) -> str:
    return noun.singular if quantity == 1 else noun.plural


def render_fact(fact: Fact, rng: random.Random, split: str, nouns: dict[str, Noun]) -> str:
    noun_info = nouns[fact.noun]
    if fact.is_reported:
        template = rng.choice(reported_templates_for(split))
        return template.format(
            speaker=fact.reported_by, subject=fact.subject, relation=fact.relation,
            quantity=fact.quantity, noun_phrase=_noun_phrase(noun_info, fact.quantity),
        )
    if fact.is_having and fact.quantity == 0:
        template = rng.choice(zero_templates_for(split))
        return template.format(subject=fact.subject, noun_phrase=noun_info.plural)
    template_pool = having_templates_for(split) if fact.is_having else observe_templates_for(split)
    template = rng.choice(template_pool)
    return template.format(
        subject=fact.subject, relation=fact.relation, quantity=fact.quantity,
        noun_phrase=_noun_phrase(noun_info, fact.quantity),
    )


def render_question(
    query: Query, rng: random.Random, split: str, nouns: dict[str, Noun],
    category_plurals: dict[str, str] = CATEGORY_PLURALS,
) -> str:
    noun_phrase = (
        category_plurals[query.target] if query.is_category else nouns[query.target].plural
    )
    templates = (
        question_actual_templates_for(split) if query.disambiguates_reported
        else question_templates_for(split)
    )
    return rng.choice(templates).format(subject=query.subject, noun_phrase=noun_phrase)


def render_answer(
    query: Query,
    value: Union[int, Literal["UNKNOWN"]],
    rng: random.Random,
    split: str,
    nouns: dict[str, Noun],
    style: str = "qa",
    category_plurals: dict[str, str] = CATEGORY_PLURALS,
) -> str:
    """A full conversational sentence, not a bare number/UNKNOWN token."""

    plural_phrase = (
        category_plurals[query.target] if query.is_category else nouns[query.target].plural
    )
    if value == UNKNOWN:
        template = rng.choice(answer_unknown_templates_for(split))
        text = template.format(subject=query.subject, noun_phrase=plural_phrase)
    else:
        noun_phrase = (
            plural_phrase if query.is_category or value != 1
            else nouns[query.target].singular
        )
        template = rng.choice(answer_templates_for(split))
        text = template.format(subject=query.subject, quantity=value, noun_phrase=noun_phrase)
    return f"A: {text}" if style == "qa" else f"Answer: {text}"


def render_example(
    facts: list[Fact],
    query: Query,
    answer: Union[int, Literal["UNKNOWN"]],
    rng: random.Random,
    split: str,
    nouns: dict[str, Noun],
    style: str = "qa",
    category_plurals: dict[str, str] = CATEGORY_PLURALS,
) -> tuple[str, str]:
    """Return ``(prompt_text, answer_text)``. Fact order is shuffled here
    (the symbolic world itself carries no ordering).

    ``category_plurals`` defaults to the Binding generator's fixed
    real-world category names; the Ontology generator passes its own
    per-split synthetic taxonomy's plural map instead.
    """

    ordered = list(facts)
    rng.shuffle(ordered)
    context = "\n".join(render_fact(f, rng, split, nouns) for f in ordered)
    question = render_question(query, rng, split, nouns, category_plurals=category_plurals)
    question_line = f"Q: {question}" if style == "qa" else question
    prompt = f"{context}\n{question_line}"
    return prompt, render_answer(
        query, answer, rng, split, nouns, style=style, category_plurals=category_plurals,
    )
