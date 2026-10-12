"""State-Update Generator: track a changing quantity across a *sequence*
of events, rather than the Binding generator's static snapshot of facts.

    Kevin has 5 parrots.
    Kevin receives 3 parrots.
    Kevin gives Maria 2 parrots.

    Q: How many parrots does Kevin have?
    A: 6

The gold answer is always the result of literally replaying every event in
order against a running per-(subject, noun) total (``resolve_state_query``)
-- never guessed from surface text, matching the rest of this package's
symbolic-world-first philosophy. An initial ``has``/``owns``/``keeps``
sentence *sets* the running total for that (subject, noun) pair; every
other event applies a signed delta. Paired events (give/receive/transfer/
move) update *both* parties' running totals atomically, since a shortcut
that only credits one side of a transfer is exactly one of the failure
modes this generator exists to catch (``ignore_ownership_transfer`` in
``state_shortcuts.py``).

A (subject, noun) pair that is never touched by any event resolves to
UNKNOWN, the same discipline as the Binding generator's query resolution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

from .query import UNKNOWN, Query
from .render import render_question
from .vocab import HAVING_RELATIONS, OBSERVE_RELATIONS, Noun, names_for, nouns_for

# state_shortcuts imports Event/StateQuery/verb-constants from this module,
# so it's imported lazily inside generate_state_update_example to avoid a
# circular import at module-load time.

# -- Event verbs --------------------------------------------------------
# Single-party: only the named subject's running total changes.
INCREASE_SINGLE_VERBS: tuple[str, ...] = ("buys", "finds", "adds")
DECREASE_SINGLE_VERBS: tuple[str, ...] = ("sells", "loses", "removes")
# Paired: the named subject's total changes, and the counterparty's total
# changes by the opposite amount (a real transfer of ownership).
INCREASE_PAIRED_VERBS: tuple[str, ...] = ("receives",)
DECREASE_PAIRED_VERBS: tuple[str, ...] = ("gives", "transfers", "moves")

INCREASE_VERBS: tuple[str, ...] = INCREASE_SINGLE_VERBS + INCREASE_PAIRED_VERBS
DECREASE_VERBS: tuple[str, ...] = DECREASE_SINGLE_VERBS + DECREASE_PAIRED_VERBS
PAIRED_VERBS: tuple[str, ...] = INCREASE_PAIRED_VERBS + DECREASE_PAIRED_VERBS
EVENT_VERBS: tuple[str, ...] = INCREASE_VERBS + DECREASE_VERBS


@dataclass(frozen=True)
class Event:
    """One line of the sequence.

    ``verb`` is either a having-relation (``has``/``owns``/``keeps``,
    which *sets* the running total rather than applying a delta -- used
    only for the initial-state sentence), one of ``EVENT_VERBS``, or an
    observation relation (``photographed``/``counted``/... -- never
    changes state at all; a pure "same noun under another relation"
    distractor). ``counterparty`` is set only for ``PAIRED_VERBS``.
    """

    verb: str
    subject: str
    noun: str
    quantity: int
    counterparty: Optional[str] = None

    @property
    def is_initial(self) -> bool:
        return self.verb in HAVING_RELATIONS

    @property
    def is_state_changing(self) -> bool:
        return self.verb in EVENT_VERBS

    @property
    def signed_delta(self) -> int:
        return self.quantity if self.verb in INCREASE_VERBS else -self.quantity


@dataclass(frozen=True)
class StateQuery:
    subject: str
    noun: str


def resolve_state_query(
    query: StateQuery, events: Sequence[Event]
) -> Union[int, Literal["UNKNOWN"]]:
    """Replay every event in order; return the final running total for
    ``(query.subject, query.noun)``, or UNKNOWN if that pair is never
    touched by any initial-state or event line."""

    key = (query.subject, query.noun)
    total = 0
    touched = False
    for event in events:
        if event.is_initial:
            if (event.subject, event.noun) == key:
                total, touched = event.quantity, True
            continue
        if not event.is_state_changing:
            continue  # observation-only line: never changes state
        delta = event.signed_delta
        if (event.subject, event.noun) == key:
            total += delta
            touched = True
        if event.counterparty is not None and (event.counterparty, event.noun) == key:
            total -= delta
            touched = True
    return total if touched else UNKNOWN


# -- Distractor event generators -----------------------------------------
# Mirrors distractors.py's naming/labeling convention: each returns an
# Event guaranteed never to touch the query's own (subject, noun) pair,
# labeled for evaluation bookkeeping only (never rendered).

def _quantity(rng: random.Random) -> int:
    return rng.randint(1, 9)


def make_other_subject_event(
    subject: str, noun: str, rng: random.Random, nouns: dict, names: tuple[str, ...],
) -> Event:
    """Another person's animals -- irrelevant subject, same or different
    noun. Increase-only and never paired: a distractor's random subject
    can coincidentally collide with a real counterparty introduced
    elsewhere in the sequence, and an uncapped decrease or an
    unaccompanied paired verb (no counterparty to credit/debit here)
    would corrupt that person's *real* running total."""

    other_subject = rng.choice([n for n in names if n != subject])
    verb = rng.choice(INCREASE_SINGLE_VERBS)
    return Event(verb, other_subject, noun, _quantity(rng))


def make_other_noun_event(
    subject: str, noun: str, rng: random.Random, nouns: dict, names: tuple[str, ...],
) -> Event:
    """Another noun owned by the query's own subject -- same person, but
    the event can never contribute to *this* noun's total. Increase-only
    for the same reason as ``make_other_subject_event``."""

    other_noun = rng.choice([n for n in nouns if n != noun])
    verb = rng.choice(INCREASE_SINGLE_VERBS)
    return Event(verb, subject, other_noun, _quantity(rng))


def make_observation_event(
    subject: str, noun: str, rng: random.Random, nouns: dict, names: tuple[str, ...],
) -> Event:
    """Same subject, same noun, but an observation relation -- looks like
    a state-changing line but never is one. The line a naive
    ``last_number`` shortcut would latch onto."""

    return Event(rng.choice(OBSERVE_RELATIONS), subject, noun, _quantity(rng))


STATE_DISTRACTOR_MAKERS = {
    "other_subject": make_other_subject_event,
    "other_noun": make_other_noun_event,
    "observation": make_observation_event,
}
ALL_STATE_DISTRACTOR_KINDS: tuple[str, ...] = tuple(STATE_DISTRACTOR_MAKERS.keys())


# -- Difficulty curriculum -------------------------------------------------

@dataclass(frozen=True)
class StateDifficultyConfig:
    level: int
    min_events: int = 1
    max_events: int = 2
    allow_decrease: bool = False
    allow_paired: bool = False
    distractor_kinds: tuple[str, ...] = ()
    min_distractors: int = 0
    max_distractors: int = 0
    query_counterparty_probability: float = 0.0
    use_inverse_pair: bool = False
    answerable_probability: float = 1.0

    @property
    def allow_distractors(self) -> bool:
        return self.max_distractors > 0


STATE_DIFFICULTY_LEVELS: dict[int, StateDifficultyConfig] = {
    1: StateDifficultyConfig(level=1, min_events=1, max_events=2),
    2: StateDifficultyConfig(level=2, min_events=1, max_events=3, allow_decrease=True),
    3: StateDifficultyConfig(
        level=3, min_events=1, max_events=3, allow_decrease=True,
        distractor_kinds=("other_subject", "other_noun"),
        min_distractors=1, max_distractors=2,
    ),
    4: StateDifficultyConfig(
        level=4, min_events=2, max_events=4, allow_decrease=True,
        distractor_kinds=ALL_STATE_DISTRACTOR_KINDS,
        min_distractors=1, max_distractors=3,
    ),
    5: StateDifficultyConfig(
        level=5, min_events=2, max_events=4, allow_decrease=True, allow_paired=True,
        distractor_kinds=ALL_STATE_DISTRACTOR_KINDS,
        min_distractors=1, max_distractors=3,
        query_counterparty_probability=0.5,
    ),
    6: StateDifficultyConfig(
        level=6, min_events=3, max_events=5, allow_decrease=True, allow_paired=True,
        distractor_kinds=ALL_STATE_DISTRACTOR_KINDS,
        min_distractors=1, max_distractors=3,
        query_counterparty_probability=0.5, use_inverse_pair=True,
    ),
    7: StateDifficultyConfig(
        level=7, min_events=2, max_events=5, allow_decrease=True, allow_paired=True,
        distractor_kinds=ALL_STATE_DISTRACTOR_KINDS,
        min_distractors=1, max_distractors=3,
        query_counterparty_probability=0.5, use_inverse_pair=True,
        answerable_probability=0.5,
    ),
    8: StateDifficultyConfig(
        level=8, min_events=4, max_events=7, allow_decrease=True, allow_paired=True,
        distractor_kinds=ALL_STATE_DISTRACTOR_KINDS,
        min_distractors=2, max_distractors=4,
        query_counterparty_probability=0.5, use_inverse_pair=True,
        answerable_probability=0.5,
    ),
}


def state_config_for_level(level: int) -> StateDifficultyConfig:
    if level not in STATE_DIFFICULTY_LEVELS:
        raise ValueError(f"state-update difficulty level must be in 1..8, got {level}")
    return STATE_DIFFICULTY_LEVELS[level]


# -- Rendering --------------------------------------------------------------

def _noun_phrase(noun_info: Noun, quantity: int) -> str:
    return noun_info.singular if quantity == 1 else noun_info.plural


def render_state_event(event: Event, nouns: dict[str, Noun]) -> str:
    noun_info = nouns[event.noun]
    phrase = _noun_phrase(noun_info, event.quantity)
    if event.is_initial:
        return f"{event.subject} {event.verb} {event.quantity} {phrase}."
    if event.verb == "receives":
        return f"{event.subject} receives {event.quantity} {phrase} from {event.counterparty}."
    if event.verb == "gives":
        return f"{event.subject} gives {event.counterparty} {event.quantity} {phrase}."
    if event.verb == "transfers":
        return f"{event.subject} transfers {event.quantity} {phrase} to {event.counterparty}."
    if event.verb == "moves":
        return f"{event.subject} moves {event.quantity} {phrase} to {event.counterparty}."
    return f"{event.subject} {event.verb} {event.quantity} {phrase}."


# Mirrors vocab.py's answer_templates_for/answer_unknown_templates_for --
# a separate, small pool since the "now has"/"ends up with" phrasing here
# is specific to a *sequence* of events rather than a static snapshot.
# The unknown templates share evaluate.py's UNKNOWN_INDICATORS phrases so
# the same parser works for every generator in the package.
STATE_ANSWER_TEMPLATES: tuple[str, ...] = (
    "{subject} now has {quantity} {noun_phrase}.",
    "{subject} ends up with {quantity} {noun_phrase}.",
)
STATE_ANSWER_UNKNOWN_TEMPLATES: tuple[str, ...] = (
    "I don't have enough information to know how many {noun_phrase} {subject} has.",
    "There's nothing here that tells me how many {noun_phrase} {subject} ends up with.",
)


def render_state_answer(
    query: StateQuery, value: Union[int, Literal["UNKNOWN"]],
    rng: random.Random, nouns: dict[str, Noun], style: str = "qa",
) -> str:
    """A full conversational sentence, not a bare number/UNKNOWN token."""

    noun_info = nouns[query.noun]
    if value == UNKNOWN:
        template = rng.choice(STATE_ANSWER_UNKNOWN_TEMPLATES)
        text = template.format(subject=query.subject, noun_phrase=noun_info.plural)
    else:
        noun_phrase = noun_info.singular if value == 1 else noun_info.plural
        template = rng.choice(STATE_ANSWER_TEMPLATES)
        text = template.format(subject=query.subject, quantity=value, noun_phrase=noun_phrase)
    return f"A: {text}" if style == "qa" else f"Answer: {text}"


# -- Example assembly ---------------------------------------------------

@dataclass
class StateExample:
    prompt: str
    answer_text: str
    answer_value: Union[int, Literal["UNKNOWN"]]
    difficulty: int
    num_events: int
    num_distractors: int
    distractor_kinds: tuple[str, ...]
    queries_counterparty: bool
    has_inverse_pair: bool
    is_unknown_case: bool
    events: list[Event]
    query: StateQuery
    shortcut_predictions: dict[str, int]


def generate_state_update_example(
    rng: random.Random, level: int, split: str = "train", style: str = "qa",
) -> StateExample:
    config = state_config_for_level(level)
    nouns = nouns_for(split)
    names = names_for(split)

    subject = rng.choice(names)
    noun = rng.choice(list(nouns.keys()))

    initial_quantity = rng.randint(1, 12)
    events: list[Event] = [Event(rng.choice(HAVING_RELATIONS), subject, noun, initial_quantity)]
    counterparty_pool = [n for n in names if n != subject]
    introduced_counterparties: list[str] = []
    # Tracks the query subject's own running total for (subject, noun) as
    # events are generated, so a decrease/give/sell/etc. can never be
    # asked to remove more than the subject actually has at that point --
    # otherwise the sequence can produce a nonsensical negative gold
    # answer ("Kevin now has -5 parrots.").
    running_total = initial_quantity

    num_events = rng.randint(config.min_events, config.max_events)
    if config.use_inverse_pair and num_events >= 2:
        pair_quantity = rng.randint(1, min(8, running_total))
        events.append(Event(rng.choice(INCREASE_SINGLE_VERBS), subject, noun, pair_quantity))
        events.append(Event(rng.choice(DECREASE_SINGLE_VERBS), subject, noun, pair_quantity))
        num_events -= 2

    for _ in range(max(0, num_events)):
        verb_pool = list(INCREASE_SINGLE_VERBS)
        if config.allow_paired and counterparty_pool:
            verb_pool += list(INCREASE_PAIRED_VERBS)
        if running_total > 0:
            if config.allow_decrease:
                verb_pool += list(DECREASE_SINGLE_VERBS)
            if config.allow_paired and counterparty_pool:
                verb_pool += list(DECREASE_PAIRED_VERBS)
        verb = rng.choice(verb_pool)
        quantity = rng.randint(1, min(9, running_total)) if verb in DECREASE_VERBS else _quantity(rng)
        running_total += quantity if verb in INCREASE_VERBS else -quantity
        if verb in PAIRED_VERBS:
            counterparty = rng.choice(counterparty_pool)
            introduced_counterparties.append(counterparty)
            events.append(Event(verb, subject, noun, quantity, counterparty=counterparty))
        else:
            events.append(Event(verb, subject, noun, quantity))

    # A counterparty's own running total isn't capped during generation
    # above (only the primary subject's is) -- e.g. "receives ... from
    # Maria" implicitly debits Maria without her ever having an
    # established baseline, which can go negative. Filter to
    # counterparties whose *actual* resolved total (via the real
    # resolver, not a re-derived approximation) is non-negative before
    # allowing a query to target one.
    queryable_counterparties = sorted({
        c for c in introduced_counterparties
        if isinstance(resolve_state_query(StateQuery(c, noun), events), int)
        and resolve_state_query(StateQuery(c, noun), events) >= 0
    })

    unknown_case = rng.random() >= config.answerable_probability
    queries_counterparty = (
        not unknown_case
        and bool(queryable_counterparties)
        and rng.random() < config.query_counterparty_probability
    )
    if unknown_case:
        # Force a genuinely untouched (subject, noun) pair: a subject that
        # never appears as either the primary actor or a counterparty in
        # any event, paired with a noun no event ever mentions.
        untouched_subjects = [
            n for n in names if n != subject and n not in introduced_counterparties
        ]
        query_subject = rng.choice(untouched_subjects or [n for n in names if n != subject])
        query_noun = rng.choice([n for n in nouns if n != noun])
    elif queries_counterparty:
        query_subject = rng.choice(queryable_counterparties)
        query_noun = noun
    else:
        query_subject = subject
        query_noun = noun

    query = StateQuery(subject=query_subject, noun=query_noun)

    # distractors
    distractor_events: list[Event] = []
    kinds_used: list[str] = []
    if config.allow_distractors:
        count = rng.randint(config.min_distractors, config.max_distractors)
        available = list(config.distractor_kinds)
        for _ in range(count):
            if not available:
                break
            kind = rng.choice(available)
            maker = STATE_DISTRACTOR_MAKERS[kind]
            distractor_events.append(maker(subject, noun, rng, nouns, names))
            kinds_used.append(kind)

    # Chronological order matters for the resolver's semantics (the
    # initial-state line must stay first among the real events), so only
    # distractor *placement* is randomized -- they're interleaved among
    # the true events at random positions, never reordering the events
    # themselves.
    ordered_events = list(events)
    for distractor in distractor_events:
        pos = rng.randint(0, len(ordered_events))
        ordered_events.insert(pos, distractor)

    answer = resolve_state_query(query, ordered_events)

    context = "\n".join(render_state_event(e, nouns) for e in ordered_events)
    question = render_question(
        Query(subject=query.subject, target=query.noun, is_category=False),
        rng, split, nouns,
    )
    question_line = f"Q: {question}" if style == "qa" else question
    prompt = f"{context}\n{question_line}"
    answer_text = render_state_answer(query, answer, rng, nouns, style=style)

    from .state_shortcuts import state_shortcut_predictions

    return StateExample(
        prompt=prompt,
        answer_text=answer_text,
        answer_value=answer,
        difficulty=level,
        num_events=len(ordered_events),
        num_distractors=len(distractor_events),
        distractor_kinds=tuple(kinds_used),
        queries_counterparty=queries_counterparty,
        has_inverse_pair=config.use_inverse_pair,
        is_unknown_case=(answer == UNKNOWN),
        events=ordered_events,
        query=query,
        shortcut_predictions=state_shortcut_predictions(ordered_events, query),
    )
