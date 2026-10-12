"""Topic-Switch Generator: suspend one task, engage with unrelated small
talk, and correctly resume the original task without contamination.

    User: Kevin has 4 cats. Maria has 8 parrots.
    Q: How many parrots does Kevin have?
    Assistant: Maria has 8 parrots.

    User: Anyway, how's it going?
    Assistant: I'm doing well, thanks for asking!

    User: Sorry, back to what we were talking about -- can you answer
    that again?
    Assistant: Maria has 8 parrots.

"Task A" is literally one example from an existing single-turn generator
(Binding, State-Update, Ontology, or Correction) -- its prompt becomes the
first User turn and its answer becomes the first Assistant turn,
unmodified. This directly validates the design doc's own cross-generator
note that generators can be composed: nothing new was needed to build
"task A" here, only the wrapper around it. "Topic B" is a small fixed bank
of generic, entity-free chit-chat exchanges, guaranteed disjoint from
every generator's name/noun vocabulary so any leakage into a topic-B
response is unambiguous contamination rather than coincidence. The
"return to A" turn re-poses task A's own question (a deliberate scope
simplification -- generating a genuinely *different* follow-up question
per source type would need dedicated per-generator work) and checks
whether the model still gives task A's correct answer after the
interruption.

Unlike this package's single-answer generators, evaluation here mirrors
the design doc's own "Detection" section directly: switch correctness,
contamination rate, and resume accuracy are evaluated as separate rates
across probed turns (see ``topic_switch_evaluate.py``), not a single
per-example accuracy number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .correction import generate_correction_example
from .generate import generate_example
from .ontology import generate_ontology_example
from .state_update import generate_state_update_example

Speaker = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    text: str


def render_turns(turns: list[Turn]) -> str:
    return "\n".join(f"{t.speaker.capitalize()}: {t.text}" for t in turns)


def _strip_answer_prefix(text: str) -> str:
    for prefix in ("Answer:", "A:"):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


# -- Task A sources -------------------------------------------------------
# Every one of these already exposes .prompt, .answer_text, and
# .query.subject -- no adapter needed beyond that uniform shape.
TASK_A_GENERATORS: dict[str, Callable] = {
    "binding": generate_example,
    "state_update": generate_state_update_example,
    "ontology": generate_ontology_example,
    "correction": generate_correction_example,
}
TASK_A_LEVELS: dict[str, int] = {
    "binding": 5, "state_update": 4, "ontology": 3, "correction": 3,
}


# -- Topic B: fixed, entity-free chit-chat bank ----------------------------
# None of these mention any name/noun from vocab.py/ontology.py's pools,
# so any of those words appearing in a model's reply here is unambiguous
# contamination, never coincidence.
TOPIC_B_BANK: tuple[tuple[str, str], ...] = (
    ("Anyway, how's it going?", "I'm doing well, thanks for asking!"),
    ("What's the weather like today?", "I don't have real-time weather access, but I hope it's nice where you are!"),
    ("Tell me a fun fact.", "Sure -- octopuses have three hearts!"),
    ("Do you like music?", "I don't have personal preferences, but I'm happy to chat about music!"),
    ("Thanks for your help earlier.", "You're welcome!"),
    ("Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything!"),
    ("How's your day going?", "I'm here and ready to help, thanks for asking!"),
    ("What do you think about robots?", "I think they're pretty interesting -- I suppose I'm one myself!"),
    ("Do you ever get tired?", "Not in the way people do -- I'm ready to help whenever you need."),
    ("What's your favorite season?", "I don't experience seasons, but I hear autumn is popular!"),
)

RESUME_TEMPLATES: tuple[str, ...] = (
    "Sorry, back to what we were talking about -- can you answer that again?",
    "Anyway, going back to {subject} -- what was the answer to my earlier question?",
    "OK, back to the earlier question about {subject}.",
)

CONTEXT_RESET_RESPONSE = "I'm sorry, I don't have enough context to answer that."


# -- Difficulty curriculum -------------------------------------------------

@dataclass(frozen=True)
class TopicSwitchDifficultyConfig:
    level: int
    task_sources: tuple[str, ...]
    min_b_exchanges: int = 1
    max_b_exchanges: int = 1


TOPIC_SWITCH_DIFFICULTY_LEVELS: dict[int, TopicSwitchDifficultyConfig] = {
    1: TopicSwitchDifficultyConfig(level=1, task_sources=("binding",), min_b_exchanges=1, max_b_exchanges=1),
    2: TopicSwitchDifficultyConfig(
        level=2, task_sources=("binding", "state_update"), min_b_exchanges=1, max_b_exchanges=2,
    ),
    3: TopicSwitchDifficultyConfig(
        level=3, task_sources=("binding", "state_update", "ontology", "correction"),
        min_b_exchanges=2, max_b_exchanges=3,
    ),
}


def topic_switch_config_for_level(level: int) -> TopicSwitchDifficultyConfig:
    if level not in TOPIC_SWITCH_DIFFICULTY_LEVELS:
        raise ValueError(f"topic-switch difficulty level must be in 1..3, got {level}")
    return TOPIC_SWITCH_DIFFICULTY_LEVELS[level]


# -- Example assembly ---------------------------------------------------

@dataclass(frozen=True)
class TurnProbe:
    """Everything needed to evaluate one assistant turn in isolation: the
    conversation prefix up to (and ending with) the "Assistant:" tag that
    should precede it, the gold reply text, and (where well-defined)
    shortcut predictions for that specific turn."""

    label: str  # "topic_b" | "resume"
    prompt: str
    gold_text: str
    shortcut_predictions: dict[str, str] = field(default_factory=dict)


@dataclass
class TopicSwitchExample:
    turns: list[Turn]
    full_text: str
    task_a_source: str
    task_a_subject: str
    num_b_exchanges: int
    probes: list[TurnProbe]


def generate_topic_switch_example(
    rng: random.Random, level: int, split: str = "train",
) -> TopicSwitchExample:
    config = topic_switch_config_for_level(level)
    source = rng.choice(config.task_sources)
    task_a = TASK_A_GENERATORS[source](rng, TASK_A_LEVELS[source], split=split)
    a_question_text = task_a.prompt
    a_answer_text = _strip_answer_prefix(task_a.answer_text)
    subject = task_a.query.subject

    turns: list[Turn] = [Turn("user", a_question_text), Turn("assistant", a_answer_text)]

    num_b_exchanges = rng.randint(config.min_b_exchanges, config.max_b_exchanges)
    b_pairs = rng.sample(TOPIC_B_BANK, k=num_b_exchanges)

    topic_b_prompt = render_turns(turns) + f"\nUser: {b_pairs[0][0]}\nAssistant:"
    topic_b_probe = TurnProbe(
        label="topic_b", prompt=topic_b_prompt, gold_text=b_pairs[0][1],
        shortcut_predictions={"previous_task_persistence": a_answer_text},
    )

    for b_user, b_assistant in b_pairs:
        turns.append(Turn("user", b_user))
        turns.append(Turn("assistant", b_assistant))

    resume_line = rng.choice(RESUME_TEMPLATES).format(subject=subject)
    resume_prompt = render_turns(turns) + f"\nUser: {resume_line}\nAssistant:"
    resume_probe = TurnProbe(
        label="resume", prompt=resume_prompt, gold_text=a_answer_text,
        shortcut_predictions={"context_reset": CONTEXT_RESET_RESPONSE},
    )
    turns.append(Turn("user", resume_line))
    turns.append(Turn("assistant", a_answer_text))

    return TopicSwitchExample(
        turns=turns,
        full_text=render_turns(turns),
        task_a_source=source,
        task_a_subject=subject,
        num_b_exchanges=num_b_exchanges,
        probes=[topic_b_probe, resume_probe],
    )
