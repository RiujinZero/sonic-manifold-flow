"""``distractor_reasoning``'s ten synthetic generators as one SmolLM3-recipe
text source, framed as an ongoing multi-turn conversation.

Binding/Aggregation/Refusal share ``generate_example`` (behavior selected
internally by the sampled difficulty level and its ``allow_*`` flags); the
other seven generators from writing-block.md each own a dedicated function.
Each ``take()`` call builds one "session" of 2-4 generator examples framed
as User/Assistant turns (reusing ``distractor_reasoning.topic_switch``'s own
``Turn``/``render_turns`` convention), occasionally separated by either a
short hand-written transition phrase or a real excerpt pulled from SmolTalk/
LMSYS-Chat conversation data -- simulating how a user's questions actually
land amid an ordinary back-and-forth rather than arriving as an abrupt list
of unrelated Q&A pairs. Wrapping all of this behind the same
``.take(count) -> list[int]`` interface used by ``HFStreamingTextSource``/
``DataTroveTokenizedSource`` lets the existing weighted-mixture sampling in
``SmolLM3BatchProvider`` treat it as an ordinary source.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from distractor_reasoning import (
    CONVERSATION_STATE_DIFFICULTY_LEVELS,
    CORRECTION_DIFFICULTY_LEVELS,
    COUNTERFACTUAL_DIFFICULTY_LEVELS,
    DIFFICULTY_LEVELS,
    INSTRUCTION_DIFFICULTY_LEVELS,
    ONTOLOGY_DIFFICULTY_LEVELS,
    STATE_DIFFICULTY_LEVELS,
    TOPIC_SWITCH_DIFFICULTY_LEVELS,
    generate_conversation_state_example,
    generate_correction_example,
    generate_counterfactual_example,
    generate_example,
    generate_instruction_example,
    generate_ontology_example,
    generate_state_update_example,
    generate_topic_switch_example,
)
from distractor_reasoning.topic_switch import Turn, render_turns

from .data.smoltalk import DialogueTurn, iter_jsonl_conversations

DISTRACTOR_REASONING_SOURCE_NAME = "distractor-reasoning"

TurnRenderFn = Callable[[random.Random, int, str], "list[Turn]"]

_MIN_SESSION_EXAMPLES = 2
_MAX_SESSION_EXAMPLES = 4
# Between consecutive examples in a session: most of the time move straight
# to the next question (already realistic -- successive questions in one
# sitting), sometimes prefix a short transition phrase, occasionally splice
# in a brief excerpt of a real conversation as an off-topic aside.
_CONNECTIVE_PROBABILITY = 0.35
_ASIDE_PROBABILITY = 0.1
_ASIDE_MIN_TURNS = 2
_ASIDE_MAX_TURNS = 4

_CONNECTIVES: tuple[str, ...] = (
    "Okay, one more question.",
    "Thanks -- can I ask something else?",
    "Got it. Next question:",
    "Alright, switching topics for a second.",
    "One more thing --",
)


def load_chat_turn_pool(
    paths: Sequence[str | Path], *, max_conversations_per_file: int = 5_000,
) -> tuple[DialogueTurn, ...]:
    """Flatten SmolTalk/LMSYS-Chat JSONL file(s) into a pool of dialogue
    turns usable as realistic conversational filler between generator
    examples. Shares the SmolTalk JSONL schema/parser
    (``gator.data.smoltalk``) that already backs GTR-ARW's own chat runs.
    Bounded per file since the LMSYS-Chat export is multiple GB; a filler
    pool doesn't need exhaustive coverage.
    """
    turns: list[DialogueTurn] = []
    for path in paths:
        for conversation in iter_jsonl_conversations(
            path, max_conversations=max_conversations_per_file, skip_invalid=True,
        ):
            turns.extend(conversation.turns)
    return tuple(turns)


def _as_turn_pair(prompt: str, answer: str) -> list[Turn]:
    return [Turn("user", prompt), Turn("assistant", answer)]


def _turns_binding(rng: random.Random, level: int, split: str) -> list[Turn]:
    example = generate_example(rng, level, split=split, style="qa")
    return _as_turn_pair(example.prompt, example.answer_text)


def _turns_correction(rng: random.Random, level: int, split: str) -> list[Turn]:
    example = generate_correction_example(rng, level, split=split, style="qa")
    return _as_turn_pair(example.prompt, example.answer_text)


def _turns_state_update(rng: random.Random, level: int, split: str) -> list[Turn]:
    example = generate_state_update_example(rng, level, split=split, style="qa")
    return _as_turn_pair(example.prompt, example.answer_text)


def _turns_ontology(rng: random.Random, level: int, split: str) -> list[Turn]:
    example = generate_ontology_example(rng, level, split=split, style="qa")
    return _as_turn_pair(example.prompt, example.answer_text)


def _turns_instruction_selection(rng: random.Random, level: int, split: str) -> list[Turn]:
    example = generate_instruction_example(rng, level, split=split, style="qa")
    return _as_turn_pair(example.prompt, example.answer_text)


def _turns_conversation_state(rng: random.Random, level: int, split: str) -> list[Turn]:
    # No style parameter on this generator, unlike the others.
    example = generate_conversation_state_example(rng, level, split=split)
    return _as_turn_pair(example.prompt, example.answer_text)


def _turns_topic_switch(rng: random.Random, level: int, split: str) -> list[Turn]:
    # No style parameter on this generator, unlike the others. Already its
    # own multi-turn structure (ask, unrelated aside, resume) -- splice its
    # turns straight into the session rather than flattening to one pair.
    example = generate_topic_switch_example(rng, level, split=split)
    return list(example.turns)


def _turns_counterfactual(rng: random.Random, level: int, split: str) -> list[Turn]:
    # Pair consistency is an evaluation-time concept (evaluate.py scores
    # both halves of a minimal pair together); for a training session each
    # half is just another example, so emit one side per draw.
    example = generate_counterfactual_example(rng, level, split=split, style="qa")
    if rng.random() < 0.5:
        return _as_turn_pair(example.prompt_a, example.answer_text_a)
    return _as_turn_pair(example.prompt_b, example.answer_text_b)


# name -> (render_fn, max difficulty level for that generator's own table).
# Levels aren't uniform across generators (State-Update only defines 1-8),
# so each entry's ceiling comes from the generator's own *_DIFFICULTY_LEVELS
# dict rather than a hardcoded constant.
_GENERATORS: tuple[tuple[str, TurnRenderFn, int], ...] = (
    ("binding_aggregation_refusal", _turns_binding, max(DIFFICULTY_LEVELS)),
    ("correction", _turns_correction, max(CORRECTION_DIFFICULTY_LEVELS)),
    ("state_update", _turns_state_update, max(STATE_DIFFICULTY_LEVELS)),
    ("ontology", _turns_ontology, max(ONTOLOGY_DIFFICULTY_LEVELS)),
    ("instruction_selection", _turns_instruction_selection, max(INSTRUCTION_DIFFICULTY_LEVELS)),
    ("conversation_state", _turns_conversation_state, max(CONVERSATION_STATE_DIFFICULTY_LEVELS)),
    ("topic_switch", _turns_topic_switch, max(TOPIC_SWITCH_DIFFICULTY_LEVELS)),
    ("counterfactual", _turns_counterfactual, max(COUNTERFACTUAL_DIFFICULTY_LEVELS)),
)


class DistractorReasoningTextSource:
    """Live synthetic-generator source, uniform across all ten generators,
    packaged as multi-example conversation sessions rather than isolated
    Q&A blocks.

    Generation is synchronous CPU work (no network), so unlike
    ``HFStreamingTextSource`` this needs no prefetch thread -- each
    ``take()`` just tops up an internal token buffer until it can satisfy
    the request. Exact-resumable via the stdlib RNG's own state plus the
    leftover buffer, the same shape as the other sources' state_dict.
    """

    FORMAT = "gator-distractor-reasoning-stream-v2"
    exact_state_supported = True

    def __init__(
        self, spec: Any, tokenizer: Any, *, seed: int = 0,
        chat_turn_pool: Sequence[DialogueTurn] = (),
    ) -> None:
        self.spec = spec
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self._seed = int(seed)
        self._examples_generated = 0
        self._rng = random.Random(self._seed)
        self._buffer: list[int] = []
        self._chat_turn_pool = tuple(chat_turn_pool)

    def _sample_aside_turns(self) -> list[Turn]:
        pool = self._chat_turn_pool
        user_indices = [i for i, turn in enumerate(pool) if turn.role == "user"]
        if not user_indices:
            return []
        start = self._rng.choice(user_indices)
        length = self._rng.randint(_ASIDE_MIN_TURNS, _ASIDE_MAX_TURNS)
        excerpt = pool[start:start + length]
        return [Turn(turn.role, turn.text) for turn in excerpt]

    def _generate_session(self) -> list[int]:
        num_examples = self._rng.randint(_MIN_SESSION_EXAMPLES, _MAX_SESSION_EXAMPLES)
        turns: list[Turn] = []
        for index in range(num_examples):
            connective: str | None = None
            if index > 0:
                if self._chat_turn_pool and self._rng.random() < _ASIDE_PROBABILITY:
                    turns.extend(self._sample_aside_turns())
                elif self._rng.random() < _CONNECTIVE_PROBABILITY:
                    connective = self._rng.choice(_CONNECTIVES)
            _name, render_fn, max_level = self._rng.choice(_GENERATORS)
            level = self._rng.randint(1, max_level)
            example_turns = render_fn(self._rng, level, "train")
            if connective is not None and example_turns:
                first = example_turns[0]
                example_turns = [Turn(first.speaker, f"{connective} {first.text}")] + example_turns[1:]
            turns.extend(example_turns)
            self._examples_generated += 1
        text = render_turns(turns)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.eos_token_id)
        return ids

    def take(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("token count must be positive")
        while len(self._buffer) < count:
            self._buffer.extend(self._generate_session())
        result, self._buffer = self._buffer[:count], self._buffer[count:]
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "identity": {"name": self.spec.name, "seed": self._seed},
            "examples_generated": self._examples_generated,
            "buffer": list(self._buffer),
            "rng_state": self._rng.getstate(),
            "exact": True,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        identity = {"name": self.spec.name, "seed": self._seed}
        if state.get("format") != self.FORMAT or state.get("identity") != identity:
            raise ValueError(f"distractor-reasoning stream identity changed for {self.spec.name}")
        self._examples_generated = int(state["examples_generated"])
        self._buffer = list(state["buffer"])
        self._rng.setstate(state["rng_state"])

    def close(self) -> None:
        pass
