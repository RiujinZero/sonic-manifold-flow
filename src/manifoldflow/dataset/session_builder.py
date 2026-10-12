"""Pregenerated episodic session builder for the production ManifoldFlow
training recipe (see the approved plan at
`C:\\Users\\user\\.claude\\plans\\rustling-nibbling-flute.md` and
`docs/Sonic-MF Training Design v0.1.md`).

Generation must happen offline, ahead of training, not live - the
96-lane batch-lane persistence probe earlier this session was CPU-bound
on live episode generation (a serial Python loop per lane per step), and
this module is explicitly the fix: pretokenized session shards written to
disk once, read cheaply many times during training.

A **session** is a chronological sequence of **episodes** of three types:

- ``"distractor"``: a `distractor_reasoning`-generated correction-chain
  unit (the same generator every probe this session used) - some
  statements plus, for episodes chosen to carry a query, a rendered
  question/answer. Reuses `_build_episode` directly from
  `sonic_reverse_query_probe.py` (quantity + name targets) and
  `sonic_noun_target_probe.py` (quantity + noun targets), picked at
  random per unit, so the encoder sees all three target types already
  validated this session rather than overfitting to one shape.
- ``"chat"``: multi-turn conversational text, read from local pre-rendered
  JSONL exports (smoltalk + lmsys-chat-1m) on the G: drive - see
  `ChatTurnStream`'s own docstring. The plan's original research found no
  local JSONL chat export in either *repo* and defaulted to streaming
  smoltalk from the HF Hub instead; that export does exist, just outside
  both repos, and using it sidesteps the HF-Hub streaming flakiness this
  session hit directly (timeouts/disconnects reproduced while testing the
  original HF-streaming version of this class).
- ``"filler"``: genuinely unrelated prose, streamed via the same
  `HFStreamingTextSource`/`SourceSpec` this project already vendored
  from `GTR-sonic-ssm/gator/smollm3_recipe.py`, reused unmodified.

Not every distractor unit's fact gets queried immediately - some are
queried later in the session (a session-level pending-fact pool, the same
"current vs. prior" idea `sonic_batch_lane_persistence_probe.py` already
validated, generalized across many episodes instead of one lane's own
history) - and some are never queried at all
(`Sonic-MF Training Design v0.1.md` §22: "not every fact gets queried").
Physical packing then slices the assembled, tokenized session into
sequential 8192-token windows (§30) - target placement within a window
falls out of episode interleaving + deferred-query placement, without a
separate randomization pass.
"""

from __future__ import annotations

import json
import queue
import random
import re
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"
sys.path.insert(0, str(_VENDOR_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

PHYSICAL_WINDOW_TOKENS = 8192

EpisodeType = Literal["distractor", "chat", "filler"]


# --------------------------------------------------------------------------
# Distractor units - reuse the exact, already-proven episode builders.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DistractorUnit:
    """One `distractor_reasoning` correction-chain unit, target-kind-agnostic.

    Mirrors `sonic_reverse_query_probe.Episode`/`sonic_noun_target_probe.Episode`
    exactly (same field shapes), just wrapped so the session builder doesn't
    care which of the two source files produced it."""

    statement_ids: list[list[int]]
    statement_value_spans: list[tuple[int, int]]
    target_statement_index: int
    # Forward (quantity) question/answer - always present.
    quantity_question_ids: list[int]
    quantity_answer_ids: list[int]
    quantity_value_span: tuple[int, int]
    # Alternate target (name or noun, depending on source) - always present.
    alt_target_kind: Literal["name", "noun"]
    alt_question_ids: list[int]
    alt_answer_ids: list[int]


def _render_statement_pieces(
    statement: Any, generator: random.Random, nouns: Any, relation: str
) -> tuple[str, str, str]:
    """Reimplements `render_statement`'s exact template logic - identical
    to every probe's own copy of this function this session
    (`sonic_reverse_query_probe.py`, `sonic_noun_target_probe.py`), kept
    self-contained here rather than importing those probe modules: they
    each pull in `manifoldflow.sonic_ssm`'s full `SonicSSMLanguageModel`
    class (for their own training-loop use, irrelevant to pregeneration)
    which in turn needs `gator_linear._hgrn_compiled` -> `gator_convergent`,
    a module not available in every environment this pipeline runs in -
    reimplementing this one small function avoids that entirely unrelated
    heavy import chain just to reuse a template renderer."""
    from distractor_reasoning.correction import (
        CORRECTION_TEMPLATES,
        HAVING_RELATIONS_PLURAL,
        INITIAL_TEMPLATES,
        OBSERVATION_TEMPLATES,
    )

    noun_info = nouns[statement.noun]
    phrase = noun_info.singular if statement.quantity == 1 else noun_info.plural
    if statement.kind == "decoy_observation":
        template = generator.choice(OBSERVATION_TEMPLATES)
        before, after = template.split("{quantity}")
        return before, str(statement.quantity), after.format(
            noun_phrase=phrase, subject=statement.subject
        )

    subject_text = "They" if statement.pronoun else statement.subject
    rel = HAVING_RELATIONS_PLURAL[relation] if statement.pronoun else relation
    template_pool = (
        CORRECTION_TEMPLATES
        if statement.kind in ("correction", "decoy_correction")
        else INITIAL_TEMPLATES
    )
    template = generator.choice(template_pool)
    before, after = template.split("{quantity}")
    return (
        before.format(subject=subject_text, relation=rel),
        str(statement.quantity),
        after.format(noun_phrase=phrase),
    )


def _build_distractor_unit(
    generator: random.Random, tokenizer: Any, *, level: int
) -> DistractorUnit | None:
    """Self-contained reimplementation of `sonic_reverse_query_probe.py`'s
    and `sonic_noun_target_probe.py`'s `_build_episode` (same generator
    call, same rendering, same ambiguity checks - see
    `_render_statement_pieces`'s docstring for why this isn't just an
    import of those files), unified so a *single* generated episode
    always carries all three target kinds (quantity/name/noun) instead of
    needing two separate generator calls to get all three."""
    from distractor_reasoning.correction import (
        HAVING_RELATIONS_PLURAL,
        generate_correction_example,
    )
    from distractor_reasoning.query import UNKNOWN
    from distractor_reasoning.vocab import HAVING_RELATIONS, nouns_for

    nouns = nouns_for("train")
    example = generate_correction_example(generator, level=level, split="train")
    if example.answer_value == UNKNOWN:
        return None

    relation = next(
        candidate
        for candidate in HAVING_RELATIONS
        if f" {candidate} " in example.answer_text
    )

    statement_ids: list[list[int]] = []
    statement_value_spans: list[tuple[int, int]] = []
    for statement in example.statements:
        prefix_text, quantity_text, suffix_text = _render_statement_pieces(
            statement, generator, nouns, relation
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
        quantity_ids = tokenizer.encode(quantity_text, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        statement_ids.append(prefix_ids + quantity_ids + suffix_ids)
        statement_value_spans.append(
            (len(prefix_ids), len(prefix_ids) + len(quantity_ids))
        )

    target_key = (example.query.subject, example.query.target)
    target_index = None
    for index, statement in enumerate(example.statements):
        if statement.kind == "decoy_observation":
            continue
        if (statement.subject, statement.noun) == target_key:
            target_index = index
    if target_index is None:
        return None

    quantity = example.answer_value
    target_statement = example.statements[target_index]
    # Name-target well-posedness (sonic_reverse_query_probe.py's check):
    # reject if the quantity isn't unique across the whole episode.
    other_quantities_any_subject = [
        statement.quantity
        for statement in example.statements
        if statement.kind != "decoy_observation" and statement is not target_statement
    ]
    if quantity in other_quantities_any_subject:
        return None
    # Noun-target well-posedness (sonic_noun_target_probe.py's check):
    # reject if the quantity isn't unique among this subject's own
    # statements specifically.
    same_subject_quantities = [
        statement.quantity
        for statement in example.statements
        if statement.kind != "decoy_observation"
        and statement is not target_statement
        and statement.subject == example.query.subject
    ]
    if quantity in same_subject_quantities:
        return None

    noun_info = nouns[example.query.target]
    noun_phrase = noun_info.singular if quantity == 1 else noun_info.plural
    relation_base = HAVING_RELATIONS_PLURAL[relation]

    quantity_question_text = (
        f"Q: How many {noun_info.plural} "
        f"does {example.query.subject} {relation_base}?"
    )
    prefix_ids = tokenizer.encode(
        f"\nA: {example.query.subject} {relation} ", add_special_tokens=False
    )
    quantity_ids = tokenizer.encode(str(quantity), add_special_tokens=False)
    suffix_ids = tokenizer.encode(f" {noun_phrase}.", add_special_tokens=False)
    quantity_answer_ids = prefix_ids + quantity_ids + suffix_ids
    quantity_value_span = (len(prefix_ids), len(prefix_ids) + len(quantity_ids))

    if generator.random() < 0.5:
        alt_target_kind: Literal["name", "noun"] = "name"
        alt_question_text = f"Q: Who {relation} {quantity} {noun_phrase}?"
        alt_answer_text = f"\nA: {example.query.subject}."
    else:
        alt_target_kind = "noun"
        alt_question_text = (
            f"Q: What does {example.query.subject} {relation_base} {quantity} of?"
        )
        alt_answer_text = f"\nA: {noun_phrase}."

    return DistractorUnit(
        statement_ids=statement_ids,
        statement_value_spans=statement_value_spans,
        target_statement_index=target_index,
        quantity_question_ids=tokenizer.encode(
            quantity_question_text, add_special_tokens=True
        ),
        quantity_answer_ids=quantity_answer_ids,
        quantity_value_span=quantity_value_span,
        alt_target_kind=alt_target_kind,
        alt_question_ids=tokenizer.encode(alt_question_text, add_special_tokens=True),
        alt_answer_ids=tokenizer.encode(alt_answer_text, add_special_tokens=False),
    )


# --------------------------------------------------------------------------
# Chat and filler text streams.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatSource:
    path: Path
    weight: float = 1.0


_CHAT_SOURCE_FILENAMES = ("smoltalk_everyday.jsonl", "lmsys_chat_1m.jsonl")


def _local_chat_data_root() -> Path:
    """Resolves the local chat-export directory, preferring a real local
    disk over the G: drive's Google-Drive-synced virtual filesystem - a
    real pregeneration run (727k sessions, ~28 workers each holding a
    long-lived read handle against the same two G: files) triggered
    Google Drive Desktop to drop its connection mid-run, killing every
    worker with `OSError: [Errno 22] Invalid argument` simultaneously
    (~13 of ~28 in one cluster). C:\\path\\to\\manifoldflow_chat_sources holds a
    one-time local copy of both files specifically to avoid depending on
    that connection staying up under concurrent load again.

    Candidates are checked in order and a directory only counts if it
    actually contains *both* expected files (not just exists) - avoids
    picking a directory mid-copy. G: candidates stay as a fallback for a
    fresh environment that hasn't run the D: copy step yet: a WSL DrvFs
    mount (`/mnt/g/...`, only works if G: happens to already be mounted -
    confirmed unreliable, doesn't auto-mount the way a real disk like D:
    does) and the native Windows drive letter (`G:\\...`, always works
    from a Windows-native Python process, which is how
    `pregenerate_manifoldflow_sessions.py` has been run and tested this
    session). Mirrors `sonic_real_corpus_memory_probe.py`'s identical
    `_DEFAULT_DATA_ROOT` os.name branch for the same underlying reason."""
    candidates = [
        Path("C:/path/to/manifoldflow_chat_sources"),
        Path("/path/to/gdrive/training_data/chat"),
        Path("C:/path/to/gdrive/training_data/chat"),
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in _CHAT_SOURCE_FILENAMES):
            return candidate
    raise FileNotFoundError(
        "local chat data root not found (needing all of "
        f"{_CHAT_SOURCE_FILENAMES}) at any of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def default_chat_sources() -> list[ChatSource]:
    """smoltalk (general everyday chat - the closest match to this
    project's "chat episode" intent; the sibling `smoltalk_reasoning.jsonl`/
    `smoltalk_openthoughts_sft.jsonl` exports are CoT-style, a better fit
    for the design doc's separate "reasoning" episode bucket than for
    plain chat, and `smoltalk_preference.jsonl` is preference-pair data,
    not turns) plus lmsys-chat-1m (real human/model conversations, a
    different distribution than smoltalk's more synthetic/instruction
    style) - equal weight, matching this project's existing convention of
    weighting *sources* rather than raw corpus size
    (`smollm3_recipe.SourceSpec.weight`), since raw size would otherwise
    let lmsys's 2.1GB (vs. smoltalk_everyday's 1.9MB) swamp the mix by
    three orders of magnitude for no principled reason."""
    root = _local_chat_data_root()
    return [
        ChatSource(path=root / "smoltalk_everyday.jsonl", weight=1.0),
        ChatSource(path=root / "lmsys_chat_1m.jsonl", weight=1.0),
    ]


class ChatTurnStream:
    """Reads pre-rendered multi-turn chat text from one or more local
    JSONL exports (schema: `{"text": "USER: ...\\n\\nASSISTANT: ..."}`,
    one conversation per line - `default_chat_sources()`'s smoltalk_everyday
    + lmsys_chat_1m files use this exact schema already, so no re-rendering
    is needed, just light truncation to `max_turns`).

    Each source is read with a bounded shuffle buffer (mirrors
    `HFStreamingTextSource`'s own `dataset.shuffle(seed=..., buffer_size=
    ...)` convention) rather than loaded into memory - some sibling
    exports in the same directory (not used by the default sources, but a
    caller could point here) are 50GB+, so a full read or in-memory
    shuffle isn't an option in general. Sources are sampled per call by
    `weight`, then reopened (looping back to the start) once exhausted -
    same infinite-stream convention as the original HF-streaming version
    of this class."""

    def __init__(
        self,
        sources: list[ChatSource] | None = None,
        *,
        seed: int = 0,
        buffer_size: int = 2000,
    ) -> None:
        self._sources = sources or default_chat_sources()
        if not self._sources:
            raise ValueError("ChatTurnStream needs at least one source")
        self._weights = [source.weight for source in self._sources]
        self._buffer_size = buffer_size
        self._select_generator = random.Random(seed)
        self._iterators = [
            self._buffered_lines(source.path, seed=seed + index)
            for index, source in enumerate(self._sources)
        ]

    def _buffered_lines(self, path: Path, *, seed: int) -> Iterator[str]:
        rng = random.Random(seed)
        while True:
            with path.open("r", encoding="utf-8") as handle:
                buffer: list[str] = []
                for line in handle:
                    buffer.append(line)
                    if len(buffer) >= self._buffer_size:
                        rng.shuffle(buffer)
                        yield from buffer
                        buffer = []
                if buffer:
                    rng.shuffle(buffer)
                    yield from buffer

    def next_turns_text(self, *, max_turns: int = 6) -> str:
        """Returns one conversation's pre-rendered `USER: .../ASSISTANT:
        ...` text, truncated to `max_turns` turns (i.e. `2 * max_turns`
        USER/ASSISTANT chunks)."""
        while True:
            source_index = self._select_generator.choices(
                range(len(self._sources)), weights=self._weights, k=1
            )[0]
            line = next(self._iterators[source_index])
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            chunks = re.split(r"\n\n(?=USER:|ASSISTANT:)", text)
            chunks = [chunk for chunk in chunks if chunk.strip()]
            if len(chunks) < 2:
                continue
            return "\n\n".join(chunks[: 2 * max_turns])


def _build_filler_source(
    tokenizer: Any, *, eos_token_id: int, cache_root: Path, seed: int
) -> Any:
    """One `HFStreamingTextSource` reading fineweb-edu, reused unmodified
    from the vendored SmolLM3 recipe - the same source used for the
    "plain LM" 60% portion and for "other distraction" filler mixed
    alongside a distractor unit's target."""
    from concurrent.futures import ThreadPoolExecutor

    from gator.smollm3_recipe import HFStreamingTextSource, SourceSpec

    spec = SourceSpec(name="fineweb-edu", repo="HuggingFaceFW/fineweb-edu", weight=1.0)
    executor = ThreadPoolExecutor(max_workers=1)
    return HFStreamingTextSource(
        spec,
        tokenizer,
        eos_token_id=eos_token_id,
        cache_root=cache_root,
        executor=executor,
        chunk_tokens=1 << 16,
        prefetch_tokens=1 << 14,
        max_document_chars=1 << 18,
    )


_FILLER_TAKE_TIMEOUT_SECONDS = 30.0


def _take_filler_with_timeout(
    filler_source: Any, count: int, *, timeout: float = _FILLER_TAKE_TIMEOUT_SECONDS
) -> list[int]:
    """Bounds one `filler_source.take()` call. `HFStreamingTextSource`'s
    underlying HF Hub fetch has no timeout of its own - confirmed to hang
    indefinitely (no error, no progress, ever) on a real pregeneration run,
    permanently blocking the worker (and, since `main()` joins every
    worker, the entire job). Runs the call on a daemon thread so a stuck
    call can't block process exit either, and raises `TimeoutError` if it
    doesn't return in time."""
    result: queue.Queue = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result.put(("ok", filler_source.take(count)))
        except Exception as error:  # pragma: no cover - network dependent
            result.put(("error", error))

    threading.Thread(target=_run, daemon=True).start()
    try:
        status, payload = result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(
            f"filler_source.take({count}) did not return within {timeout}s"
        ) from None
    if status == "error":
        raise payload
    return payload


# --------------------------------------------------------------------------
# Session assembly.
# --------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    episode_type: EpisodeType
    token_ids: list[int]
    # distractor-only fields (None otherwise):
    statement_spans: list[tuple[int, int]] | None = None
    # Each statement's own digit sub-span within `statement_spans` (same
    # index correspondence) - lets the training loop pool only the
    # quantity token for the write-side value, not the whole statement
    # ("value_source='quantity'" - sonic-correction-quantity-pooling-fix.md
    # found this a dramatic fix for decode accuracy in every probe that
    # used it; kept optional here since older shards won't have it).
    statement_value_spans: list[tuple[int, int]] | None = None
    target_statement_index: int | None = None
    query_span: tuple[int, int] | None = None
    answer_span: tuple[int, int] | None = None
    answer_value_span: tuple[int, int] | None = None
    target_kind: Literal["quantity", "name", "noun"] | None = None
    # Unique per distractor unit (shared by its write episode and its later
    # query episode) - needed because `target_statement_index` is only
    # local to one unit's own statement list (e.g. 0-6), and the deferred-
    # query design lets many units be pending at once (see `build_session`),
    # so a query episode alone can't otherwise say *which* unit's write it
    # targets once more than one is in flight - the common case at real
    # session lengths. The training loop resolves a query's global bank
    # slot as base_offset[unit_id] + target_statement_index.
    unit_id: int | None = None


@dataclass
class Session:
    session_id: int
    episodes: list[EpisodeRecord] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(len(episode.token_ids) for episode in self.episodes)


# Top-level episode-type mix (Sonic-MF Training Design v0.1.md §25/§43,
# collapsing "reasoning" and "code" into one "filler" bucket for v1 - no
# reasoning/code generator exists yet, and generic prose already exercises
# the same "irrelevant surrounding content" property those categories were
# meant to provide).
_EPISODE_TYPE_WEIGHTS: dict[EpisodeType, float] = {
    "chat": 0.35,
    "distractor": 0.35,
    "filler": 0.30,
}


def _query_distance_is_mature(progress_fraction: float) -> bool:
    """Sonic-MF Training Design v0.1.md §20: "do not use a fixed delay,
    sample logarithmically" - implemented here as a two-stage schedule tied
    to training progress (0.0 at the start of the curriculum, 1.0 at the
    end), independent of the accuracy-gated Sonic-unfreeze trigger, which
    is a training-loop concern, not a dataset-generation one."""
    return progress_fraction >= 0.5


def _sample_deferred_gap(generator: random.Random, *, mature: bool) -> int:
    """How many further episodes to place before querying a pending fact -
    §20's early/mature distance-share tables, approximated as a small
    number of discrete "episodes until queried" buckets rather than an
    exact token-distance sampler (token distance is an emergent
    consequence of how many/how-long the intervening episodes turn out to
    be, which isn't known until they're generated)."""
    if mature:
        buckets = [0, 1, 2, 4, 8, 16]
        weights = [0.05, 0.15, 0.20, 0.30, 0.20, 0.10]
    else:
        buckets = [0, 1, 2, 4, 8]
        weights = [0.20, 0.20, 0.25, 0.20, 0.15]
    return generator.choices(buckets, weights=weights, k=1)[0]


def build_session(
    session_id: int,
    generator: random.Random,
    tokenizer: Any,
    *,
    target_tokens: int,
    level: int,
    progress_fraction: float,
    chat_stream: ChatTurnStream,
    filler_source: Any,
    never_queried_fraction: float = 0.15,
) -> Session:
    """Sonic-MF Training Design v0.1.md §29's `build_session` loop, made
    concrete: sample an episode type, append it, repeat until the session
    reaches roughly `target_tokens`. Distractor units go into a
    session-level pending pool; queries for them get deferred per
    `_sample_deferred_gap` (or dropped forever per `never_queried_fraction`,
    §22) rather than always asked immediately."""
    session = Session(session_id=session_id)
    mature = _query_distance_is_mature(progress_fraction)
    # (unit, episodes_until_query, unit_id) - unit_id ties a query episode
    # back to its own write episode's bank slots (see EpisodeRecord.unit_id).
    pending: list[tuple[DistractorUnit, int, int]] = []
    next_unit_id = 0
    types = list(_EPISODE_TYPE_WEIGHTS.keys())
    weights = list(_EPISODE_TYPE_WEIGHTS.values())

    while session.total_tokens < target_tokens:
        # Countdown and fire any pending queries whose deferred gap has
        # elapsed, before appending this step's new episode - keeps query
        # placement causal (the query never appears before its gap ends).
        still_pending: list[tuple[DistractorUnit, int, int]] = []
        for unit, remaining, unit_id in pending:
            if remaining <= 0:
                session.episodes.append(
                    _render_query_episode(unit, generator, unit_id)
                )
                continue
            still_pending.append((unit, remaining - 1, unit_id))
        pending = still_pending

        episode_type = generator.choices(types, weights=weights, k=1)[0]
        if episode_type == "chat":
            text = chat_stream.next_turns_text()
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            session.episodes.append(
                EpisodeRecord(episode_type="chat", token_ids=token_ids)
            )
        elif episode_type == "filler":
            try:
                token_ids = _take_filler_with_timeout(
                    filler_source, generator.randint(64, 512)
                )
            except TimeoutError:
                # A real 727k-session run hung 8 of 28 workers permanently
                # (no error, no progress, indefinitely) here - HFStreamingTextSource's
                # underlying HF Hub fetch has no timeout of its own, and
                # since main() joins every worker, one stuck worker blocks
                # the entire job forever. Skipping this filler draw bounds
                # the damage to `_FILLER_TAKE_TIMEOUT_SECONDS` instead of
                # hanging forever - not a full fix for a poisoned
                # connection (a worker that draws filler repeatedly after
                # its source wedges pays the timeout tax every time), but
                # turns an unbounded hang into a bounded delay.
                continue
            session.episodes.append(
                EpisodeRecord(episode_type="filler", token_ids=list(token_ids))
            )
        else:
            unit = _build_distractor_unit(generator, tokenizer, level=level)
            if unit is None:
                continue
            unit_id = next_unit_id
            next_unit_id += 1
            session.episodes.append(_render_write_episode(unit, unit_id))
            if generator.random() < never_queried_fraction:
                continue  # §22: some facts are never queried at all.
            gap = _sample_deferred_gap(generator, mature=mature)
            pending.append((unit, gap, unit_id))

    # Flush any still-pending queries so most written facts get answered
    # somewhere in the session, rather than silently dropping them all.
    for unit, _, unit_id in pending:
        session.episodes.append(_render_query_episode(unit, generator, unit_id))
    return session


def _render_write_episode(unit: DistractorUnit, unit_id: int) -> EpisodeRecord:
    """Renders a distractor unit's statements only - no query yet. Each
    statement is tracked by its own token span so the training loop can
    write it to `BasinMemory` individually, exactly like every probe's
    per-statement write loop already does."""
    token_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    value_spans: list[tuple[int, int]] = []
    for statement, (value_start, value_end) in zip(
        unit.statement_ids, unit.statement_value_spans, strict=True
    ):
        start = len(token_ids)
        token_ids.extend(statement)
        spans.append((start, len(token_ids)))
        value_spans.append((start + value_start, start + value_end))
    return EpisodeRecord(
        episode_type="distractor",
        token_ids=token_ids,
        statement_spans=spans,
        statement_value_spans=value_spans,
        target_statement_index=unit.target_statement_index,
        unit_id=unit_id,
    )


def _render_query_episode(
    unit: DistractorUnit, generator: random.Random, unit_id: int
) -> EpisodeRecord:
    """Renders a distractor unit's query+answer only (statements were
    already written by `_render_write_episode` earlier in the session) -
    picks quantity or the unit's alternate target (name/noun) at random,
    same target-type variety `build_session` already applies at write
    time."""
    if generator.random() < 0.5:
        question_ids, answer_ids = unit.quantity_question_ids, unit.quantity_answer_ids
        value_start, value_end = unit.quantity_value_span
        answer_value_span: tuple[int, int] | None = (
            len(question_ids) + value_start,
            len(question_ids) + value_end,
        )
        target_kind: Literal["quantity", "name", "noun"] = "quantity"
    else:
        question_ids, answer_ids = unit.alt_question_ids, unit.alt_answer_ids
        answer_value_span = None  # full-sequence match, no digit sub-span
        target_kind = unit.alt_target_kind
    token_ids = list(question_ids) + list(answer_ids)
    query_span = (0, len(question_ids))
    answer_span = (len(question_ids), len(token_ids))
    if answer_value_span is not None:
        answer_value_span = answer_value_span  # already offset from question start
    return EpisodeRecord(
        episode_type="distractor",
        token_ids=token_ids,
        target_statement_index=unit.target_statement_index,
        query_span=query_span,
        answer_span=answer_span,
        answer_value_span=answer_value_span,
        target_kind=target_kind,
        unit_id=unit_id,
    )


# --------------------------------------------------------------------------
# Physical packing and serialization.
# --------------------------------------------------------------------------


def pack_session(session: Session) -> dict[str, Any]:
    """Flattens a session's episodes into one token stream plus an index
    mapping (episode boundaries, statement/query spans in *packed*
    coordinates) - windowing into physical 8192-token slices
    (`PHYSICAL_WINDOW_TOKENS`) is a pure, lossless re-slice of this at
    training time, so it isn't baked into the stored shard (letting the
    training loop choose a different physical window size later without
    regenerating data)."""
    token_ids: list[int] = []
    episode_index: list[dict[str, Any]] = []
    for episode in session.episodes:
        start = len(token_ids)
        token_ids.extend(episode.token_ids)
        record: dict[str, Any] = {
            "episode_type": episode.episode_type,
            "start": start,
            "end": len(token_ids),
        }
        if episode.episode_type == "distractor":
            record["target_statement_index"] = episode.target_statement_index
            record["unit_id"] = episode.unit_id
            if episode.statement_spans is not None:
                record["statement_spans"] = [
                    (start + a, start + b) for a, b in episode.statement_spans
                ]
            if episode.statement_value_spans is not None:
                record["statement_value_spans"] = [
                    (start + a, start + b) for a, b in episode.statement_value_spans
                ]
            if episode.query_span is not None:
                a, b = episode.query_span
                record["query_span"] = (start + a, start + b)
            if episode.answer_span is not None:
                a, b = episode.answer_span
                record["answer_span"] = (start + a, start + b)
            if episode.answer_value_span is not None:
                a, b = episode.answer_value_span
                record["answer_value_span"] = (start + a, start + b)
            if episode.target_kind is not None:
                record["target_kind"] = episode.target_kind
        episode_index.append(record)
    return {
        "session_id": session.session_id,
        "token_ids": token_ids,
        "episodes": episode_index,
        "total_tokens": len(token_ids),
    }


def write_shard(sessions: list[dict[str, Any]], path: Path | str) -> None:
    """One shard = one JSON-lines file of packed sessions - simple,
    inspectable, and streamable without a full-shard load, matching the
    project's existing pretokenized-shard convention for the real-corpus
    probe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session))
            handle.write("\n")


def read_shard(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
