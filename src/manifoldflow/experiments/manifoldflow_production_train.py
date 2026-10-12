"""Production ManifoldFlow training loop - Phase 2 of the approved plan
(`C:\\Users\\user\\.claude\\plans\\rustling-nibbling-flute.md`). Consumes
pregenerated session shards written by `pregenerate_manifoldflow_sessions.py`
(`manifoldflow.dataset.session_builder`), replays them through persistent
batch lanes - generalizing `sonic_batch_lane_persistence_probe.py`'s
per-lane detach-between-steps design from single correction-episodes to
real packed 8192-token session windows - and trains the ManifoldFlow
adapter (port + key/value projections) jointly with a cross-optimizer
staged unfreeze of Sonic itself, gated on retrieval accuracy reaching
`--unfreeze-accuracy-threshold` (default 0.95) for `--unfreeze-patience`
consecutive eval checkpoints - the same mechanism
`sonic_noun_target_probe.py` already proved at probe scale.

ManifoldFlow is wired in on every window, not just "memory task" windows:
one batched `model.forward_hidden` call per step produces hidden states
for the whole lane batch's current window, and that SAME hidden tensor
backs three things every step: (1) the plain teacher-forced LM loss over
every position (`model.chunked_cross_entropy`), (2) statement writes into
each lane's persistent `BasinMemory` for any distractor statement fully
contained in this window, and (3) retrieval + port-injected decode for
any due query fully contained in this window. Nothing about the forward
path changes between window types - plain-LM windows simply have nothing
to do at (2) or (3), avoiding the train/inference architecture mismatch
the plan called out. Unlike the batch-lane probe (which ran a *separate*
forward_hidden call over each isolated statement/query snippet), writes
and queries here pool directly out of the one whole-window hidden tensor,
so each statement is now contextualized against everything earlier in the
packed session too - the richer signal real persistent-state training is
meant to exercise, not a bug relative to the probes' isolated version.

60/40 interpretation: the user's spec ("60% of the training should still
be regular teacher forced autoregressive language training... [with]
manifoldflow... on the entire time") is implemented here as a fixed loss-
weighting ratio (`--plain-lm-loss-weight` / `--distractor-loss-weight`,
default 0.6 / 0.4) between the two objectives, not a literal 60/40 split
of which *steps* or *lanes* carry which objective. A literal split would
require deciding, before generation, which windows will or won't contain
a due query - but a lane's windows must replay its own pregenerated
session in strict chronological order (persistent state depends on it),
so which windows end up carrying a due query is downstream of session
generation, not something this loop can reassign without breaking
causality. Both loss terms are simply summed with these fixed weights
every step (the distractor term is 0 on steps where no lane in the batch
has a due query this window). The empirical due-query-window frequency is
logged at every eval checkpoint (`eval_due_query_window_fraction`) so a
mismatch against the intended ~40% can be diagnosed by revisiting
`session_builder._EPISODE_TYPE_WEIGHTS`, not by re-tuning these weights.

Checkpoint/resume mirrors `sonic_ssm_train.py`'s conventions (atomic
`.tmp` + `path.replace()` writes; a strict resume-guard that fails loudly
on any format/model-config/tokenizer mismatch) with one scope-limited
difference: lane state (each lane's in-progress session, bank contents,
and stored statement keys/values) is not serialized - a resumed run
starts every lane on a fresh session rather than reconstructing exactly
where it left off mid-session. Documented here as a known simplification
for this first working version, matching the plan's "get something
working and smoke-tested first, not the full curriculum in one shot"
scope; the trained weights and optimizer state themselves resume exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from manifoldflow.compute import (
    CompileConfig,
    Float32RuntimeConfig,
    configure_float32_runtime,
    mark_compiled_iteration_begin,
)
from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.dataset.session_builder import PHYSICAL_WINDOW_TOKENS, read_shard
from manifoldflow.muon import Muon
from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
    info_nce_retrieval_loss,
)
from manifoldflow.sonic_memory_port import SonicMemoryPort, SonicMemoryPortConfig
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

CHECKPOINT_FORMAT = "manifoldflow-production-v1"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOKENIZER_DIR = _PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k"
sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))


# ----------------------------------------------------------------------------
# Session streaming and windowing.
# ----------------------------------------------------------------------------


class SessionStream:
    """Cycles through pregenerated session shards under `shard_dir`,
    reshuffling shard order and reopening once exhausted (mirrors
    `session_builder.ChatTurnStream`'s infinite-loop convention). Each lane
    owns its own stream so lanes don't contend for the same session."""

    def __init__(self, shard_dir: Path, *, seed: int) -> None:
        # Recursive: separate pregeneration runs (e.g. a supplemental batch
        # topping up an earlier run that fell short) are written into their
        # own subdirectory to avoid worker/shard-index filename collisions
        # with the first run - see pregenerate_manifoldflow_sessions.py's
        # per-worker shard naming.
        self._paths = sorted(Path(shard_dir).rglob("*.jsonl"))
        if not self._paths:
            raise FileNotFoundError(f"no *.jsonl shards found under {shard_dir}")
        self._generator = random.Random(seed)
        self._iterator: Iterator[dict[str, Any]] | None = None
        # PrefetchingSessionPool's initial buffer fill submits multiple
        # next_session() calls for the same lane's stream before the first
        # one has returned - this stream's own generator/iterator state
        # isn't otherwise safe for concurrent access.
        self._lock = threading.Lock()

    def _open(self) -> Iterator[dict[str, Any]]:
        paths = list(self._paths)
        self._generator.shuffle(paths)
        for path in paths:
            yield from read_shard(path)

    def next_session(self) -> dict[str, Any]:
        with self._lock:
            if self._iterator is None:
                self._iterator = self._open()
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = self._open()
                return next(self._iterator)


class PrefetchingSessionPool:
    """Background-thread session prefetcher shared across all lanes.

    Every pregenerated session is exactly `target_tokens` (32768 in
    production) = 4 physical windows, and every lane starts fresh
    (`session=None`) at step 1 - so nearly all lanes exhaust their session
    and call `reset_session()` in the same handful of steps, roughly every
    4 steps thereafter too. Without prefetching, that turns "occasional
    disk I/O + JSON parsing" into a synchronized burst across every lane
    at once, blocking the main training loop (and idling the GPU) right
    before it - confirmed as a real, periodic stall via a live GPU
    utilization histogram during an actual run, not just theory.

    A small shared pool (`num_workers`, independent of lane count) keeps
    `buffer_size` sessions per lane pre-fetched, each background task
    running both `SessionStream.next_session` (read+parse) *and*
    `_precompute_windows` (the window/episode-grouping pass - see its own
    docstring for why that needed to move off the main thread too, not
    just the read), so `next_session(lane_index)` is normally just a
    `Future.result()` on already-finished work instead of synchronous CPU
    work on the main thread."""

    def __init__(
        self,
        streams: list[SessionStream],
        *,
        window_tokens: int,
        pad_token_id: int,
        num_workers: int = 8,
        buffer_size: int = 2,
    ) -> None:
        self._streams = streams
        self._window_tokens = window_tokens
        self._pad_token_id = pad_token_id
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="session-prefetch"
        )
        self._pending: list[list[Future]] = [[] for _ in streams]
        for lane_index in range(len(streams)):
            for _ in range(buffer_size):
                self._submit(lane_index)

    def _prepare(self, lane_index: int) -> "PreparedSession":
        session = self._streams[lane_index].next_session()
        windows = _precompute_windows(
            session, window_tokens=self._window_tokens, pad_token_id=self._pad_token_id
        )
        return PreparedSession(windows=windows)

    def _submit(self, lane_index: int) -> None:
        self._pending[lane_index].append(self._executor.submit(self._prepare, lane_index))

    def next_session(self, lane_index: int) -> "PreparedSession":
        future = self._pending[lane_index].pop(0)
        session = future.result()
        self._submit(lane_index)  # keep the buffer topped back up
        return session

    def handle(self, lane_index: int) -> "PrefetchHandle":
        return PrefetchHandle(pool=self, lane_index=lane_index)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class PrefetchHandle:
    """Per-lane view into a shared `PrefetchingSessionPool` - exposes a
    `next_session() -> PreparedSession` interface so `Lane` doesn't need
    to know it's talking to a prefetch pool rather than a plain stream."""

    pool: PrefetchingSessionPool
    lane_index: int

    def next_session(self) -> "PreparedSession":
        return self.pool.next_session(self.lane_index)


@dataclass(frozen=True)
class WindowWrite:
    unit_id: int
    statement_spans: list[tuple[int, int]]  # window-local
    statement_value_spans: list[tuple[int, int]] | None  # window-local


@dataclass(frozen=True)
class WindowQuery:
    unit_id: int
    target_statement_index: int  # local to the unit
    query_span: tuple[int, int]  # window-local
    answer_span: tuple[int, int]  # window-local
    answer_value_span: tuple[int, int] | None  # window-local
    target_kind: str


@dataclass(frozen=True)
class Window:
    token_ids: list[int]  # length == window_tokens, right-padded
    valid_length: int  # number of real (non-pad) tokens
    writes: list[WindowWrite]
    queries: list[WindowQuery]


def _fully_contained(span: tuple[int, int], window_start: int, window_end: int) -> bool:
    a, b = span
    return window_start <= a and b <= window_end


def build_window(
    session: dict[str, Any],
    window_start: int,
    window_end: int,
    *,
    window_tokens: int,
    pad_token_id: int,
) -> Window:
    """Slices `session["token_ids"][window_start:window_end]` into a fixed-
    width, right-padded window, and classifies every distractor episode
    into a write or a query only if *all* of its relevant spans are fully
    contained in this window - a statement or query straddling a physical
    window boundary is simply skipped this window (and, for a write, its
    fact is lost for retrieval purposes - an accepted first-version
    limitation, since re-querying across boundaries would need per-window
    partial-span carry-over machinery this version doesn't have).

    A span's end must additionally be `<= window_tokens - 1`: the causal
    hidden tensor produced from `input_ids = window[:-1]` only has
    `window_tokens - 1` positions (`hidden[t]` is "the state after seeing
    token t"), so a span whose last token is the window's very last token
    has no hidden state to pool - vanishingly rare given windows are 8192
    tokens and spans are a few dozen at most, but guarded here regardless.
    """
    token_ids = session["token_ids"]
    raw = token_ids[window_start:window_end]
    padded = raw + [pad_token_id] * (window_tokens - len(raw))
    usable_end = window_start + min(window_end - window_start, window_tokens - 1)

    writes: list[WindowWrite] = []
    queries: list[WindowQuery] = []
    for episode in session["episodes"]:
        if episode["episode_type"] != "distractor":
            continue
        if "statement_spans" in episode:
            spans = episode["statement_spans"]
            if not all(_fully_contained(s, window_start, usable_end) for s in spans):
                continue
            value_spans = episode.get("statement_value_spans")
            writes.append(
                WindowWrite(
                    unit_id=episode["unit_id"],
                    statement_spans=[(a - window_start, b - window_start) for a, b in spans],
                    statement_value_spans=(
                        [(a - window_start, b - window_start) for a, b in value_spans]
                        if value_spans is not None
                        else None
                    ),
                )
            )
        elif "query_span" in episode:
            if not (
                _fully_contained(episode["query_span"], window_start, usable_end)
                and _fully_contained(episode["answer_span"], window_start, usable_end)
            ):
                continue
            value_span = episode.get("answer_value_span")
            qa, qb = episode["query_span"]
            aa, ab = episode["answer_span"]
            queries.append(
                WindowQuery(
                    unit_id=episode["unit_id"],
                    target_statement_index=episode["target_statement_index"],
                    query_span=(qa - window_start, qb - window_start),
                    answer_span=(aa - window_start, ab - window_start),
                    answer_value_span=(
                        (value_span[0] - window_start, value_span[1] - window_start)
                        if value_span is not None
                        else None
                    ),
                    target_kind=episode["target_kind"],
                )
            )
    return Window(token_ids=padded, valid_length=len(raw), writes=writes, queries=queries)


@dataclass(frozen=True)
class PreparedSession:
    """A session's physical windows, computed once. Replaces the raw
    session dict once ready - a lane never needs the raw token_ids/episode
    list again after this, only the precomputed windows, so dropping the
    dict here is a memory win too, not just a CPU one."""

    windows: list[Window]


def _precompute_windows(
    session: dict[str, Any], *, window_tokens: int, pad_token_id: int
) -> list[Window]:
    """Computes every physical window for a session once, instead of the
    equivalent `build_window()` call re-scanning the session's *entire*
    episode list from scratch on every `Lane.advance_window()` call. A
    real production session can carry 100+ episodes, and unlike the
    session read+parse itself (fixed by `PrefetchingSessionPool`), this
    cost repeats on *every* window (not just at session reset) - so it
    kept showing up as a real, CPU-only, unoverlapped gap between GPU
    bursts even after prefetching and batching (confirmed via a live GPU
    utilization histogram: wider idle troughs, not just smaller ones).
    Called from the prefetch pool's background workers, so - like the
    read+parse before it - this happens off the main training-loop
    thread entirely."""
    token_ids = session["token_ids"]
    num_windows = max(1, -(-len(token_ids) // window_tokens))  # ceil div
    windows: list[Window] = []
    for window_index in range(num_windows):
        start = window_index * window_tokens
        end = min(start + window_tokens, len(token_ids))
        windows.append(
            build_window(session, start, end, window_tokens=window_tokens, pad_token_id=pad_token_id)
        )
    return windows


# ----------------------------------------------------------------------------
# Persistent batch lanes.
# ----------------------------------------------------------------------------


@dataclass
class Lane:
    stream: PrefetchHandle
    d_model: int
    session: PreparedSession | None = None
    window_index: int = 0
    bank: BasinMemory = field(init=False)
    statement_keys: list[torch.Tensor] = field(default_factory=list)
    statement_values: list[torch.Tensor] = field(default_factory=list)
    unit_base_offset: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bank = BasinMemory(dimension=self.d_model, novelty_threshold=0.0)

    def reset_session(self) -> None:
        self.session = self.stream.next_session()
        self.window_index = 0
        self.bank = BasinMemory(dimension=self.d_model, novelty_threshold=0.0)
        self.statement_keys = []
        self.statement_values = []
        self.unit_base_offset = {}

    def advance_window(self) -> Window:
        # windows are precomputed (see PreparedSession/_precompute_windows) -
        # this is now a plain index into an already-built list, not a fresh
        # per-step scan over the session's episodes.
        if self.session is None or self.window_index >= len(self.session.windows):
            self.reset_session()
        window = self.session.windows[self.window_index]
        self.window_index += 1
        return window

    def detach_(self) -> None:
        self.statement_keys = [key.detach() for key in self.statement_keys]
        self.statement_values = [value.detach() for value in self.statement_values]


def new_lane(stream: PrefetchHandle, d_model: int) -> Lane:
    return Lane(stream=stream, d_model=d_model)


# ----------------------------------------------------------------------------
# One step: forward, write, retrieve, decode.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryOutcome:
    retrieval_loss: torch.Tensor
    decode_loss: torch.Tensor
    retrieval_correct: bool
    # None when track_accuracy=False (training steps - these aren't read by
    # the training loop, only by run_eval, so computing them costs a real
    # CPU-GPU sync via .item() for nothing every step - see run_step's
    # track_accuracy docstring note).
    answer_correct: bool | None
    value_correct: bool | None  # also None when target_kind isn't "quantity"


@dataclass(frozen=True)
class StepOutcome:
    plain_lm_loss: torch.Tensor
    distractor_loss: torch.Tensor
    query_outcomes: list[QueryOutcome]
    windows_with_due_query: int
    total_windows: int


def _pool(hidden: torch.Tensor, span: tuple[int, int], window_tokens: int) -> torch.Tensor:
    a, b = span
    b = min(b, window_tokens - 1)
    return hidden[a:b].mean(dim=0, keepdim=True)


def _gather_padded_spans(
    hidden_float: torch.Tensor,
    clamped_spans: list[tuple[int, int, int]],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gathers many (lane_index, a, b) token spans out of `hidden_float`
    into one `[len(clamped_spans), max_len, d_model]` padded batch, plus a
    same-shaped validity mask, via a single `index_select` against a flat
    view of `hidden_float` - instead of the equivalent Python-loop
    version's one `padded[row, :length] = hidden_float[lane_index, a:b]`
    slice-assign per item. Confirmed as a real GPU-idling bottleneck on a
    live run: a live GPU utilization histogram stayed choppy even after
    fixing the accuracy-tracking sync and adding session prefetching, with
    16+ lanes each carrying several statements/queries per window - 50-100+
    tiny serialized ops per training step, none individually slow but
    collectively enough that the CPU couldn't issue work as fast as the
    GPU finished each one. This matters even more once Sonic is unfrozen:
    `hidden_float` then carries gradients, and the per-item slice-assign
    loop registers one `SelectBackward0` per item - confirmed via
    torch.profiler as the single largest CUDA time sink of an unfrozen
    step (~7s of ~15s, across ~950 calls in one optimizer update) once
    every write/quantity/query/answer-span job across a real session
    starts paying real autograd bookkeeping instead of a plain no-op-
    under-no_grad slice. A single index_select's backward is one fused
    scatter-add, not one op per item.

    Index/mask construction is plain Python-loop integer bookkeeping with
    no autograd cost - those tensors never require grad, so the loop below
    is cheap regardless of whether `hidden_float` requires grad."""
    num_lanes, seq_len, d_model = hidden_float.shape
    flat_indices = torch.zeros((len(clamped_spans), max_len), dtype=torch.long)
    mask = torch.zeros((len(clamped_spans), max_len))
    for row, (lane_index, a, b) in enumerate(clamped_spans):
        length = max(b - a, 0)
        if length > 0:
            flat_indices[row, :length] = torch.arange(a, a + length) + lane_index * seq_len
            mask[row, :length] = 1.0
    flat_indices = flat_indices.to(hidden_float.device)
    mask = mask.to(hidden_float.device)

    hidden_flat = hidden_float.reshape(num_lanes * seq_len, d_model)
    gathered = hidden_flat.index_select(0, flat_indices.reshape(-1)).view(
        len(clamped_spans), max_len, d_model
    )
    return gathered, mask


def _batched_pool(
    hidden_float: torch.Tensor,
    jobs: list[tuple[int, tuple[int, int]]],
    window_tokens: int,
) -> torch.Tensor:
    """Mean-pools many (lane_index, span) pairs into one `[len(jobs),
    d_model]` batch in a single masked reduction - see
    `_gather_padded_spans` for why this is batched instead of the
    equivalent Python-loop version's one `_pool` call (and, at the call
    sites below, one separate `key_projection`/`value_projection`
    nn.Module forward) per item."""
    d_model = hidden_float.shape[-1]
    if not jobs:
        return hidden_float.new_zeros((0, d_model))
    clamped = [(lane_index, a, min(b, window_tokens - 1)) for lane_index, (a, b) in jobs]
    max_len = max(max(b - a, 0) for _, a, b in clamped)
    max_len = max(max_len, 1)
    gathered, mask = _gather_padded_spans(hidden_float, clamped, max_len)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (gathered * mask.unsqueeze(-1)).sum(dim=1) / counts


def run_step(
    *,
    lanes: list[Lane],
    model: SonicSSMLanguageModel,
    device: str,
    port: SonicMemoryPort,
    key_projection: SonicKeyProjection,
    value_projection: SonicKeyProjection,
    window_tokens: int,
    pad_token_id: int,
    temperature: float,
    recency_weight: float,
    recency_decay: float,
    value_source: str,
    mixed_precision: str,
    sonic_requires_grad: bool,
    ce_chunk_rows: int,
    compile_config: CompileConfig,
    track_accuracy: bool,
) -> StepOutcome:
    # track_accuracy=False (used by the main training loop) skips
    # answer_correct/value_correct's `.all().item()` calls - each forces a
    # hard CPU-GPU sync, and with due_query_window_frac often close to 1.0
    # that happens on every due query on every training step, for metrics
    # the training loop never reads (only run_eval consumes
    # query_outcomes). Confirmed as a real stall source on a live GPU run
    # (bursty, not sustained, GPU utilization) - retrieval_correct stays
    # cheap regardless (a plain Python int comparison, no tensor sync), so
    # it's always computed.
    #
    # Must run once per logical iteration, before the first compiled forward
    # call - see sonic_batch_lane_persistence_probe.py's identical comment;
    # a documented no-op when --compile is "none" (this script's default).
    mark_compiled_iteration_begin(compile_config, device)
    windows = [lane.advance_window() for lane in lanes]

    batch = torch.tensor([w.token_ids for w in windows], device=device, dtype=torch.long)
    input_ids = batch[:, :-1]
    labels = batch[:, 1:].clone()
    for row, window in enumerate(windows):
        if window.valid_length < window_tokens:
            labels[row, max(window.valid_length - 1, 0) :] = -100

    autocast_enabled = mixed_precision == "bf16" and device in ("cuda", "cpu")
    grad_context = contextlib.nullcontext() if sonic_requires_grad else torch.no_grad()
    with grad_context, torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled
    ):
        hidden, _ = model.forward_hidden(input_ids)
        plain_lm_loss = model.chunked_cross_entropy(hidden, labels, chunk_rows=ce_chunk_rows)
    hidden_float = hidden.float()

    query_outcomes: list[QueryOutcome] = []
    distractor_losses: list[torch.Tensor] = []
    windows_with_due_query = 0

    # ---- Phase 1: statement writes, batched across every lane's window
    # this step (not per-statement in a Python loop - see _batched_pool's
    # docstring). A pure-Python bookkeeping pass runs first so
    # unit_base_offset is correct even when one lane has multiple write
    # episodes packed into the same window this step - base_index for a
    # later write in that lane depends on the *count* from an earlier one
    # this same step, which doesn't need the actual tensors yet.
    write_jobs: list[tuple[int, tuple[int, int]]] = []  # (lane_index, span)
    write_value_spans: list[tuple[int, int] | None] = []  # parallel to write_jobs
    lane_running_count = [len(lane.statement_keys) for lane in lanes]
    for lane_index, window in enumerate(windows):
        lane = lanes[lane_index]
        for write in window.writes:
            base_index = lane_running_count[lane_index]
            lane.unit_base_offset[write.unit_id] = base_index
            value_spans = write.statement_value_spans
            for index, span in enumerate(write.statement_spans):
                if span[1] - span[0] <= 0 or span[1] > window_tokens - 1:
                    continue
                write_jobs.append((lane_index, span))
                write_value_spans.append(
                    value_spans[index] if value_source == "quantity" and value_spans is not None else None
                )
                lane_running_count[lane_index] += 1

    if write_jobs:
        pooled = _batched_pool(hidden_float, write_jobs, window_tokens)
        keys = key_projection(pooled)

        quantity_rows = [row for row, span in enumerate(write_value_spans) if span is not None]
        if quantity_rows:
            quantity_jobs = [
                (write_jobs[row][0], write_value_spans[row]) for row in quantity_rows
            ]
            quantity_pooled_subset = _batched_pool(hidden_float, quantity_jobs, window_tokens)
            quantity_pooled_full = pooled.new_zeros(pooled.shape)
            quantity_pooled_full[torch.tensor(quantity_rows, device=device)] = quantity_pooled_subset
            quantity_mask = torch.zeros(len(write_jobs), dtype=torch.bool, device=device)
            quantity_mask[torch.tensor(quantity_rows, device=device)] = True
            value_pooled = torch.where(quantity_mask.unsqueeze(-1), quantity_pooled_full, pooled)
        else:
            value_pooled = pooled
        values = value_projection(value_pooled)

        for row, (lane_index, _span) in enumerate(write_jobs):
            lane = lanes[lane_index]
            key = keys[row]
            # write_always_new (not write()): every Lane's bank is
            # constructed with novelty_threshold=0.0, which makes write()'s
            # own nearest-existing-basin novelty check provably a no-op
            # (see write_always_new's docstring) - skipping it removes two
            # of the three CPU-GPU syncs write() pays per call, confirmed
            # as a real stall source (a live GPU histogram stayed choppy
            # even after batching every other per-item op in this function).
            lane.bank.write_always_new(key.detach(), "statement", Source("user"))
            lane.statement_keys.append(key)
            lane.statement_values.append(values[row])

    # ---- Phase 2: due queries. Pooling + key projection are batched the
    # same way; retrieval (BasinMemory read + InfoNCE) stays per-query
    # since each lane's key stack differs in size/content - it's a cheap
    # op relative to the projection/decode layers, so batching it isn't
    # worth the added complexity of padding+masking a per-lane-sized
    # softmax. Decode (port injection + LM head + cross-entropy) is
    # batched across every due query this step.
    query_jobs: list[tuple[int, WindowQuery, int]] = []  # (lane_index, query, target_index)
    for lane_index, window in enumerate(windows):
        if window.queries:
            windows_with_due_query += 1
        lane = lanes[lane_index]
        for query in window.queries:
            base_index = lane.unit_base_offset.get(query.unit_id)
            num_statements = len(lane.statement_keys)
            if base_index is None or num_statements == 0:
                # The target's write episode landed in an earlier, already-
                # exhausted window slice (or a padded/truncated boundary) -
                # skip rather than score against an undefined target.
                continue
            target_index = base_index + query.target_statement_index
            if target_index >= num_statements:
                continue
            query_jobs.append((lane_index, query, target_index))

    if query_jobs:
        query_pool_jobs = [(lane_index, query.query_span) for lane_index, query, _ in query_jobs]
        query_pooled = _batched_pool(hidden_float, query_pool_jobs, window_tokens)
        query_keys = key_projection(query_pooled)

        retrieval_losses: list[torch.Tensor] = []
        raw_reads: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        top_ks: list[int] = []
        for row, (lane_index, _query, target_index) in enumerate(query_jobs):
            lane = lanes[lane_index]
            num_statements = len(lane.statement_keys)
            query_key = query_keys[row]
            stacked_keys = torch.stack(lane.statement_keys)
            retrieval_losses.append(
                info_nce_retrieval_loss(query_key, stacked_keys, target_index, temperature=temperature)
            )
            top_k = num_statements if recency_weight > 0 else 1
            top_ks.append(top_k)
            # read_raw (not read()): defers the CPU-GPU sync read() pays
            # per call - collected across every due query this step and
            # synced exactly once below, instead of once per query (see
            # BasinMemory.read_raw's docstring; confirmed as a real stall
            # source alongside write()'s own, fixed the same way above).
            raw_reads.append(lane.bank.read_raw(query_key.detach(), top_k=top_k))

        max_k = max(top_ks) if top_ks else 1
        ids_rows = []
        scores_rows = []
        for raw in raw_reads:
            if raw is None:
                ids_rows.append(torch.full((max_k,), -1, dtype=torch.long, device=device))
                scores_rows.append(torch.full((max_k,), float("-inf"), device=device))
                continue
            ids, scores = raw
            pad = max_k - ids.shape[0]
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), -1, dtype=torch.long, device=ids.device)])
                scores = torch.cat([scores, torch.full((pad,), float("-inf"), device=scores.device)])
            ids_rows.append(ids)
            scores_rows.append(scores)
        # The one CPU-GPU sync for every due query's retrieval this step,
        # instead of one per query.
        ids_list = torch.stack(ids_rows).tolist()
        scores_list = torch.stack(scores_rows).tolist()

        retrieval_corrects: list[bool] = []
        retrieved_values: list[torch.Tensor] = []
        for row, (lane_index, _query, target_index) in enumerate(query_jobs):
            lane = lanes[lane_index]
            num_statements = len(lane.statement_keys)
            top_k = top_ks[row]
            best_hit_id: int | None = None
            if raw_reads[row] is not None:
                row_ids = ids_list[row][:top_k]
                row_scores = scores_list[row][:top_k]
                if recency_weight > 0:
                    best_index = max(
                        range(len(row_ids)),
                        key=lambda i: row_scores[i]
                        + recency_weight * recency_decay ** (num_statements - 1 - row_ids[i]),
                    )
                else:
                    best_index = 0
                best_hit_id = row_ids[best_index]
            retrieved_values.append(
                lane.statement_values[best_hit_id]
                if best_hit_id is not None
                else torch.zeros(model.config.d_model, device=device)
            )
            retrieval_corrects.append(bool(best_hit_id is not None and best_hit_id == target_index))

        retrieved_batch = torch.stack(retrieved_values)

        answer_spans_clamped = [
            (lane_index, query.answer_span[0], min(query.answer_span[1], window_tokens - 1))
            for lane_index, query, _ in query_jobs
        ]
        max_answer_len = max(max(b - a, 0) for _, a, b in answer_spans_clamped)
        max_answer_len = max(max_answer_len, 1)
        answer_hidden_padded, answer_mask = _gather_padded_spans(
            hidden_float, answer_spans_clamped, max_answer_len
        )
        answer_hidden_padded = answer_hidden_padded * answer_mask.unsqueeze(-1)
        answer_valid_len = [max(b - a, 0) for _, a, b in answer_spans_clamped]

        answer_hidden_injected = port(answer_hidden_padded, retrieved_batch)
        logits_batch = model.logits_from_hidden(answer_hidden_injected)

        # Targets: the raw window tokens at [answer_start+1, answer_end+1) -
        # `answer_hidden[i]` is the state after processing window token
        # answer_start+i, which predicts window token answer_start+i+1 (the
        # same shift `_autoregressive_batch` applies to the plain-LM loss),
        # so the shift is already baked into this slice - no further
        # [1:]/[:-1] adjustment needed on top of it. Built as one padded
        # [M, max_answer_len] buffer (ignore_index=-100 for padding) via a
        # single Python-list construction + one CPU->GPU transfer, instead
        # of the equivalent per-query torch.tensor(...) + F.cross_entropy(...)
        # calls - both were confirmed as more of the same "many tiny
        # serialized ops" pattern the write/pooling/decode-forward batching
        # already fixed elsewhere in this function.
        target_rows: list[list[int]] = []
        target_valid_len: list[int] = []
        for row, (lane_index, query, _target_index) in enumerate(query_jobs):
            window = windows[lane_index]
            answer_start, answer_end = query.answer_span
            answer_end = min(answer_end, window_tokens - 1)
            raw_targets = window.token_ids[answer_start + 1 : answer_end + 1]
            length = min(len(raw_targets), answer_valid_len[row])
            target_rows.append(raw_targets[:length] + [-100] * (max_answer_len - length))
            target_valid_len.append(length)
        target_ids_padded = torch.tensor(target_rows, device=device, dtype=torch.long)

        per_token_loss = F.cross_entropy(
            logits_batch.transpose(1, 2), target_ids_padded, ignore_index=-100, reduction="none"
        )
        valid_mask = target_ids_padded != -100
        valid_counts = valid_mask.sum(dim=1).clamp(min=1).float()
        decode_losses = per_token_loss.sum(dim=1) / valid_counts

        for row, (lane_index, query, _target_index) in enumerate(query_jobs):
            length = target_valid_len[row]
            if length == 0:
                continue
            decode_loss = decode_losses[row]

            answer_correct: bool | None = None
            value_correct: bool | None = None
            if track_accuracy:
                predicted_logits = logits_batch[row, :length]
                predicted_targets = target_ids_padded[row, :length]
                answer_correct = bool(
                    (predicted_logits.argmax(dim=-1) == predicted_targets).all().item()
                )
                if query.answer_value_span is not None:
                    answer_start = query.answer_span[0]
                    value_start = max(query.answer_value_span[0] - answer_start - 1, 0)
                    value_end = query.answer_value_span[1] - answer_start - 1
                    value_end = min(value_end, length)
                    if value_end > value_start:
                        value_correct = bool(
                            (
                                predicted_logits[value_start:value_end].argmax(dim=-1)
                                == predicted_targets[value_start:value_end]
                            )
                            .all()
                            .item()
                        )

            distractor_losses.append(retrieval_losses[row] + decode_loss)
            query_outcomes.append(
                QueryOutcome(
                    retrieval_loss=retrieval_losses[row],
                    decode_loss=decode_loss,
                    retrieval_correct=retrieval_corrects[row],
                    answer_correct=answer_correct,
                    value_correct=value_correct,
                )
            )

    for lane in lanes:
        lane.detach_()

    distractor_loss = (
        torch.stack(distractor_losses).mean()
        if distractor_losses
        else torch.zeros((), device=device)
    )
    return StepOutcome(
        plain_lm_loss=plain_lm_loss,
        distractor_loss=distractor_loss,
        query_outcomes=query_outcomes,
        windows_with_due_query=windows_with_due_query,
        total_windows=len(lanes),
    )


# ----------------------------------------------------------------------------
# Checkpointing - mirrors sonic_ssm_train.py's conventions (atomic write,
# strict resume-guard) but does not serialize lane state (see module
# docstring's "Checkpoint/resume" section).
# ----------------------------------------------------------------------------


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    return {"vocab_size": len(tokenizer), "name_or_path": str(tokenizer.name_or_path)}


def split_sonic_params_for_muon(
    model: SonicSSMLanguageModel,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Splits Sonic's own parameters into (muon_params, adamw_params).

    Muon is only defined for 2D matrices, and only makes sense for a
    network's own *hidden* weight matrices - not for an embedding table (a
    per-token lookup, not a matrix a matmul-orthogonalizing update helps)
    or a projection feeding straight off one. `token_embedding.weight` is
    also the tied output/lm_head projection here, so excluding it covers
    both ends of the standard "embeddings and heads stay on Adam" split;
    `lexical_projection.weight` sits directly between the embedding and
    the first hidden layer and is excluded for the same reason. Every
    other 2D parameter is a genuine hidden-layer matrix (the per-layer
    SSM's in_proj/b_proj/c_proj/dt_proj/out_proj, and the sparse gate's
    mode_selector/mode_value) - 87.2M of Sonic's 100.6M params, confirmed
    via direct inspection of the loaded checkpoint's parameter shapes."""
    excluded_names = {"token_embedding.weight", "lexical_projection.weight"}
    muon_params = []
    adamw_params = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and name not in excluded_names:
            muon_params.append(parameter)
        else:
            adamw_params.append(parameter)
    return muon_params, adamw_params


def _migrate_legacy_optimizer_group(
    legacy_state: dict[str, Any], *, group_index: int
) -> dict[str, Any]:
    """Extracts one param_group's slice of a pre-Muon checkpoint's single
    combined-optimizer state dict, renumbered to load cleanly into a fresh
    single-group optimizer.

    Before Muon, this script used one `torch.optim.AdamW` with two groups
    (adapter, then all of Sonic's params) - PyTorch assigns each parameter
    an integer id by flattening every group's params in order, so group 0
    (the adapter) always owned ids `0..len(adapter_params)-1` regardless
    of how large group 1 (Sonic) was. That makes the adapter's slice safe
    to lift out positionally and hand to its own new optimizer unchanged."""
    ids = legacy_state["param_groups"][group_index]["params"]
    remap = {old_id: new_id for new_id, old_id in enumerate(ids)}
    new_group = dict(legacy_state["param_groups"][group_index])
    new_group["params"] = list(remap.values())
    new_state = {remap[old_id]: legacy_state["state"][old_id] for old_id in ids if old_id in legacy_state["state"]}
    return {"state": new_state, "param_groups": [new_group]}


def save_checkpoint(
    path: Path,
    *,
    update: int,
    tokens_seen: int,
    sonic_unfrozen: bool,
    consecutive_high_evals: int,
    best_eval_retrieval_accuracy: float | None,
    model: SonicSSMLanguageModel,
    model_config: SonicSSMConfig,
    port: SonicMemoryPort,
    key_projection: SonicKeyProjection,
    value_projection: SonicKeyProjection,
    adapter_optimizer: torch.optim.Optimizer,
    sonic_adamw_optimizer: torch.optim.Optimizer,
    sonic_muon_optimizer: torch.optim.Optimizer | None,
    tokenizer_identity: dict[str, Any],
    recipe: dict[str, Any],
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "update": update,
        "tokens_seen": tokens_seen,
        "sonic_unfrozen": sonic_unfrozen,
        "consecutive_high_evals": consecutive_high_evals,
        "best_eval_retrieval_accuracy": best_eval_retrieval_accuracy,
        "model": model.state_dict(),
        "model_config": model_config.to_dict(),
        "port": port.state_dict(),
        "key_projection": key_projection.state_dict(),
        "value_projection": value_projection.state_dict(),
        "adapter_optimizer": adapter_optimizer.state_dict(),
        "sonic_adamw_optimizer": sonic_adamw_optimizer.state_dict(),
        "sonic_muon_optimizer": (
            sonic_muon_optimizer.state_dict() if sonic_muon_optimizer is not None else None
        ),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "tokenizer": tokenizer_identity,
        "recipe": recipe,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume_state(
    path: Path,
    *,
    model_config: SonicSSMConfig,
    tokenizer_identity: dict[str, Any],
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"resume checkpoint format {state.get('format')!r} does not match "
            f"expected {CHECKPOINT_FORMAT!r} - refusing to resume into a mismatch"
        )
    if state.get("model_config") != model_config.to_dict():
        raise ValueError(
            "resume checkpoint's model_config does not match the freshly-"
            "constructed model_config - refusing to resume into a mismatch"
        )
    if state.get("tokenizer") != tokenizer_identity:
        raise ValueError(
            "resume checkpoint's tokenizer identity does not match the "
            "current tokenizer - refusing to resume into a mismatch"
        )
    return state


# ----------------------------------------------------------------------------
# Evaluation.
# ----------------------------------------------------------------------------


def run_eval(
    *,
    eval_lanes: list[Lane],
    model: SonicSSMLanguageModel,
    device: str,
    port: SonicMemoryPort,
    key_projection: SonicKeyProjection,
    value_projection: SonicKeyProjection,
    window_tokens: int,
    pad_token_id: int,
    temperature: float,
    recency_weight: float,
    recency_decay: float,
    value_source: str,
    mixed_precision: str,
    eval_windows: int,
    ce_chunk_rows: int,
    compile_config: CompileConfig,
) -> dict[str, Any]:
    outcomes: list[QueryOutcome] = []
    due_query_windows = 0
    total_windows = 0
    with torch.no_grad():
        for _ in range(eval_windows):
            step = run_step(
                lanes=eval_lanes,
                model=model,
                device=device,
                port=port,
                key_projection=key_projection,
                value_projection=value_projection,
                window_tokens=window_tokens,
                pad_token_id=pad_token_id,
                temperature=temperature,
                recency_weight=recency_weight,
                recency_decay=recency_decay,
                value_source=value_source,
                mixed_precision=mixed_precision,
                sonic_requires_grad=False,
                ce_chunk_rows=ce_chunk_rows,
                compile_config=compile_config,
                track_accuracy=True,
            )
            outcomes.extend(step.query_outcomes)
            due_query_windows += step.windows_with_due_query
            total_windows += step.total_windows

    value_outcomes = [o for o in outcomes if o.value_correct is not None]
    return {
        "eval_retrieval_accuracy": (
            sum(o.retrieval_correct for o in outcomes) / len(outcomes) if outcomes else None
        ),
        "eval_answer_accuracy": (
            sum(o.answer_correct for o in outcomes) / len(outcomes) if outcomes else None
        ),
        "eval_value_accuracy": (
            sum(o.value_correct for o in value_outcomes) / len(value_outcomes)
            if value_outcomes
            else None
        ),
        "eval_query_count": len(outcomes),
        "eval_due_query_window_fraction": (
            due_query_windows / total_windows if total_windows else None
        ),
    }


# ----------------------------------------------------------------------------
# Training loop.
# ----------------------------------------------------------------------------


def wsd_lr_multiplier(
    update: int, *, total_steps: int, warmup_steps: int, decay_steps: int, min_lr_ratio: float
) -> float:
    """Warmup-Stable-Decay: linear ramp 0->1 over the first `warmup_steps`,
    held at 1.0 through the (potentially very long, accuracy-gated-unfreeze-
    length-dependent) stable middle, linear decay 1.0->`min_lr_ratio` over
    the last `decay_steps`. Chosen over a cosine schedule specifically
    because of the staged-unfreeze design: a cosine curve needs to know its
    full shape in advance, but the adapter-only stable phase here runs
    until an accuracy threshold fires, not a predetermined step - WSD's
    flat stable region tolerates that being any length without needing to
    guess it upfront, and decay only has to cover the final, bounded
    stretch of the run. `decay_steps=0` disables decay entirely (pure
    warmup + stable)."""
    if warmup_steps > 0 and update <= warmup_steps:
        return update / warmup_steps
    decay_start = total_steps - decay_steps
    if decay_steps > 0 and update > decay_start:
        progress = min(1.0, (update - decay_start) / decay_steps)
        return 1.0 - progress * (1.0 - min_lr_ratio)
    return 1.0


class _RunLogger:
    """Writes each message to stdout (so live `tail`-based monitoring keeps
    working unchanged) and, if a path was given, appends it to a dedicated
    log file too - separate files for training-step and eval lines, so the
    two don't need to be grepped apart after the fact to document a run."""

    def __init__(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8") if path is not None else None

    def __call__(self, text: str) -> None:
        print(text, flush=True)
        if self._handle is not None:
            self._handle.write(text + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "This script requires the qwen extra (shared tokenizer dependency): "
            "python -m pip install -e '.[qwen]'"
        ) from error
    return transformers


def run(args: argparse.Namespace) -> None:
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    best_checkpoint = args.best_checkpoint or args.output_checkpoint.with_name(
        args.output_checkpoint.stem + "_best" + args.output_checkpoint.suffix
    )
    log_file = args.log_file or args.output_checkpoint.parent / "train.log"
    eval_log_file = args.eval_log_file or args.output_checkpoint.parent / "eval.log"
    train_log = _RunLogger(log_file)
    eval_log = _RunLogger(eval_log_file)

    transformers = _require_transformers()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model_config = SonicSSMConfig(**checkpoint["model_config"])
    model = SonicSSMLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    configure_float32_runtime(
        Float32RuntimeConfig(matmul_precision=args.matmul_precision), device
    )
    model.configure_execution(
        scan_backend=args.scan_backend, packed_projections=args.packed_projections
    )
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(every=args.checkpoint_every_layers)
    compile_config = CompileConfig(enabled=args.compile != "none", mode=args.compile_mode)
    if args.compile == "layers":
        model.compile_layers(mode=args.compile_mode)
    elif args.compile == "blocks":
        model.compile_ssm_blocks(mode=args.compile_mode)

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    if len(tokenizer) != model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocab size {len(tokenizer)} does not match "
            f"model vocab_size {model_config.vocab_size}"
        )
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    tokenizer_identity = _tokenizer_identity(tokenizer)

    torch.manual_seed(args.port_seed)
    port = SonicMemoryPort(
        SonicMemoryPortConfig(
            d_model=model_config.d_model,
            memory_dim=model_config.d_model,
            rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
        )
    ).to(device)
    key_projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=model_config.d_model)).to(
        device
    )
    value_projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=model_config.d_model)).to(
        device
    )
    # Compiling the adapter (port/key_projection/value_projection) was
    # tried and measured worse, not better: its batch dimension is the
    # number of due write/quantity/query/answer jobs in a window, which
    # varies with real session content, so a dynamic=True compiled graph
    # pays per-call guard-check overhead these small (one/two-Linear-layer)
    # modules don't have enough compute to amortize. A live GPU comparison
    # (same resume point, same steps) measured ~13.1s/update compiled vs.
    # ~12.1-12.4s/update eager - reverted rather than kept as a --compile
    # option, since it was strictly worse in the one place it was tested.

    use_fused_adamw = args.fused_adamw and device == "cuda"
    adapter_optimizer = torch.optim.AdamW(
        list(port.parameters()) + list(key_projection.parameters()) + list(value_projection.parameters()),
        lr=args.learning_rate,
        weight_decay=0.0,
        fused=use_fused_adamw,
    )

    # Sonic's own params join their optimizer(s) from step 1, at their own
    # (much smaller) LR, but stay frozen (requires_grad_(False) above)
    # until the accuracy-gated trigger below fires - both AdamW and Muon
    # safely skip params with no gradient in the meantime (see run()'s own
    # staged-unfreeze design, matching sonic_noun_target_probe.py's).
    #
    # Split three ways rather than the adapter/Sonic two-way split this
    # script used before Muon: Muon only applies to Sonic's own 2D hidden
    # weight matrices (see split_sonic_params_for_muon's docstring for why
    # embeddings/lexical_projection are excluded) - everything else Sonic
    # owns (norms, biases, the scalar sparse-gate temperature) still needs
    # an elementwise optimizer, so it joins its own AdamW group instead of
    # Muon's, separate from the adapter's AdamW group so the adapter's own
    # optimizer state stays untouched by this change.
    sonic_muon_params, sonic_adamw_params = split_sonic_params_for_muon(model)
    if not args.use_muon:
        sonic_adamw_params = sonic_adamw_params + sonic_muon_params
        sonic_muon_params = []
    sonic_adamw_optimizer = torch.optim.AdamW(
        sonic_adamw_params, lr=args.sonic_learning_rate, weight_decay=0.0, fused=use_fused_adamw
    )
    sonic_muon_optimizer = (
        Muon(
            sonic_muon_params,
            lr=args.sonic_muon_lr,
            momentum=args.muon_momentum,
            nesterov=True,
            ns_steps=args.muon_ns_steps,
            weight_decay=args.muon_weight_decay,
        )
        if sonic_muon_params
        else None
    )

    recipe = {
        "window_tokens": args.window_tokens,
        "num_lanes": args.num_lanes,
        "learning_rate": args.learning_rate,
        "sonic_learning_rate": args.sonic_learning_rate,
        "warmup_steps": args.warmup_steps,
        "decay_steps": args.decay_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "unfreeze_accuracy_threshold": args.unfreeze_accuracy_threshold,
        "unfreeze_patience": args.unfreeze_patience,
        "plain_lm_loss_weight": args.plain_lm_loss_weight,
        "distractor_loss_weight": args.distractor_loss_weight,
        "value_source": args.value_source,
        "scan_backend": args.scan_backend,
        "activation_checkpointing": args.activation_checkpointing,
        "checkpoint_every_layers": args.checkpoint_every_layers,
        "fused_adamw": use_fused_adamw,
        "grad_accum_steps": args.grad_accum_steps,
        "use_muon": bool(sonic_muon_params),
        "sonic_muon_lr": args.sonic_muon_lr,
        "muon_momentum": args.muon_momentum,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_weight_decay": args.muon_weight_decay,
        "sonic_muon_param_count": len(sonic_muon_params),
        "sonic_adamw_param_count": len(sonic_adamw_params),
    }

    update = 0
    tokens_seen = 0
    sonic_unfrozen = False
    consecutive_high_evals = 0
    best_eval_retrieval_accuracy: float | None = None

    if args.resume_checkpoint is not None:
        state = load_resume_state(
            args.resume_checkpoint, model_config=model_config, tokenizer_identity=tokenizer_identity
        )
        model.load_state_dict(state["model"])
        port.load_state_dict(state["port"])
        key_projection.load_state_dict(state["key_projection"])
        value_projection.load_state_dict(state["value_projection"])

        def _fix_fused_device(adamw_optimizer: torch.optim.Optimizer) -> None:
            # load_state_dict overwrites every non-params key (including
            # "fused") with whatever the checkpoint's optimizer had at save
            # time - restore the requested fused setting rather than
            # silently inheriting the old run's value, and move any
            # CPU-resident "step" state (from a checkpoint saved under
            # non-fused Adam/AdamW) onto each param's own device, which the
            # fused kernel requires.
            for group in adamw_optimizer.param_groups:
                group["fused"] = use_fused_adamw
                if use_fused_adamw:
                    for parameter in group["params"]:
                        param_state = adamw_optimizer.state.get(parameter)
                        if param_state and "step" in param_state:
                            param_state["step"] = param_state["step"].to(parameter.device)

        if "adapter_optimizer" in state:
            # Current (post-Muon) checkpoint format: each of the three
            # optimizers has its own independently-saved state.
            adapter_optimizer.load_state_dict(state["adapter_optimizer"])
            _fix_fused_device(adapter_optimizer)
            if state.get("sonic_adamw_optimizer") is not None:
                sonic_adamw_optimizer.load_state_dict(state["sonic_adamw_optimizer"])
                _fix_fused_device(sonic_adamw_optimizer)
            if sonic_muon_optimizer is not None and state.get("sonic_muon_optimizer") is not None:
                sonic_muon_optimizer.load_state_dict(state["sonic_muon_optimizer"])
            # Any other combination (e.g. resuming a --no-muon checkpoint
            # with Muon now enabled, or vice versa) leaves that side fresh -
            # its params never had continuable state for the requested
            # optimizer anyway.
        else:
            # Pre-Muon checkpoint format: one AdamW instance, param_groups
            # = [adapter, all of Sonic's params]. The adapter's slice (see
            # _migrate_legacy_optimizer_group's docstring) is the only part
            # worth carrying over - Sonic's own group was always frozen in
            # every run this checkpoint format was used for, so its saved
            # Adam state was still at its untouched zero-initialized value;
            # sonic_adamw_optimizer/sonic_muon_optimizer simply start fresh.
            adapter_optimizer.load_state_dict(
                _migrate_legacy_optimizer_group(state["optimizer"], group_index=0)
            )
            _fix_fused_device(adapter_optimizer)
        update = state["update"]
        tokens_seen = state["tokens_seen"]
        sonic_unfrozen = state["sonic_unfrozen"]
        consecutive_high_evals = state["consecutive_high_evals"]
        best_eval_retrieval_accuracy = state["best_eval_retrieval_accuracy"]
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if sonic_unfrozen:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        print(f"resumed from {args.resume_checkpoint} at update={update}", flush=True)

    train_streams = [
        SessionStream(args.shard_dir, seed=args.data_seed + lane_index)
        for lane_index in range(args.num_lanes)
    ]
    train_pool = PrefetchingSessionPool(
        train_streams,
        window_tokens=args.window_tokens,
        pad_token_id=pad_token_id,
        num_workers=args.session_prefetch_workers,
        buffer_size=args.session_prefetch_buffer,
    )
    train_lanes = [
        new_lane(train_pool.handle(lane_index), model_config.d_model)
        for lane_index in range(args.num_lanes)
    ]

    eval_shard_dir = args.eval_shard_dir or args.shard_dir
    eval_streams = [
        SessionStream(eval_shard_dir, seed=args.eval_seed + lane_index)
        for lane_index in range(args.eval_lanes)
    ]
    eval_pool = PrefetchingSessionPool(
        eval_streams,
        window_tokens=args.window_tokens,
        pad_token_id=pad_token_id,
        num_workers=args.session_prefetch_workers,
        buffer_size=args.session_prefetch_buffer,
    )
    eval_lanes_list = [
        new_lane(eval_pool.handle(lane_index), model_config.d_model)
        for lane_index in range(args.eval_lanes)
    ]

    history: list[dict[str, Any]] = []
    start_time = time.monotonic()
    last_file_log = start_time
    heartbeat_every = max(1, min(args.eval_every, 20))
    profile_updates_seen = 0
    profile_updates_done = 0

    while update < args.steps:
        update += 1

        if (
            not sonic_unfrozen
            and args.unfreeze_sonic_at_step is not None
            and update >= args.unfreeze_sonic_at_step
        ):
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            sonic_unfrozen = True
            train_log(f"unfreezing Sonic at update {update} (fixed-step trigger)")

        lr_multiplier = wsd_lr_multiplier(
            update,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        adapter_optimizer.param_groups[0]["lr"] = args.learning_rate * lr_multiplier
        sonic_adamw_optimizer.param_groups[0]["lr"] = args.sonic_learning_rate * lr_multiplier
        if sonic_muon_optimizer is not None:
            sonic_muon_optimizer.param_groups[0]["lr"] = args.sonic_muon_lr * lr_multiplier

        sonic_requires_grad = sonic_unfrozen
        adapter_optimizer.zero_grad(set_to_none=True)
        sonic_adamw_optimizer.zero_grad(set_to_none=True)
        if sonic_muon_optimizer is not None:
            sonic_muon_optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation: each "update" is one optimizer step over
        # args.grad_accum_steps micro-batches, each pulling a fresh window
        # per lane (lanes' persistent BasinMemory state carries across
        # micro-batches exactly as it does across ordinary steps) - this
        # reaches a larger effective batch size without needing that many
        # lanes to fit in VRAM simultaneously (vanilla Sonic itself trains
        # at physical batch=6, accum=8, effective=48).
        # Accumulate on-GPU (no .item()/isfinite host sync per micro-step -
        # each one forces the CPU to block until the GPU pipeline drains,
        # which is exactly the kind of per-step stall that shows up as
        # burstiness in a live utilization histogram even when it barely
        # moves an averaged tok/s number). Sync exactly once per update,
        # after all micro-steps' backward() calls are queued.
        loss_sum = torch.zeros((), device=device)
        plain_lm_sum = torch.zeros((), device=device)
        distractor_sum = torch.zeros((), device=device)

        # Optional operator-level profiling of a few steady-state unfrozen
        # updates (skipping the first few post-unfreeze updates, which
        # include one-off recompilation for the newly grad-tracked path).
        profiling_this_update = (
            args.profile_steps > 0
            and sonic_requires_grad
            and profile_updates_seen >= args.profile_skip_updates
            and profile_updates_done < args.profile_steps
        )
        if sonic_requires_grad:
            profile_updates_seen += 1
        profiler_ctx = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            if profiling_this_update
            else contextlib.nullcontext()
        )
        with profiler_ctx as prof:
            for micro_step in range(args.grad_accum_steps):
                step = run_step(
                    lanes=train_lanes,
                    model=model,
                    device=device,
                    port=port,
                    key_projection=key_projection,
                    value_projection=value_projection,
                    window_tokens=args.window_tokens,
                    pad_token_id=pad_token_id,
                    temperature=args.temperature,
                    recency_weight=args.recency_weight,
                    recency_decay=args.recency_decay,
                    value_source=args.value_source,
                    mixed_precision=args.mixed_precision,
                    sonic_requires_grad=sonic_requires_grad,
                    ce_chunk_rows=args.ce_chunk_rows,
                    compile_config=compile_config,
                    track_accuracy=False,
                )
                micro_loss = (
                    args.plain_lm_loss_weight * step.plain_lm_loss
                    + args.distractor_loss_weight * step.distractor_loss
                )
                (micro_loss / args.grad_accum_steps).backward()
                loss_sum = loss_sum + micro_loss.detach()
                plain_lm_sum = plain_lm_sum + step.plain_lm_loss.detach()
                distractor_sum = distractor_sum + step.distractor_loss.detach()

        if profiling_this_update:
            profile_updates_done += 1
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
            train_log(f"=== profiler table (update {update}) ===\n{table}")
            if args.profile_trace_dir is not None:
                args.profile_trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = args.profile_trace_dir / f"update_{update}.json"
                prof.export_chrome_trace(str(trace_path))
                train_log(f"profiler trace written to {trace_path}")
            if profile_updates_done >= args.profile_steps:
                train_log("profile-steps reached - exiting early")
                train_pool.shutdown()
                eval_pool.shutdown()
                return

        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f"non-finite loss at update {update}: {loss_sum.item()}")
        adapter_optimizer.step()
        sonic_adamw_optimizer.step()
        if sonic_muon_optimizer is not None:
            sonic_muon_optimizer.step()
        tokens_seen += args.num_lanes * args.window_tokens * args.grad_accum_steps

        # Defer the loss .item() sync to logging updates only, same as the
        # single-step loop did before gradient accumulation existed - most
        # updates skip it entirely, which matters for keeping the GPU
        # pipeline from stalling on a host round-trip every single update.
        should_log_heartbeat = update == 1 or update % heartbeat_every == 0
        should_log_regular = update % args.log_every == 0 or update == args.steps
        if should_log_heartbeat or should_log_regular:
            loss_value = loss_sum.item() / args.grad_accum_steps
            plain_lm_value = plain_lm_sum.item() / args.grad_accum_steps
            distractor_value = distractor_sum.item() / args.grad_accum_steps

        if should_log_heartbeat:
            muon_lr_note = (
                f" muon_lr={sonic_muon_optimizer.param_groups[0]['lr']:.6g}"
                if sonic_muon_optimizer is not None
                else ""
            )
            train_log(
                f"update {update}/{args.steps} loss={loss_value:.4f} "
                f"plain_lm={plain_lm_value:.4f} "
                f"distractor={distractor_value:.4f} "
                f"lr={adapter_optimizer.param_groups[0]['lr']:.6g}{muon_lr_note} "
                f"tokens_seen={tokens_seen} "
                f"elapsed={time.monotonic() - start_time:.1f}s"
            )

        if should_log_regular:
            train_log(
                f"loss={loss_value:.6f} ppl={math.exp(plain_lm_value):.3f} "
                f"lr={adapter_optimizer.param_groups[0]['lr']:.6g} "
                f"tokens={tokens_seen} update={update}"
            )

        now = time.monotonic()
        if args.log_every_seconds > 0 and now - last_file_log >= args.log_every_seconds:
            # Wall-clock heartbeat independent of --log-every's update-count
            # cadence, matching sonic_ssm_train.py's --log-every-seconds -
            # guarantees progress is visible even if a single step is slow
            # enough that the update-count cadence alone would go quiet for
            # a long stretch.
            train_log(
                f"[heartbeat] update {update}/{args.steps} elapsed={now - start_time:.1f}s "
                f"tokens_seen={tokens_seen}"
            )
            last_file_log = now

        if update % args.eval_every == 0 or update == args.steps:
            eval_row = run_eval(
                eval_lanes=eval_lanes_list,
                model=model,
                device=device,
                port=port,
                key_projection=key_projection,
                value_projection=value_projection,
                window_tokens=args.window_tokens,
                pad_token_id=pad_token_id,
                temperature=args.temperature,
                recency_weight=args.recency_weight,
                recency_decay=args.recency_decay,
                value_source=args.value_source,
                mixed_precision=args.mixed_precision,
                eval_windows=args.eval_windows,
                ce_chunk_rows=args.ce_chunk_rows,
                compile_config=compile_config,
            )
            eval_row["update"] = update
            history.append(eval_row)
            retrieval_accuracy = eval_row["eval_retrieval_accuracy"]
            is_new_best = retrieval_accuracy is not None and (
                best_eval_retrieval_accuracy is None or retrieval_accuracy > best_eval_retrieval_accuracy
            )
            if is_new_best:
                best_eval_retrieval_accuracy = retrieval_accuracy

            streak_note = ""
            if args.unfreeze_accuracy_threshold is not None and not sonic_unfrozen:
                # Gate the unfreeze on the adapter's own retrieval reliability
                # actually converging, not a fixed step count -
                # sonic_noun_target_probe.py's identical design: requires
                # `--unfreeze-patience` consecutive eval checkpoints at or
                # above threshold before firing, so one lucky noisy
                # checkpoint can't trigger it early.
                if (
                    retrieval_accuracy is not None
                    and retrieval_accuracy >= args.unfreeze_accuracy_threshold
                ):
                    consecutive_high_evals += 1
                else:
                    consecutive_high_evals = 0
                streak_note = f" unfreeze_streak={consecutive_high_evals}/{args.unfreeze_patience}"
                if consecutive_high_evals >= args.unfreeze_patience:
                    for parameter in model.parameters():
                        parameter.requires_grad_(True)
                    sonic_unfrozen = True
                    train_log(
                        f"unfreezing Sonic at update {update} "
                        f"(retrieval_acc={retrieval_accuracy:.3f} held for "
                        f"{args.unfreeze_patience} evals, "
                        f"sonic_learning_rate={args.sonic_learning_rate})"
                    )

            eval_log(
                f"eval update={update}/{args.steps} "
                f"retrieval_acc={eval_row['eval_retrieval_accuracy']} "
                f"answer_acc={eval_row['eval_answer_accuracy']} "
                f"value_acc={eval_row['eval_value_accuracy']} "
                f"n={eval_row['eval_query_count']} "
                f"due_query_window_frac={eval_row['eval_due_query_window_fraction']}"
                f"{streak_note}"
                f"{' NEW_BEST' if is_new_best else ''}"
            )

            if is_new_best:
                save_checkpoint(
                    best_checkpoint,
                    update=update,
                    tokens_seen=tokens_seen,
                    sonic_unfrozen=sonic_unfrozen,
                    consecutive_high_evals=consecutive_high_evals,
                    best_eval_retrieval_accuracy=best_eval_retrieval_accuracy,
                    model=model,
                    model_config=model_config,
                    port=port,
                    key_projection=key_projection,
                    value_projection=value_projection,
                    adapter_optimizer=adapter_optimizer,
                    sonic_adamw_optimizer=sonic_adamw_optimizer,
                    sonic_muon_optimizer=sonic_muon_optimizer,
                    tokenizer_identity=tokenizer_identity,
                    recipe=recipe,
                )
                train_log(
                    f"new best checkpoint (retrieval_acc={best_eval_retrieval_accuracy:.3f}) "
                    f"saved at update {update} -> {best_checkpoint}"
                )

        if update % args.checkpoint_every_updates == 0 or update == args.steps:
            save_checkpoint(
                args.output_checkpoint,
                update=update,
                tokens_seen=tokens_seen,
                sonic_unfrozen=sonic_unfrozen,
                consecutive_high_evals=consecutive_high_evals,
                best_eval_retrieval_accuracy=best_eval_retrieval_accuracy,
                model=model,
                model_config=model_config,
                port=port,
                key_projection=key_projection,
                value_projection=value_projection,
                adapter_optimizer=adapter_optimizer,
                sonic_adamw_optimizer=sonic_adamw_optimizer,
                sonic_muon_optimizer=sonic_muon_optimizer,
                tokenizer_identity=tokenizer_identity,
                recipe=recipe,
            )
            train_log(f"checkpoint saved at update {update} -> {args.output_checkpoint}")

        elapsed = time.monotonic() - start_time
        if args.max_runtime_seconds > 0 and elapsed >= args.max_runtime_seconds:
            train_log(
                f"max_runtime_seconds ({args.max_runtime_seconds}) reached at "
                f"update {update} - force-saving and exiting"
            )
            save_checkpoint(
                args.output_checkpoint,
                update=update,
                tokens_seen=tokens_seen,
                sonic_unfrozen=sonic_unfrozen,
                consecutive_high_evals=consecutive_high_evals,
                best_eval_retrieval_accuracy=best_eval_retrieval_accuracy,
                model=model,
                model_config=model_config,
                port=port,
                key_projection=key_projection,
                value_projection=value_projection,
                adapter_optimizer=adapter_optimizer,
                sonic_adamw_optimizer=sonic_adamw_optimizer,
                sonic_muon_optimizer=sonic_muon_optimizer,
                tokenizer_identity=tokenizer_identity,
                recipe=recipe,
            )
            break

    train_pool.shutdown()
    eval_pool.shutdown()

    if args.history_output is not None:
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        args.history_output.write_text(
            json.dumps({"recipe": recipe, "history": history}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    train_log.close()
    eval_log.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--eval-shard-dir", type=Path, default=None)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=None,
        help=(
            "Saved whenever eval_retrieval_accuracy hits a new best, "
            "separate from --output-checkpoint's latest-only save. Default: "
            "<output-checkpoint stem>_best<suffix> alongside it."
        ),
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--history-output", type=Path, default=None)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Training-step log lines are appended here (as well as printed "
        "to stdout). Default: train.log next to --output-checkpoint.",
    )
    parser.add_argument(
        "--eval-log-file",
        type=Path,
        default=None,
        help="Eval-result log lines are appended here (as well as printed "
        "to stdout), kept separate from --log-file so a run's training "
        "curve and its eval curve don't need to be grepped apart "
        "afterward. Default: eval.log next to --output-checkpoint.",
    )

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--window-tokens", type=int, default=PHYSICAL_WINDOW_TOKENS)
    parser.add_argument("--num-lanes", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--eval-lanes", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-windows", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--sonic-learning-rate",
        type=float,
        default=1e-5,
        help="AdamW LR for Sonic's own non-2D params (embeddings, norms, "
        "biases, the scalar sparse-gate temperature) - its 2D hidden "
        "weight matrices use --sonic-muon-lr instead unless --no-muon",
    )
    parser.add_argument(
        "--sonic-muon-lr",
        type=float,
        default=0.02,
        help="Muon LR for Sonic's own 2D hidden weight matrices "
        "(in_proj/b_proj/c_proj/dt_proj/out_proj, sparse-gate mode_selector/"
        "mode_value) - deliberately a different scale than --sonic-learning-"
        "rate, since Muon's orthogonalized update has different magnitude "
        "properties than Adam's per-element one",
    )
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--no-muon",
        dest="use_muon",
        action="store_false",
        default=True,
        help="fall back to routing Sonic's 2D hidden weight matrices through "
        "the sonic_learning_rate AdamW optimizer instead of Muon",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="WSD (warmup-stable-decay) schedule: linear ramp 0->peak LR "
        "over this many updates. Applied as a multiplier to both "
        "--learning-rate and --sonic-learning-rate.",
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=0,
        help="WSD schedule: linear decay from peak to --min-lr-ratio*peak "
        "over the last this-many updates of the run. 0 disables decay "
        "(pure warmup + stable) - the staged-unfreeze design means the "
        "stable phase's length is accuracy-gated, not known in advance, so "
        "decay is anchored to the end of the run rather than a fixed "
        "cosine shape covering the whole thing.",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument(
        "--unfreeze-accuracy-threshold",
        type=float,
        default=0.95,
        help=(
            "Unfreeze Sonic once eval_retrieval_accuracy has been at or above "
            "this threshold for --unfreeze-patience consecutive eval "
            "checkpoints. Pass a negative value together with "
            "--unfreeze-sonic-at-step to use the fixed-step trigger instead."
        ),
    )
    parser.add_argument("--unfreeze-patience", type=int, default=3)
    parser.add_argument("--unfreeze-sonic-at-step", type=int, default=None)

    parser.add_argument("--plain-lm-loss-weight", type=float, default=0.6)
    parser.add_argument("--distractor-loss-weight", type=float, default=0.4)
    parser.add_argument("--retrieval-temperature", "--temperature", dest="temperature", type=float, default=0.1)
    parser.add_argument("--recency-weight", type=float, default=0.3)
    parser.add_argument("--recency-decay", type=float, default=0.7)
    parser.add_argument("--value-source", choices=("full", "quantity"), default="quantity")

    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--port-seed", type=int, default=17)
    parser.add_argument("--data-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=31)
    parser.add_argument(
        "--session-prefetch-workers",
        type=int,
        default=8,
        help=(
            "Shared background-thread pool size for PrefetchingSessionPool - "
            "independent of --num-lanes/--eval-lanes, since one session "
            "read+parse is fast enough that a small shared pool keeps up "
            "with many lanes' buffers."
        ),
    )
    parser.add_argument(
        "--session-prefetch-buffer",
        type=int,
        default=2,
        help="Sessions kept pre-fetched ahead of consumption, per lane.",
    )

    parser.add_argument("--ce-chunk-rows", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-every-seconds", type=float, default=60.0)
    parser.add_argument("--checkpoint-every-updates", type=int, default=200)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)

    parser.add_argument("--compile", choices=("none", "layers", "blocks"), default="none")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--mixed-precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--scan-backend", choices=("doubling", "hgrn"), default="doubling")
    parser.add_argument("--packed-projections", action="store_true", default=True)
    parser.add_argument(
        "--no-packed-projections", dest="packed_projections", action="store_false"
    )
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="highest")
    parser.add_argument("--activation-checkpointing", action="store_true", default=False)
    parser.add_argument("--checkpoint-every-layers", type=int, default=1)
    parser.add_argument("--fused-adamw", action="store_true", default=True)
    parser.add_argument("--no-fused-adamw", dest="fused_adamw", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="if >0, profile this many steady-state unfrozen updates with "
        "torch.profiler, log an operator-time table, then exit early",
    )
    parser.add_argument("--profile-skip-updates", type=int, default=2)
    parser.add_argument("--profile-trace-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.unfreeze_accuracy_threshold is not None and args.unfreeze_accuracy_threshold < 0:
        args.unfreeze_accuracy_threshold = None
    try:
        run(args)
    except BaseException:
        sys.stdout.flush()
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
