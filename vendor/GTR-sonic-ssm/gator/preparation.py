"""Ordered, bounded parallel preparation primitives.

Data preparation has different correctness constraints from model compute:
source traversal order must remain deterministic, worker failures must not be
hidden, and an eager producer must not materialize an unbounded stream.  This
module keeps those policies in one reusable API.

Workers should only perform pure, per-item work.  Coordinator-owned state such
as shuffle order, deduplication sets, acceptance statistics, and stream restart
counters must be updated by the caller in the ordered result loop.  Process
workers always use ``spawn`` so they cannot accidentally inherit CUDA state or
mutable coordinator state.  Consequently, process callables and their payloads
must be pickleable; in practice the callable should be defined at module scope.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from dataclasses import dataclass
from itertools import islice
import multiprocessing
import os
from typing import Callable, Iterable, Iterator, Literal, TypeVar

from .data import AssistantExample
from .tokenization import CompactTokenizer


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

ParallelMapBackend = Literal["serial", "threads", "processes"]

__all__ = [
    "AssistantTokenizationJob",
    "ParallelMapBackend",
    "ParallelMapConfig",
    "TokenizedAssistantExample",
    "ordered_parallel_map",
    "tokenize_assistant_example",
    "tokenize_assistant_examples",
]


@dataclass(frozen=True)
class ParallelMapConfig:
    """Execution and backpressure policy for :func:`ordered_parallel_map`.

    ``chunk_size`` bounds each submitted work unit.  ``max_pending_chunks``
    bounds how many such units may be submitted but not yet consumed.  When it
    is omitted, at most two chunks per worker are kept in flight.

    ``workers=None`` selects a conservative host-derived worker count.  The
    setting is ignored by the serial backend.
    """

    backend: ParallelMapBackend = "serial"
    workers: int | None = None
    chunk_size: int = 64
    max_pending_chunks: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"serial", "threads", "processes"}:
            raise ValueError(
                "backend must be 'serial', 'threads', or 'processes'"
            )
        if self.workers is not None and self.workers <= 0:
            raise ValueError("workers must be positive or None")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if (
            self.max_pending_chunks is not None
            and self.max_pending_chunks <= 0
        ):
            raise ValueError("max_pending_chunks must be positive or None")

    def resolved_workers(self) -> int:
        """Return the concrete worker count used by a parallel backend."""

        if self.workers is not None:
            return self.workers
        process_cpu_count = getattr(os, "process_cpu_count", None)
        available = (
            process_cpu_count()
            if process_cpu_count is not None
            else os.cpu_count()
        )
        available = available or 1
        if self.backend == "threads":
            return min(32, available + 4)
        return min(32, available)

    def resolved_pending_chunks(self) -> int:
        """Return the concrete upper bound on submitted chunks."""

        if self.max_pending_chunks is not None:
            return self.max_pending_chunks
        return max(1, 2 * self.resolved_workers())


def _apply_chunk(
    function: Callable[[InputT], OutputT],
    items: tuple[InputT, ...],
) -> tuple[OutputT, ...]:
    """Process one pickleable chunk.

    This helper is intentionally module-level so ``ProcessPoolExecutor`` can
    import it in a freshly spawned interpreter.
    """

    return tuple(function(item) for item in items)


def _next_chunk(
    items: Iterator[InputT],
    chunk_size: int,
) -> tuple[InputT, ...]:
    return tuple(islice(items, chunk_size))


def _parallel_executor(
    config: ParallelMapConfig,
    workers: int,
) -> Executor:
    if config.backend == "threads":
        return ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="gator-prepare",
        )
    if config.backend == "processes":
        # Spawn is deliberate even on Unix: preparation must not fork a
        # process after CUDA or other thread-owning runtimes are initialized.
        return ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    raise AssertionError(f"unexpected parallel backend: {config.backend!r}")


def ordered_parallel_map(
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    config: ParallelMapConfig = ParallelMapConfig(),
) -> Iterator[OutputT]:
    """Map ``function`` while preserving input order and bounded memory.

    Serial execution is lazy and maps one item at a time.  Thread and process
    execution submit fixed-size chunks through a bounded queue, but consume
    futures strictly in submission order.  Later chunks may finish first
    without reordering their results.

    Closing the returned iterator, exhausting it, or receiving an exception
    cancels work that has not begun and shuts down the executor.  A worker
    exception is re-raised when its ordered chunk is reached.  For the process
    backend, ``function``, every input, and every result must be pickleable.
    """

    if config.backend == "serial":
        for item in items:
            yield function(item)
        return

    workers = config.resolved_workers()
    pending_limit = config.resolved_pending_chunks()
    source = iter(items)
    executor = _parallel_executor(config, workers)
    pending: deque[Future[tuple[OutputT, ...]]] = deque()
    source_exhausted = False

    def submit_one() -> bool:
        nonlocal source_exhausted
        chunk = _next_chunk(source, config.chunk_size)
        if not chunk:
            source_exhausted = True
            return False
        pending.append(executor.submit(_apply_chunk, function, chunk))
        return True

    try:
        while len(pending) < pending_limit and submit_one():
            pass

        while pending:
            future = pending.popleft()
            # result() preserves worker exception type and traceback context.
            result_chunk = future.result()
            if not source_exhausted:
                submit_one()
            yield from result_chunk
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class AssistantTokenizationJob:
    """Immutable input for pure assistant-example tokenization."""

    example: AssistantExample
    tokenizer: CompactTokenizer


@dataclass(frozen=True)
class TokenizedAssistantExample:
    """Task-neutral encoded fields returned by a preparation worker."""

    conversation_id: int
    turn_index: int
    context_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    source: str | None = None


def tokenize_assistant_example(
    job: AssistantTokenizationJob,
) -> TokenizedAssistantExample:
    """Encode one assistant example without mutating stream/filter state.

    Jobs in one chunk may share the same ``CompactTokenizer`` instance.  The
    process pickler then serializes that shared object once per submitted chunk
    instead of once per example.
    """

    example = job.example
    tokenizer = job.tokenizer
    return TokenizedAssistantExample(
        conversation_id=example.conversation_id,
        turn_index=example.turn_index,
        context_ids=tuple(tokenizer.encode_turns(example.context)),
        response_ids=tuple(
            tokenizer.encode(example.response) + [tokenizer.EOS_ID]
        ),
        source=example.source,
    )


def tokenize_assistant_examples(
    examples: Iterable[AssistantExample],
    tokenizer: CompactTokenizer,
    *,
    config: ParallelMapConfig = ParallelMapConfig(),
) -> Iterator[TokenizedAssistantExample]:
    """Tokenize an ordered example stream through the selected worker pool.

    The job generator reuses one immutable tokenizer object. Chunk pickling
    preserves that shared reference, so process mode serializes the tokenizer
    once per work chunk rather than once per example.
    """

    jobs = (
        AssistantTokenizationJob(example=example, tokenizer=tokenizer)
        for example in examples
    )
    yield from ordered_parallel_map(
        tokenize_assistant_example,
        jobs,
        config=config,
    )
