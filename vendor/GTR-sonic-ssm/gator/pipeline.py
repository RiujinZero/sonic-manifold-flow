"""Deterministic CPU sampling and device-transfer pipelines.

The pipeline deliberately keeps the prefetch queue on the CPU.  A checkpoint
therefore has a simple, portable ordering contract:

1. ``queued_cpu_batches`` contains the next batches in consumption order.
2. ``provider_state`` describes the provider *after* those batches were
   sampled.

Restoring both pieces reproduces the exact next-batch sequence when the
provider implements ``state_dict`` and ``load_state_dict`` and advertises
``exact_state_supported``. Providers without that optional protocol can still
be used for serial or prefetched training, but restoring their already queued
portion requires the caller to opt into ``require_exact=False``.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import torch

from .tasks.language import LanguageBatch


BatchPipelineMode = Literal["serial", "prefetch"]


@runtime_checkable
class LanguageBatchProvider(Protocol):
    """Minimum interface consumed by :class:`LanguageBatchPipeline`."""

    def sample_cpu(self, batch_size: int) -> LanguageBatch:
        """Return a freshly sampled CPU batch."""


@dataclass(frozen=True)
class BatchPipelineConfig:
    """Configuration for deterministic sampling and transfer.

    ``prefetch_batches`` is a hard upper bound on the number of CPU batches
    held by the producer.  It is ignored in serial mode, but remains validated
    so switching modes cannot silently create an invalid configuration.
    """

    batch_size: int
    device: torch.device | str = "cpu"
    mode: BatchPipelineMode = "serial"
    prefetch_batches: int = 2
    pin_memory: bool = False
    non_blocking: bool = True
    dedicated_cuda_stream: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.mode not in ("serial", "prefetch"):
            raise ValueError(f"unsupported batch pipeline mode: {self.mode!r}")
        if self.prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive")


@dataclass(frozen=True)
class BatchPipelineStats:
    """Point-in-time pipeline counters."""

    mode: BatchPipelineMode
    sampled_batches: int
    consumed_batches: int
    restored_batches: int
    device_transfers: int
    producer_failures: int
    producer_waits: int
    queue_depth: int
    queue_capacity: int
    queue_high_watermark: int
    sample_seconds: float
    transfer_seconds: float
    worker_alive: bool
    closed: bool


def _batch_to_state(batch: LanguageBatch) -> dict[str, Any]:
    """Clone a batch into an unpinned, CPU-only checkpoint representation."""

    result: dict[str, Any] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            result[field.name] = value.detach().cpu().clone()
        else:
            result[field.name] = copy.deepcopy(value)
    return result


def _batch_from_state(state: Mapping[str, Any]) -> LanguageBatch:
    values: dict[str, Any] = {}
    expected = {field.name for field in fields(LanguageBatch)}
    missing = expected.difference(state)
    if missing:
        raise ValueError(
            "queued LanguageBatch state is missing fields: "
            + ", ".join(sorted(missing))
        )
    for field in fields(LanguageBatch):
        value = state[field.name]
        if isinstance(value, torch.Tensor):
            values[field.name] = value.detach().cpu().clone()
        else:
            values[field.name] = copy.deepcopy(value)
    return LanguageBatch(**values)


def _validate_cpu_batch(batch: LanguageBatch) -> None:
    if not isinstance(batch, LanguageBatch):
        raise TypeError(
            "sample_cpu() must return LanguageBatch, "
            f"received {type(batch).__name__}"
        )
    non_cpu = [
        field.name
        for field in fields(batch)
        if isinstance((value := getattr(batch, field.name)), torch.Tensor)
        and value.device.type != "cpu"
    ]
    if non_cpu:
        raise ValueError(
            "sample_cpu() returned non-CPU tensors in fields: "
            + ", ".join(non_cpu)
        )


class LanguageBatchPipeline:
    """Serial or bounded single-producer language-batch pipeline.

    The class is both an iterator and a context manager.  ``close`` is
    idempotent and joins the producer thread before returning.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        provider: LanguageBatchProvider,
        config: BatchPipelineConfig,
    ) -> None:
        if not callable(getattr(provider, "sample_cpu", None)):
            raise TypeError("provider must implement sample_cpu(batch_size)")
        self.provider = provider
        self.config = config
        self.device = torch.device(config.device)

        self._condition = threading.Condition()
        self._consumer_lock = threading.Lock()
        self._cpu_queue: deque[LanguageBatch] = deque()
        self._paused = False
        self._producer_busy = False
        self._closed = False
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None

        self._sampled_batches = 0
        self._consumed_batches = 0
        self._restored_batches = 0
        self._device_transfers = 0
        self._producer_failures = 0
        self._producer_waits = 0
        self._queue_high_watermark = 0
        self._sample_seconds = 0.0
        self._transfer_seconds = 0.0

        self._copy_stream: torch.cuda.Stream | None = None
        self._pending_sources: deque[
            tuple[torch.cuda.Event, LanguageBatch]
        ] = deque()
        if (
            self.device.type == "cuda"
            and config.dedicated_cuda_stream
            and torch.cuda.is_available()
        ):
            self._copy_stream = torch.cuda.Stream(device=self.device)

        if config.mode == "prefetch":
            self._start_producer()

    @property
    def stats(self) -> BatchPipelineStats:
        """Return an internally consistent statistics snapshot."""

        with self._condition:
            thread = self._thread
            return BatchPipelineStats(
                mode=self.config.mode,
                sampled_batches=self._sampled_batches,
                consumed_batches=self._consumed_batches,
                restored_batches=self._restored_batches,
                device_transfers=self._device_transfers,
                producer_failures=self._producer_failures,
                producer_waits=self._producer_waits,
                queue_depth=len(self._cpu_queue),
                queue_capacity=(
                    self.config.prefetch_batches
                    if self.config.mode == "prefetch"
                    else 0
                ),
                queue_high_watermark=self._queue_high_watermark,
                sample_seconds=self._sample_seconds,
                transfer_seconds=self._transfer_seconds,
                worker_alive=bool(thread is not None and thread.is_alive()),
                closed=self._closed,
            )

    @property
    def exact_resume_supported(self) -> bool:
        """Whether the provider declares its complete sample state restorable."""

        state_fn = getattr(self.provider, "state_dict", None)
        load_fn = getattr(self.provider, "load_state_dict", None)
        return (
            callable(state_fn)
            and callable(load_fn)
            and bool(getattr(self.provider, "exact_state_supported", False))
        )

    def __enter__(self) -> LanguageBatchPipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __iter__(self) -> LanguageBatchPipeline:
        return self

    def __next__(self) -> LanguageBatch:
        return self.next_batch()

    def _start_producer(self) -> None:
        with self._condition:
            if self._closed:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._producer_main,
                name="gator-language-batch-prefetch",
                # Explicit close still joins this worker.  Daemon mode is a
                # final safety net when an unrelated trainer exception skips
                # normal teardown.
                daemon=True,
            )
            self._thread.start()

    def _prepare_cpu_batch(self, batch: LanguageBatch) -> LanguageBatch:
        _validate_cpu_batch(batch)
        if self.config.pin_memory:
            batch = batch.pin_memory()
            _validate_cpu_batch(batch)
        return batch

    def _sample_cpu(self) -> LanguageBatch:
        started = time.perf_counter()
        try:
            batch = self.provider.sample_cpu(self.config.batch_size)
            return self._prepare_cpu_batch(batch)
        finally:
            elapsed = time.perf_counter() - started
            with self._condition:
                self._sample_seconds += elapsed

    def _producer_main(self) -> None:
        while True:
            with self._condition:
                waited = False
                while (
                    not self._closed
                    and (
                        self._paused
                        or len(self._cpu_queue) >= self.config.prefetch_batches
                    )
                ):
                    waited = True
                    self._condition.wait()
                if waited:
                    self._producer_waits += 1
                if self._closed:
                    return
                self._producer_busy = True

            try:
                batch = self._sample_cpu()
            except BaseException as error:
                with self._condition:
                    self._failure = error
                    self._producer_failures += 1
                    self._producer_busy = False
                    self._condition.notify_all()
                return

            with self._condition:
                self._producer_busy = False
                if self._closed:
                    self._condition.notify_all()
                    return
                self._cpu_queue.append(batch)
                self._sampled_batches += 1
                self._queue_high_watermark = max(
                    self._queue_high_watermark, len(self._cpu_queue)
                )
                self._condition.notify_all()

    def _serial_cpu_batch(self) -> LanguageBatch:
        with self._condition:
            while self._paused and not self._closed:
                self._condition.wait()
            if self._closed:
                raise StopIteration
            if self._cpu_queue:
                batch = self._cpu_queue.popleft()
                self._condition.notify_all()
                return batch
            self._producer_busy = True

        try:
            batch = self._sample_cpu()
        except BaseException:
            with self._condition:
                self._producer_failures += 1
            raise
        finally:
            with self._condition:
                self._producer_busy = False
                self._condition.notify_all()

        with self._condition:
            self._sampled_batches += 1
        return batch

    def _prefetched_cpu_batch(self) -> LanguageBatch:
        with self._condition:
            while True:
                if self._closed:
                    raise StopIteration
                if self._cpu_queue:
                    batch = self._cpu_queue.popleft()
                    self._condition.notify_all()
                    return batch
                if self._failure is not None:
                    raise self._failure
                self._condition.wait()

    def _release_completed_sources(self) -> None:
        while self._pending_sources and self._pending_sources[0][0].query():
            self._pending_sources.popleft()

    def _to_device(self, batch: LanguageBatch) -> LanguageBatch:
        started = time.perf_counter()
        if self._copy_stream is None:
            result = batch.to(
                self.device,
                non_blocking=self.config.non_blocking,
            )
        else:
            self._release_completed_sources()
            with torch.cuda.stream(self._copy_stream):
                result = batch.to(
                    self.device,
                    non_blocking=self.config.non_blocking,
                )
                copied = torch.cuda.Event()
                copied.record(self._copy_stream)
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_event(copied)
            for field in fields(result):
                value = getattr(result, field.name)
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    value.record_stream(current_stream)
            # Keep pinned source storage alive until the asynchronous copy has
            # actually completed.
            self._pending_sources.append((copied, batch))

        elapsed = time.perf_counter() - started
        with self._condition:
            self._device_transfers += 1
            self._transfer_seconds += elapsed
        return result

    def next_batch(self) -> LanguageBatch:
        """Return the next deterministic batch on the configured device."""

        with self._consumer_lock:
            if self.config.mode == "prefetch":
                batch = self._prefetched_cpu_batch()
            else:
                batch = self._serial_cpu_batch()
            result = self._to_device(batch)
            with self._condition:
                self._consumed_batches += 1
            return result

    def state_dict(self) -> dict[str, Any]:
        """Atomically snapshot queued CPU batches and post-queue provider state."""

        with self._condition:
            self._paused = True
            self._condition.notify_all()
            try:
                while self._producer_busy:
                    self._condition.wait()
                if self._failure is not None:
                    raise RuntimeError(
                        "cannot checkpoint a failed batch producer"
                    ) from self._failure

                state_fn = getattr(self.provider, "state_dict", None)
                load_fn = getattr(self.provider, "load_state_dict", None)
                provider_state_supported = callable(state_fn) and callable(load_fn)
                provider_state = (
                    copy.deepcopy(state_fn())
                    if provider_state_supported
                    else None
                )
                provider_payload_exact = not isinstance(
                    provider_state, Mapping
                ) or bool(provider_state.get("exact", True))
                exact_provider_state = (
                    provider_state_supported
                    and bool(
                        getattr(
                            self.provider,
                            "exact_state_supported",
                            False,
                        )
                    )
                    and provider_payload_exact
                )
                queue = [_batch_to_state(batch) for batch in self._cpu_queue]
                counters = {
                    "sampled_batches": self._sampled_batches,
                    "consumed_batches": self._consumed_batches,
                    "restored_batches": self._restored_batches,
                    "device_transfers": self._device_transfers,
                    "producer_failures": self._producer_failures,
                    "producer_waits": self._producer_waits,
                    "queue_high_watermark": self._queue_high_watermark,
                    "sample_seconds": self._sample_seconds,
                    "transfer_seconds": self._transfer_seconds,
                }
                return {
                    "version": self._STATE_VERSION,
                    "batch_size": self.config.batch_size,
                    "mode": self.config.mode,
                    "queued_cpu_batches": queue,
                    "provider_state_supported": provider_state_supported,
                    "exact_provider_state": exact_provider_state,
                    "provider_state": provider_state,
                    "counters": counters,
                }
            finally:
                self._paused = False
                self._condition.notify_all()

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        require_exact: bool = True,
    ) -> None:
        """Restore queue/provider state into an open pipeline.

        Exact restoration is required by default. Callers must explicitly set
        ``require_exact=False`` to replay a best-effort queue/provider state
        whose source cannot prove complete continuation.
        """

        version = int(state.get("version", -1))
        if version != self._STATE_VERSION:
            raise ValueError(
                f"unsupported batch pipeline state version {version}; "
                f"expected {self._STATE_VERSION}"
            )
        batch_size = int(state.get("batch_size", -1))
        if batch_size != self.config.batch_size:
            raise ValueError(
                f"batch_size mismatch: checkpoint={batch_size}, "
                f"pipeline={self.config.batch_size}"
            )

        raw_queue = state.get("queued_cpu_batches")
        if not isinstance(raw_queue, list):
            raise ValueError("queued_cpu_batches must be a list")
        if (
            self.config.mode == "prefetch"
            and len(raw_queue) > self.config.prefetch_batches
        ):
            raise ValueError(
                "checkpoint queue exceeds this pipeline's prefetch capacity"
            )
        if require_exact:
            if not bool(state.get("exact_provider_state", False)):
                raise ValueError(
                    "checkpoint provider state is not exactly restorable"
                )
            if not self.exact_resume_supported:
                raise ValueError(
                    "current provider does not support exact restoration"
                )

        restored = [_batch_from_state(item) for item in raw_queue]
        if self.config.pin_memory:
            restored = [batch.pin_memory() for batch in restored]

        restart_producer = False
        with self._condition:
            if self._closed:
                raise RuntimeError("cannot restore a closed batch pipeline")
            self._paused = True
            self._condition.notify_all()
            try:
                while self._producer_busy:
                    self._condition.wait()

                provider_state_supported = bool(
                    state.get("provider_state_supported", False)
                )
                if provider_state_supported:
                    load_fn = getattr(self.provider, "load_state_dict", None)
                    if not callable(load_fn):
                        raise TypeError(
                            "checkpoint contains provider state but this "
                            "provider has no load_state_dict"
                        )
                    load_fn(copy.deepcopy(state.get("provider_state")))

                self._cpu_queue.clear()
                self._cpu_queue.extend(restored)
                self._failure = None

                counters = state.get("counters", {})
                if not isinstance(counters, Mapping):
                    raise ValueError("pipeline counters must be a mapping")
                self._sampled_batches = int(
                    counters.get("sampled_batches", 0)
                )
                self._consumed_batches = int(
                    counters.get("consumed_batches", 0)
                )
                self._restored_batches = int(
                    counters.get("restored_batches", 0)
                ) + len(restored)
                self._device_transfers = int(
                    counters.get("device_transfers", 0)
                )
                self._producer_failures = int(
                    counters.get("producer_failures", 0)
                )
                self._producer_waits = int(
                    counters.get("producer_waits", 0)
                )
                self._queue_high_watermark = max(
                    int(counters.get("queue_high_watermark", 0)),
                    len(restored),
                )
                self._sample_seconds = float(
                    counters.get("sample_seconds", 0.0)
                )
                self._transfer_seconds = float(
                    counters.get("transfer_seconds", 0.0)
                )
                restart_producer = (
                    self.config.mode == "prefetch"
                    and (
                        self._thread is None
                        or not self._thread.is_alive()
                    )
                )
            finally:
                self._paused = False
                self._condition.notify_all()

        if restart_producer:
            self._start_producer()

    def close(self) -> None:
        """Stop and join the producer; safe to call more than once."""

        with self._condition:
            if self._closed:
                thread = self._thread
            else:
                self._closed = True
                self._paused = False
                self._condition.notify_all()
                thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join()

        if self._copy_stream is not None:
            self._copy_stream.synchronize()
            self._pending_sources.clear()
