"""Streaming ImageNet-1K ingestion via the Hugging Face `datasets` library.

Full ImageNet-1K train is ~146GB (`imagenet-scale-readiness.md`), too large
to persist locally in this project's environment. This streams a split
directly from the Hugging Face Hub, applies the frozen backbone's eval
transform, and yields batched `(image, label)` tensors compatible with
`sharded_latents.extract_sharded_latents` - so a bounded subset can be
encoded into latent shards without ever writing the raw dataset to disk.

Rows are consumed in the remote dataset's on-disk shard order (streaming
never shuffles), so repeated calls with a growing `skip` count observe a
stable prefix - required for `extract_sharded_latents`'s resume logic to
stay correct across reruns.

A large `skip()` against this gated dataset has been observed to hang
indefinitely with no error - a stalled fetch somewhere below Python's
thread visibility (only a daemon `tqdm` thread remained; the process still
needed an external kill after ~8 hours with the manifest never advancing).
`with_timeout` turns that into a loud `TimeoutError` instead of a silent
multi-hour hang; callers should wrap `streaming_imagenet_batches` with it.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, TypeVar

import torch
from torch import Tensor

T = TypeVar("T")


def batches_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    transform: Any,
    batch_size: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Batch an iterable of `{"image": PIL.Image, "label": int}` rows.

    Kept separate from the Hugging Face Hub call so it is testable with a
    small synthetic row list and no network access.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image_batch: list[Tensor] = []
    label_batch: list[int] = []
    for row in rows:
        image_batch.append(transform(row["image"].convert("RGB")))
        label_batch.append(int(row["label"]))
        if len(image_batch) == batch_size:
            yield torch.stack(image_batch), torch.tensor(label_batch, dtype=torch.long)
            image_batch, label_batch = [], []
    if image_batch:
        yield torch.stack(image_batch), torch.tensor(label_batch, dtype=torch.long)


def streaming_imagenet_batches(
    *,
    split: str,
    transform: Any,
    batch_size: int,
    token: str,
    skip: int = 0,
    take: int | None = None,
    repo_id: str = "ILSVRC/imagenet-1k",
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield batched `(image, label)` tensors streamed from the Hub.

    `skip` resumes past rows a prior call already covered; `take` bounds how
    many rows (after skipping) this call will read. Neither downloads the
    full split - `datasets` streams remote parquet shards on demand.
    """
    if skip < 0:
        raise ValueError("skip must be non-negative")
    if take is not None and take <= 0:
        raise ValueError("take must be positive when provided")
    if not token:
        raise ValueError("token must be a non-empty Hugging Face access token")
    try:
        import datasets
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Streaming ImageNet requires the `datasets` package: "
            "python -m pip install manifold-flow[stream]"
        ) from error

    stream = datasets.load_dataset(repo_id, split=split, streaming=True, token=token)
    if skip:
        stream = stream.skip(skip)
    if take is not None:
        stream = stream.take(take)
    return batches_from_rows(stream, transform=transform, batch_size=batch_size)


def with_timeout(iterable: Iterable[T], *, seconds: float) -> Iterator[T]:
    """Wrap an iterator so each `next()` call is bounded by a hard timeout.

    Runs the wrapped iterable in a background (daemon) thread and reads its
    results through a queue; if no item arrives within `seconds`, raises
    `TimeoutError` instead of blocking indefinitely. The background thread
    is not killed on timeout - Python threads cannot be force-killed - it is
    simply abandoned (daemon, so it will not block process exit on its own;
    pair this with an explicit `os._exit` after real work completes if the
    underlying library can leave non-Python-visible work running past that).
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            for item in iterable:
                result_queue.put(("item", item))
        except BaseException as error:  # noqa: BLE001 - propagated to the caller below
            result_queue.put(("error", error))
        else:
            result_queue.put(("done", None))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while True:
        try:
            kind, payload = result_queue.get(timeout=seconds)
        except queue.Empty as error:
            raise TimeoutError(
                f"no data from the stream within {seconds}s "
                "(likely a stalled network connection)"
            ) from error
        if kind == "item":
            yield payload  # type: ignore[misc]
        elif kind == "error":
            raise payload  # type: ignore[misc]
        else:
            return
