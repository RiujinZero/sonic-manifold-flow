"""Simultaneous masked-response formation task."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import random
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..data import AssistantExample, ascii_letter_ratio
from ..tokenization import CompactTokenizer


@dataclass(frozen=True)
class EncodedAssistantExample:
    conversation_id: int
    turn_index: int
    context_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    source: str | None = None


@dataclass(frozen=True)
class LanguageBatchConfig:
    context_tokens: int = 64
    response_tokens: int = 32
    reference_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.context_tokens <= 0 or self.response_tokens <= 0:
            raise ValueError("context_tokens and response_tokens must be positive")
        if self.reference_tokens is not None and not (
            0 < self.reference_tokens <= self.response_tokens
        ):
            raise ValueError(
                "reference_tokens must be positive and no larger than response_tokens"
            )

    @property
    def target_tokens(self) -> int:
        return (
            self.response_tokens
            if self.reference_tokens is None
            else self.reference_tokens
        )


@dataclass(frozen=True)
class LanguageExampleFilterConfig:
    """Quality and complete-window requirements applied before collation."""

    require_complete_windows: bool = False
    require_complete_context: bool = False
    require_complete_response: bool = False
    trim_context_turns: bool = False
    require_user_prompt: bool = False
    minimum_ascii_letter_ratio: float = 0.0
    minimum_response_characters: int = 0
    deduplicate: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_ascii_letter_ratio <= 1.0:
            raise ValueError("minimum_ascii_letter_ratio must be between 0 and 1")
        if self.minimum_response_characters < 0:
            raise ValueError("minimum_response_characters cannot be negative")


@dataclass
class LanguageExampleFilterStats:
    """Mutable counters for streamed filtering diagnostics."""

    seen: int = 0
    accepted: int = 0
    rejected_user_prompt: int = 0
    rejected_response_length: int = 0
    rejected_language: int = 0
    rejected_context_window: int = 0
    trimmed_context: int = 0
    rejected_response_window: int = 0
    rejected_duplicate: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.seen, 1)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "seen": self.seen,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "rejected_user_prompt": self.rejected_user_prompt,
            "rejected_response_length": self.rejected_response_length,
            "rejected_language": self.rejected_language,
            "rejected_context_window": self.rejected_context_window,
            "trimmed_context": self.trimmed_context,
            "rejected_response_window": self.rejected_response_window,
            "rejected_duplicate": self.rejected_duplicate,
        }


def filter_language_examples(
    examples: Iterable[AssistantExample],
    tokenizer: CompactTokenizer,
    window: LanguageBatchConfig,
    config: LanguageExampleFilterConfig,
    *,
    stats: LanguageExampleFilterStats | None = None,
) -> Iterator[AssistantExample]:
    """Yield only complete, quality-gated, optionally unique examples.

    Token lengths are checked before ``LanguageBatcher`` can truncate them.
    Deduplication is scoped to one traversal of ``examples`` so a restarted
    stream can begin a new epoch without retaining an unbounded global set.
    """

    counters = stats if stats is not None else LanguageExampleFilterStats()
    seen_hashes: set[bytes] = set()
    for example in examples:
        counters.seen += 1
        if config.require_user_prompt and (
            not example.context or example.context[-1].role != "user"
        ):
            counters.rejected_user_prompt += 1
            continue
        if len(example.response.strip()) < config.minimum_response_characters:
            counters.rejected_response_length += 1
            continue
        quality_text = "\n".join(
            [*(turn.text for turn in example.context), example.response]
        )
        if ascii_letter_ratio(quality_text) < config.minimum_ascii_letter_ratio:
            counters.rejected_language += 1
            continue
        prepared = example
        context_length = len(tokenizer.encode_turns(prepared.context))
        if context_length > window.context_tokens and config.trim_context_turns:
            fitted_context = None
            for start in range(1, len(prepared.context)):
                candidate = prepared.context[start:]
                if candidate[0].role != "user":
                    continue
                if len(tokenizer.encode_turns(candidate)) <= window.context_tokens:
                    fitted_context = candidate
                    break
            if fitted_context is not None:
                prepared = replace(prepared, context=tuple(fitted_context))
                counters.trimmed_context += 1
                context_length = len(tokenizer.encode_turns(prepared.context))
        if config.require_complete_windows or config.require_complete_context:
            if context_length > window.context_tokens:
                counters.rejected_context_window += 1
                continue
        if config.require_complete_windows or config.require_complete_response:
            response_length = len(tokenizer.encode(prepared.response)) + 1
            if response_length > window.target_tokens:
                counters.rejected_response_window += 1
                continue
        if config.deduplicate:
            prepared_text = "\n".join(
                [*(turn.text for turn in prepared.context), prepared.response]
            )
            normalized = " ".join(prepared_text.casefold().split())
            digest = hashlib.blake2b(
                normalized.encode("utf-8"), digest_size=16
            ).digest()
            if digest in seen_hashes:
                counters.rejected_duplicate += 1
                continue
            seen_hashes.add(digest)
        counters.accepted += 1
        yield prepared


@dataclass(frozen=True)
class LanguageBatch:
    context_ids: torch.Tensor
    context_mask: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    target_complete: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "LanguageBatch":
        """Move every batch tensor together.

        Keeping transfer policy on the batch object gives prefetchers one
        stable API and prevents individual training loops from forgetting a
        mask or metadata tensor.
        """

        return LanguageBatch(
            context_ids=self.context_ids.to(
                device=device, non_blocking=non_blocking
            ),
            context_mask=self.context_mask.to(
                device=device, non_blocking=non_blocking
            ),
            target_ids=self.target_ids.to(
                device=device, non_blocking=non_blocking
            ),
            target_mask=self.target_mask.to(
                device=device, non_blocking=non_blocking
            ),
            target_complete=self.target_complete.to(
                device=device, non_blocking=non_blocking
            ),
        )

    def pin_memory(self) -> "LanguageBatch":
        """Return a page-locked CPU batch when CUDA transfers can use it."""

        if not torch.cuda.is_available():
            return self
        return LanguageBatch(
            context_ids=self.context_ids.pin_memory(),
            context_mask=self.context_mask.pin_memory(),
            target_ids=self.target_ids.pin_memory(),
            target_mask=self.target_mask.pin_memory(),
            target_complete=self.target_complete.pin_memory(),
        )

    def index_select(self, indices: torch.Tensor) -> "LanguageBatch":
        """Gather batch rows without materializing token rows in Python."""

        if indices.ndim != 1:
            raise ValueError("batch row indices must be one-dimensional")
        indices = indices.to(device=self.context_ids.device, dtype=torch.long)
        return LanguageBatch(
            context_ids=self.context_ids.index_select(0, indices),
            context_mask=self.context_mask.index_select(0, indices),
            target_ids=self.target_ids.index_select(0, indices),
            target_mask=self.target_mask.index_select(0, indices),
            target_complete=self.target_complete.index_select(0, indices),
        )


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _validate_row_indices(indices: torch.Tensor, rows: int) -> torch.Tensor:
    if indices.ndim != 1:
        raise ValueError("row indices must be one-dimensional")
    if indices.device.type != "cpu":
        raise ValueError("CPU batch row indices must reside on the CPU")
    indices = indices.to(dtype=torch.long)
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= rows
    ):
        raise IndexError("batch row index is out of range")
    return indices


def _fixed_window_corpus_fingerprint(
    tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    pad_token_id: int,
    eos_token_id: int,
    chunk_rows: int = 16_384,
) -> str:
    """Hash immutable sampling content without one giant byte copy."""

    digest = hashlib.sha256()
    digest.update(b"gator-fixed-window-corpus-v1\0")
    digest.update(str(pad_token_id).encode("ascii"))
    digest.update(b":")
    digest.update(str(eos_token_id).encode("ascii"))
    for name, tensor in tensors:
        tensor = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("ascii"))
        digest.update(b":")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b":")
        digest.update(
            ",".join(str(size) for size in tensor.shape).encode("ascii")
        )
        digest.update(b"\0")
        for start in range(0, tensor.shape[0], chunk_rows):
            chunk = tensor[start : start + chunk_rows].contiguous()
            try:
                byte_view = memoryview(chunk.numpy()).cast("B")
                digest.update(byte_view)
            except RuntimeError as error:
                if "Numpy is not available" not in str(error):
                    raise
                # NumPy is normally available with PyTorch. This fallback
                # preserves the exact contract in minimal installations at
                # the cost of slower one-time construction.
                for value in chunk.reshape(-1).tolist():
                    digest.update(
                        int(value).to_bytes(
                            tensor.element_size(),
                            "little",
                            signed=True,
                        )
                    )
    return digest.hexdigest()


def _interleave_language_batches(
    pieces: Sequence[tuple[torch.Tensor, LanguageBatch]],
    batch_size: int,
) -> LanguageBatch:
    """Restore independently gathered source batches to requested row order."""

    nonempty = [
        (positions.to(dtype=torch.long, device="cpu"), batch)
        for positions, batch in pieces
        if positions.numel()
    ]
    if not nonempty:
        raise ValueError("cannot interleave an empty set of language batches")
    positions = torch.cat([position for position, _ in nonempty])
    if positions.numel() != batch_size:
        raise ValueError("interleaved pieces must cover every output row")
    order = positions.argsort()

    def restored(name: str) -> torch.Tensor:
        values = torch.cat([getattr(batch, name) for _, batch in nonempty], dim=0)
        return values.index_select(0, order.to(values.device))

    return LanguageBatch(
        context_ids=restored("context_ids"),
        context_mask=restored("context_mask"),
        target_ids=restored("target_ids"),
        target_mask=restored("target_mask"),
        target_complete=restored("target_complete"),
    )


def encode_assistant_examples(
    examples: Iterable[AssistantExample], tokenizer: CompactTokenizer
) -> list[EncodedAssistantExample]:
    result = []
    for example in examples:
        context_ids = tokenizer.encode_turns(example.context)
        response_ids = tokenizer.encode(example.response) + [tokenizer.EOS_ID]
        result.append(
            EncodedAssistantExample(
                conversation_id=example.conversation_id,
                turn_index=example.turn_index,
                context_ids=tuple(context_ids),
                response_ids=tuple(response_ids),
                source=example.source,
            )
        )
    return result


class LanguageBatcher:
    """Pre-tokenized deterministic sampling and collation.

    The collated corpus is immutable provider state. Exact resume fingerprints
    IDs, masks, and completion flags, including active PAD-like token cases.
    """

    STATE_FORMAT = "gator-language-batcher-v2"

    @property
    def exact_state_supported(self) -> bool:
        return True

    @property
    def exact_resume_supported(self) -> bool:
        return self.exact_state_supported

    def __init__(
        self,
        examples: Sequence[EncodedAssistantExample],
        config: LanguageBatchConfig,
        *,
        pad_token_id: int = CompactTokenizer.PAD_ID,
        eos_token_id: int = CompactTokenizer.EOS_ID,
        seed: int = 0,
    ) -> None:
        if not examples:
            raise ValueError("language batcher requires at least one example")
        self.examples = tuple(examples)
        self.config = config
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.random = random.Random(seed)
        # The finite corpus is collated once. Training then samples it with one
        # vectorized row gather instead of rebuilding token Python lists on
        # every optimizer step.
        self._packed_batch = self.collate_cpu(self.examples)
        self.corpus_fingerprint = _fixed_window_corpus_fingerprint(
            (
                ("context_ids", self._packed_batch.context_ids),
                ("context_mask", self._packed_batch.context_mask),
                ("target_ids", self._packed_batch.target_ids),
                ("target_mask", self._packed_batch.target_mask),
                (
                    "target_complete",
                    self._packed_batch.target_complete,
                ),
            ),
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

    def random_indices(self, batch_size: int) -> torch.Tensor:
        _validate_batch_size(batch_size)
        return torch.tensor(
            [
                self.random.randrange(len(self.examples))
                for _ in range(batch_size)
            ],
            dtype=torch.long,
        )

    def batch_rows_cpu(self, indices: torch.Tensor) -> LanguageBatch:
        indices = _validate_row_indices(indices, len(self.examples))
        return self._packed_batch.index_select(indices)

    def sample_cpu(self, batch_size: int) -> LanguageBatch:
        """Sample a complete CPU batch with one packed tensor-row gather."""

        return self.batch_rows_cpu(self.random_indices(batch_size))

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> LanguageBatch:
        """Compatibility wrapper around CPU sampling plus batch transfer."""

        return self.sample_cpu(batch_size).to(
            device, non_blocking=non_blocking
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "format": self.STATE_FORMAT,
            "examples": len(self.examples),
            "context_tokens": self.config.context_tokens,
            "response_tokens": self.config.response_tokens,
            "reference_tokens": self.config.reference_tokens,
            "corpus_fingerprint": self.corpus_fingerprint,
            "random_state": self.random.getstate(),
            "exact": True,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "format": self.STATE_FORMAT,
            "examples": len(self.examples),
            "context_tokens": self.config.context_tokens,
            "response_tokens": self.config.response_tokens,
            "reference_tokens": self.config.reference_tokens,
            "corpus_fingerprint": self.corpus_fingerprint,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"incompatible language batcher state for {key!r}"
                )
        random_state = state.get("random_state")
        if not isinstance(random_state, tuple):
            raise ValueError("language batcher state is missing random_state")
        self.random.setstate(random_state)

    def collate(
        self, examples: Sequence[EncodedAssistantExample], device: torch.device | str
    ) -> LanguageBatch:
        return self.collate_cpu(examples).to(device)

    def collate_cpu(
        self, examples: Sequence[EncodedAssistantExample]
    ) -> LanguageBatch:
        if not examples:
            raise ValueError("cannot collate an empty language batch")
        context_rows, context_masks = [], []
        target_rows, target_masks, target_completes = [], [], []
        for example in examples:
            context = list(example.context_ids[-self.config.context_tokens :])
            context_padding = self.config.context_tokens - len(context)
            context_rows.append([self.pad_token_id] * context_padding + context)
            context_masks.append([False] * context_padding + [True] * len(context))

            target = list(example.response_ids[: self.config.target_tokens])
            target_padding = self.config.response_tokens - len(target)
            target_rows.append(target + [self.pad_token_id] * target_padding)
            target_masks.append([True] * len(target) + [False] * target_padding)
            target_completes.append(self.eos_token_id in target)

        return LanguageBatch(
            context_ids=torch.tensor(context_rows, dtype=torch.long),
            context_mask=torch.tensor(context_masks, dtype=torch.bool),
            target_ids=torch.tensor(target_rows, dtype=torch.long),
            target_mask=torch.tensor(target_masks, dtype=torch.bool),
            target_complete=torch.tensor(
                target_completes, dtype=torch.bool
            ),
        )


class StreamingLanguageSource:
    """Shuffle-buffered encoded examples from a restartable text stream."""

    STATE_FORMAT = "gator-streaming-language-source-v1"

    def __init__(
        self,
        example_factory: Callable[[], Iterable[AssistantExample]],
        tokenizer: CompactTokenizer,
        *,
        shuffle_buffer: int = 2048,
        seed: int = 0,
        repeat: bool = True,
    ) -> None:
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")
        self.example_factory = example_factory
        self.tokenizer = tokenizer
        self.shuffle_buffer = shuffle_buffer
        self.repeat = repeat
        self.random = random.Random(seed)
        self._iterator: Iterator[AssistantExample] = iter(example_factory())
        self._buffer: list[EncodedAssistantExample] = []
        self._raw_exhausted = False
        self._raw_items_seen = 0
        self.restarts = 0
        self.items_emitted = 0

    @property
    def exact_state_supported(self) -> bool:
        # The encoded shuffle buffer is checkpointed and the deterministic
        # source factory is replayed to the exact post-buffer raw position.
        return True

    @property
    def exact_resume_supported(self) -> bool:
        return self.exact_state_supported

    def _encode(self, example: AssistantExample) -> EncodedAssistantExample:
        return EncodedAssistantExample(
            conversation_id=example.conversation_id,
            turn_index=example.turn_index,
            context_ids=tuple(self.tokenizer.encode_turns(example.context)),
            response_ids=tuple(
                self.tokenizer.encode(example.response) + [self.tokenizer.EOS_ID]
            ),
            source=example.source,
        )

    def _next_raw(self) -> AssistantExample:
        try:
            example = next(self._iterator)
            self._raw_items_seen += 1
            return example
        except StopIteration:
            if self._raw_items_seen == 0:
                raise ValueError(
                    "streaming language source produced no examples"
                )
            raise StreamingLanguageSourceExhausted

    def _start_next_epoch(self) -> None:
        self.restarts += 1
        self._iterator = iter(self.example_factory())
        self._raw_exhausted = False
        self._raw_items_seen = 0

    def next(self) -> EncodedAssistantExample:
        if not self._buffer and self._raw_exhausted:
            if not self.repeat:
                raise StreamingLanguageSourceExhausted
            self._start_next_epoch()
        while (
            len(self._buffer) < self.shuffle_buffer
            and not self._raw_exhausted
        ):
            try:
                self._buffer.append(self._encode(self._next_raw()))
            except StreamingLanguageSourceExhausted:
                self._raw_exhausted = True
        if not self._buffer:
            raise StreamingLanguageSourceExhausted
        index = self.random.randrange(len(self._buffer))
        selected = self._buffer[index]
        if self._raw_exhausted:
            self._buffer.pop(index)
        else:
            try:
                self._buffer[index] = self._encode(self._next_raw())
            except StreamingLanguageSourceExhausted:
                self._raw_exhausted = True
                self._buffer.pop(index)
        self.items_emitted += 1
        return selected

    def state_dict(self) -> dict[str, object]:
        return {
            "format": self.STATE_FORMAT,
            "shuffle_buffer": self.shuffle_buffer,
            "repeat": self.repeat,
            "random_state": self.random.getstate(),
            "buffer": tuple(self._buffer),
            "raw_exhausted": self._raw_exhausted,
            "raw_items_seen": self._raw_items_seen,
            "restarts": self.restarts,
            "items_emitted": self.items_emitted,
            "exact": True,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("format") != self.STATE_FORMAT:
            raise ValueError("incompatible streaming language source state")
        if int(state.get("shuffle_buffer", -1)) != self.shuffle_buffer:
            raise ValueError("streaming source shuffle buffer differs")
        if bool(state.get("repeat", self.repeat)) != self.repeat:
            raise ValueError("streaming source repeat policy differs")
        random_state = state.get("random_state")
        buffer = state.get("buffer")
        if not isinstance(random_state, tuple):
            raise ValueError("streaming source state is missing RNG state")
        if not isinstance(buffer, (tuple, list)) or any(
            not isinstance(item, EncodedAssistantExample) for item in buffer
        ):
            raise ValueError("streaming source state has an invalid buffer")
        raw_items_seen = int(state.get("raw_items_seen", -1))
        if raw_items_seen < 0:
            raise ValueError("streaming source raw cursor is invalid")

        iterator = iter(self.example_factory())
        for _ in range(raw_items_seen):
            try:
                next(iterator)
            except StopIteration as error:
                raise ValueError(
                    "streaming source ended before the saved raw cursor"
                ) from error
        self._iterator = iterator
        self._buffer = list(buffer)
        self._raw_exhausted = bool(state.get("raw_exhausted", False))
        self._raw_items_seen = raw_items_seen
        self.restarts = int(state.get("restarts", 0))
        self.items_emitted = int(state.get("items_emitted", 0))
        self.random.setstate(random_state)


class StreamingLanguageSourceExhausted(RuntimeError):
    """Signal that a non-repeating streaming corpus has been drained."""


class PackedLanguageSource:
    """Memory-packed fixed-window examples for high-throughput random sampling.

    Corpus tensors are cloned on construction and must be treated as immutable
    thereafter. Masks/completion are deterministic functions of the stored IDs.
    """

    STATE_FORMAT = "gator-packed-language-source-v2"

    @property
    def exact_state_supported(self) -> bool:
        return True

    @property
    def exact_resume_supported(self) -> bool:
        return self.exact_state_supported

    def __init__(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        pad_token_id: int = CompactTokenizer.PAD_ID,
        eos_token_id: int = CompactTokenizer.EOS_ID,
        seed: int = 0,
    ) -> None:
        if context_ids.ndim != 2 or target_ids.ndim != 2:
            raise ValueError("packed context and target tensors must be two-dimensional")
        if context_ids.shape[0] != target_ids.shape[0] or context_ids.shape[0] == 0:
            raise ValueError("packed context and target tensors must have equal nonzero rows")
        # Clone caller storage so later mutation of the input tensors cannot
        # silently change a corpus after its exact-resume identity is fixed.
        self.context_ids = (
            context_ids.to(dtype=torch.int32, device="cpu")
            .contiguous()
            .clone()
        )
        self.target_ids = (
            target_ids.to(dtype=torch.int32, device="cpu")
            .contiguous()
            .clone()
        )
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.random = random.Random(seed)
        self.corpus_fingerprint = _fixed_window_corpus_fingerprint(
            (
                ("context_ids", self.context_ids),
                ("target_ids", self.target_ids),
            ),
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

    @property
    def examples(self) -> int:
        return self.context_ids.shape[0]

    @property
    def storage_bytes(self) -> int:
        return (
            self.context_ids.numel() * self.context_ids.element_size()
            + self.target_ids.numel() * self.target_ids.element_size()
        )

    @classmethod
    def build(
        cls,
        examples: Iterable[AssistantExample],
        tokenizer: CompactTokenizer,
        config: LanguageBatchConfig,
        *,
        chunk_size: int = 4096,
        seed: int = 0,
        progress_every: int = 100_000,
    ) -> "PackedLanguageSource":
        return cls.build_encoded(
            (
                encoded
                for example in examples
                for encoded in encode_assistant_examples([example], tokenizer)
            ),
            config,
            chunk_size=chunk_size,
            seed=seed,
            progress_every=progress_every,
        )

    @classmethod
    def build_encoded(
        cls,
        examples: Iterable[EncodedAssistantExample],
        config: LanguageBatchConfig,
        *,
        chunk_size: int = 4096,
        seed: int = 0,
        progress_every: int = 100_000,
    ) -> "PackedLanguageSource":
        """Pack an already-tokenized ordered stream without re-encoding it."""

        context_chunks, target_chunks = [], []
        pending: list[EncodedAssistantExample] = []
        collator = None
        count = 0

        def flush() -> None:
            nonlocal pending, collator
            if not pending:
                return
            if collator is None:
                collator = LanguageBatcher(pending, config, seed=seed)
                batch = collator._packed_batch
            else:
                batch = collator.collate_cpu(pending)
            context_chunks.append(batch.context_ids.to(torch.int32))
            target_chunks.append(batch.target_ids.to(torch.int32))
            pending = []

        for example in examples:
            pending.append(example)
            count += 1
            if len(pending) >= chunk_size:
                flush()
            if progress_every and count % progress_every == 0:
                print(f"packed_lmsys_examples={count}", flush=True)
        flush()
        if not context_chunks:
            raise ValueError("cannot pack an empty language source")
        return cls(
            torch.cat(context_chunks, 0),
            torch.cat(target_chunks, 0),
            pad_token_id=collator.pad_token_id,
            eos_token_id=collator.eos_token_id,
            seed=seed,
        )

    def random_indices(self, batch_size: int) -> torch.Tensor:
        _validate_batch_size(batch_size)
        return torch.tensor(
            [self.random.randrange(self.examples) for _ in range(batch_size)],
            dtype=torch.long,
        )

    def batch_rows_cpu(self, indices: torch.Tensor) -> LanguageBatch:
        """Gather fixed-window rows directly, without Python token objects."""

        indices = _validate_row_indices(indices, self.examples)
        context_ids = self.context_ids.index_select(0, indices).to(torch.long)
        target_ids = self.target_ids.index_select(0, indices).to(torch.long)
        return LanguageBatch(
            context_ids=context_ids,
            context_mask=context_ids.ne(self.pad_token_id),
            target_ids=target_ids,
            target_mask=target_ids.ne(self.pad_token_id),
            target_complete=target_ids.eq(self.eos_token_id).any(dim=1),
        )

    def sample_cpu(self, batch_size: int) -> LanguageBatch:
        return self.batch_rows_cpu(self.random_indices(batch_size))

    def state_dict(self) -> dict[str, object]:
        return {
            "format": self.STATE_FORMAT,
            "examples": self.examples,
            "context_tokens": self.context_ids.shape[1],
            "response_tokens": self.target_ids.shape[1],
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "corpus_fingerprint": self.corpus_fingerprint,
            "random_state": self.random.getstate(),
            "exact": True,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "format": self.STATE_FORMAT,
            "examples": self.examples,
            "context_tokens": self.context_ids.shape[1],
            "response_tokens": self.target_ids.shape[1],
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "corpus_fingerprint": self.corpus_fingerprint,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"incompatible packed language source state for {key!r}"
                )
        random_state = state.get("random_state")
        if not isinstance(random_state, tuple):
            raise ValueError("packed source state is missing random_state")
        self.random.setstate(random_state)

    def next(self) -> EncodedAssistantExample:
        # Compatibility oracle for APIs that still consume one encoded object.
        # High-throughput batching uses ``sample_cpu`` and never enters this
        # token-by-token Python conversion path.
        index = int(self.random_indices(1)[0])
        context = self.context_ids[index]
        target = self.target_ids[index]
        context_ids = tuple(int(token) for token in context if token != self.pad_token_id)
        target_ids = tuple(int(token) for token in target if token != self.pad_token_id)
        return EncodedAssistantExample(
            conversation_id=-1,
            turn_index=-1,
            context_ids=context_ids,
            response_ids=target_ids,
            source=None,
        )


class WeightedLanguageSource:
    """Randomly interleave named restartable sources with observable weights."""

    STATE_FORMAT = "gator-weighted-language-source-v1"

    @property
    def exact_state_supported(self) -> bool:
        return all(
            bool(getattr(source, "exact_state_supported", False))
            for _, source in self.sources
        )

    @property
    def exact_resume_supported(self) -> bool:
        return self.exact_state_supported

    def __init__(
        self,
        sources: Sequence[
            tuple[str, StreamingLanguageSource | PackedLanguageSource]
        ],
        weights: Sequence[float],
        *,
        seed: int = 0,
    ) -> None:
        if not sources:
            raise ValueError("weighted language source requires at least one source")
        if len(sources) != len(weights):
            raise ValueError("source names and weights must have equal length")
        names = [name for name, _ in sources]
        if any(not name for name in names):
            raise ValueError("source names cannot be empty")
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("source weights must be finite and positive")

        self.sources = tuple(sources)
        self.weights = tuple(float(weight) for weight in weights)
        self.random = random.Random(seed)
        self.sample_counts = {name: 0 for name in names}
        self.items_emitted = 0
        self._active_indices = list(range(len(self.sources)))
        self.exhausted_sources: set[str] = set()

    @property
    def restarts(self) -> dict[str, int | None]:
        return {
            name: getattr(source, "restarts", None)
            for name, source in self.sources
        }

    def next(self) -> EncodedAssistantExample:
        while self._active_indices:
            selected_index = self.random.choices(
                self._active_indices,
                weights=[
                    self.weights[index]
                    for index in self._active_indices
                ],
                k=1,
            )[0]
            name, source = self.sources[selected_index]
            try:
                example = source.next()
            except StreamingLanguageSourceExhausted:
                self._active_indices.remove(selected_index)
                self.exhausted_sources.add(name)
                continue
            self.sample_counts[name] += 1
            self.items_emitted += 1
            return example
        raise ValueError("all streaming language sources are exhausted")

    def sample_cpu(
        self,
        batch_size: int,
        collator: LanguageBatcher,
    ) -> LanguageBatch:
        """Sample named sources while gathering packed corpora tensor-wise."""

        _validate_batch_size(batch_size)
        if any(
            isinstance(source, StreamingLanguageSource)
            and not source.repeat
            for _, source in self.sources
        ):
            return collator.collate_cpu(
                [self.next() for _ in range(batch_size)]
            )
        selections = torch.tensor(
            self.random.choices(
                range(len(self.sources)),
                weights=self.weights,
                k=batch_size,
            ),
            dtype=torch.long,
        )
        pieces: list[tuple[torch.Tensor, LanguageBatch]] = []
        for source_index, (name, source) in enumerate(self.sources):
            positions = selections.eq(source_index).nonzero(
                as_tuple=False
            ).flatten()
            count = positions.numel()
            if not count:
                continue
            self.sample_counts[name] += count
            if isinstance(source, PackedLanguageSource):
                batch = source.sample_cpu(count)
                if (
                    batch.context_ids.shape[1] != collator.config.context_tokens
                    or batch.target_ids.shape[1]
                    != collator.config.response_tokens
                ):
                    raise ValueError(
                        "packed source windows do not match mixture batch config"
                    )
            else:
                batch = collator.collate_cpu(
                    [source.next() for _ in range(count)]
                )
            pieces.append((positions, batch))
        self.items_emitted += batch_size
        return _interleave_language_batches(pieces, batch_size)

    def state_dict(self) -> dict[str, object]:
        source_states = []
        incomplete_sources = []
        for name, source in self.sources:
            state_method = getattr(source, "state_dict", None)
            if callable(state_method):
                nested_state = state_method()
                source_states.append(nested_state)
                if not bool(
                    getattr(source, "exact_state_supported", False)
                ):
                    incomplete_sources.append(name)
            else:
                source_states.append(None)
                incomplete_sources.append(name)
        return {
            "format": self.STATE_FORMAT,
            "names": tuple(name for name, _ in self.sources),
            "weights": self.weights,
            "random_state": self.random.getstate(),
            "sample_counts": dict(self.sample_counts),
            "items_emitted": self.items_emitted,
            "active_indices": tuple(self._active_indices),
            "exhausted_sources": tuple(sorted(self.exhausted_sources)),
            "source_states": tuple(source_states),
            "exact": not incomplete_sources,
            "incomplete_sources": tuple(incomplete_sources),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("format") != self.STATE_FORMAT:
            raise ValueError("incompatible weighted source state format")
        names = tuple(name for name, _ in self.sources)
        if state.get("names") != names or state.get("weights") != self.weights:
            raise ValueError("weighted source topology does not match saved state")
        random_state = state.get("random_state")
        if not isinstance(random_state, tuple):
            raise ValueError("weighted source state is missing random_state")
        counts = state.get("sample_counts")
        if not isinstance(counts, Mapping) or set(counts) != set(names):
            raise ValueError("weighted source sample counts are incompatible")
        nested = state.get("source_states")
        if not isinstance(nested, (tuple, list)) or len(nested) != len(
            self.sources
        ):
            raise ValueError("weighted source nested states are incompatible")
        for (_, source), source_state in zip(
            self.sources, nested, strict=True
        ):
            if source_state is None:
                continue
            load_method = getattr(source, "load_state_dict", None)
            if not callable(load_method):
                raise ValueError("source cannot restore its saved state")
            load_method(source_state)
        self.random.setstate(random_state)
        self.sample_counts = {
            name: int(counts[name]) for name in names
        }
        self.items_emitted = int(state.get("items_emitted", 0))
        active_indices = state.get(
            "active_indices",
            tuple(range(len(self.sources))),
        )
        if (
            not isinstance(active_indices, (tuple, list))
            or any(
                not isinstance(index, int)
                or not 0 <= index < len(self.sources)
                for index in active_indices
            )
            or len(set(active_indices)) != len(active_indices)
        ):
            raise ValueError(
                "weighted source active indices are incompatible"
            )
        self._active_indices = list(active_indices)
        exhausted_sources = state.get("exhausted_sources", ())
        if (
            not isinstance(exhausted_sources, (tuple, list))
            or any(name not in names for name in exhausted_sources)
        ):
            raise ValueError(
                "weighted source exhausted names are incompatible"
            )
        self.exhausted_sources = set(exhausted_sources)


class MixtureLanguageBatcher:
    """Draw batches from a finite corpus and a streaming corpus by weight."""

    STATE_FORMAT = "gator-mixture-language-batcher-v1"

    @property
    def exact_state_supported(self) -> bool:
        return (
            self.finite_batcher.exact_state_supported
            and (
                self.streaming_weight == 0.0
                or bool(
                    getattr(
                        self.streaming_source,
                        "exact_state_supported",
                        False,
                    )
                )
            )
        )

    @property
    def exact_resume_supported(self) -> bool:
        return self.exact_state_supported

    def __init__(
        self,
        finite_examples: Sequence[EncodedAssistantExample],
        streaming_source: (
            StreamingLanguageSource
            | PackedLanguageSource
            | WeightedLanguageSource
        ),
        config: LanguageBatchConfig,
        *,
        streaming_weight: float = 0.9,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= streaming_weight <= 1.0:
            raise ValueError("streaming_weight must be between 0 and 1")
        self.finite_batcher = LanguageBatcher(finite_examples, config, seed=seed)
        self.streaming_source = streaming_source
        self.streaming_weight = streaming_weight
        self.config = config
        self.random = random.Random(seed)
        self.streaming_samples = 0
        self.finite_samples = 0

    def sample_cpu(self, batch_size: int) -> LanguageBatch:
        _validate_batch_size(batch_size)
        streaming_flags: list[bool] = []
        finite_indices: list[int] = []
        for _ in range(batch_size):
            use_streaming = self.random.random() < self.streaming_weight
            streaming_flags.append(use_streaming)
            if not use_streaming:
                finite_indices.append(
                    self.random.randrange(len(self.finite_batcher.examples))
                )
        streaming = torch.tensor(streaming_flags, dtype=torch.bool)
        streaming_positions = streaming.nonzero(as_tuple=False).flatten()
        finite_positions = (~streaming).nonzero(as_tuple=False).flatten()
        pieces: list[tuple[torch.Tensor, LanguageBatch]] = []

        streaming_count = streaming_positions.numel()
        if streaming_count:
            if isinstance(self.streaming_source, PackedLanguageSource):
                streaming_batch = self.streaming_source.sample_cpu(
                    streaming_count
                )
                if (
                    streaming_batch.context_ids.shape[1]
                    != self.config.context_tokens
                    or streaming_batch.target_ids.shape[1]
                    != self.config.response_tokens
                ):
                    raise ValueError(
                        "packed source windows do not match mixture batch config"
                    )
            elif isinstance(self.streaming_source, WeightedLanguageSource):
                streaming_batch = self.streaming_source.sample_cpu(
                    streaming_count, self.finite_batcher
                )
            else:
                streaming_batch = self.finite_batcher.collate_cpu(
                    [
                        self.streaming_source.next()
                        for _ in range(streaming_count)
                    ]
                )
            pieces.append((streaming_positions, streaming_batch))

        finite_count = finite_positions.numel()
        if finite_count:
            pieces.append(
                (
                    finite_positions,
                    self.finite_batcher.batch_rows_cpu(
                        torch.tensor(finite_indices, dtype=torch.long)
                    ),
                )
            )
        self.streaming_samples += streaming_count
        self.finite_samples += finite_count
        return _interleave_language_batches(pieces, batch_size)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> LanguageBatch:
        return self.sample_cpu(batch_size).to(
            device, non_blocking=non_blocking
        )

    def state_dict(self) -> dict[str, object]:
        source_state_method = getattr(
            self.streaming_source, "state_dict", None
        )
        source_state = (
            source_state_method()
            if callable(source_state_method)
            else None
        )
        source_exact = self.streaming_weight == 0.0 or (
            source_state is not None
            and bool(
                getattr(
                    self.streaming_source,
                    "exact_state_supported",
                    False,
                )
            )
        )
        return {
            "format": self.STATE_FORMAT,
            "streaming_weight": self.streaming_weight,
            "random_state": self.random.getstate(),
            "finite_batcher": self.finite_batcher.state_dict(),
            "streaming_source": source_state,
            "streaming_samples": self.streaming_samples,
            "finite_samples": self.finite_samples,
            "exact": source_exact,
            "incomplete_components": (
                () if source_exact else ("streaming_source",)
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            state.get("format") != self.STATE_FORMAT
            or state.get("streaming_weight") != self.streaming_weight
        ):
            raise ValueError("mixture batcher state is incompatible")
        random_state = state.get("random_state")
        finite_state = state.get("finite_batcher")
        if not isinstance(random_state, tuple) or not isinstance(
            finite_state, Mapping
        ):
            raise ValueError("mixture batcher state is incomplete")
        source_state = state.get("streaming_source")
        if source_state is not None:
            load_method = getattr(
                self.streaming_source, "load_state_dict", None
            )
            if not callable(load_method):
                raise ValueError(
                    "mixture streaming source cannot restore saved state"
                )
            load_method(source_state)
        self.finite_batcher.load_state_dict(finite_state)
        self.random.setstate(random_state)
        self.streaming_samples = int(state.get("streaming_samples", 0))
        self.finite_samples = int(state.get("finite_samples", 0))


def masked_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    precomputed_log_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("logits and target IDs must have [batch, slots] leading shapes")
    if target_mask.shape != target_ids.shape:
        raise ValueError("target_mask must match target_ids")
    active = target_mask.bool()
    if precomputed_log_probs is None:
        log_probs = F.log_softmax(logits, dim=-1)
    else:
        if precomputed_log_probs.shape != logits.shape:
            raise ValueError("precomputed_log_probs must match logits")
        log_probs = precomputed_log_probs
    token_nll = -log_probs.gather(
        2, target_ids[..., None]
    ).squeeze(2)
    # ``where`` blocks both value and gradient from impossible inactive PAD
    # targets without synchronizing a GPU predicate back to Python. The clamped
    # denominator makes an all-inactive batch a connected finite zero.
    masked_nll = torch.where(active, token_nll, torch.zeros_like(token_nll))
    return masked_nll.sum() / active.sum().clamp_min(1).to(masked_nll.dtype)


def legal_response_token_mask(
    tokenizer: CompactTokenizer, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return output IDs that may legally appear in assistant response slots.

    UTF-8 bytes, learned text pieces, EOS, and the optional V2 BLANK are legal.
    Input/control-only PAD, BOS, MASK, USER, and ASSISTANT markers are censored.
    """

    legal = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    legal[tokenizer.BYTE_OFFSET : tokenizer.LEARNED_OFFSET] = True
    legal[tokenizer.LEARNED_OFFSET : tokenizer.learned_end] = True
    legal[tokenizer.EOS_ID] = True
    if tokenizer.blank_token_id is not None:
        legal[tokenizer.blank_token_id] = True
    return legal


def rigid_blank_tail_targets(
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    target_complete: torch.Tensor,
    *,
    blank_token_id: int,
    score_censored_tail: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label the unused tail of complete targets with a real BLANK token.

    This is the deliberately rigid baseline for comparing against latent
    alignment. Existing active labels remain unchanged. Truncated/incomplete
    examples retain their original mask because their true ending is unknown,
    unless ``score_censored_tail`` declares that the extra physical slots were
    deliberately reserved as alignment slack beyond the reference cap.
    """

    if target_ids.ndim != 2 or target_mask.shape != target_ids.shape:
        raise ValueError(
            "target_ids and target_mask must have matching [batch, slots] shapes"
        )
    if target_complete.shape != (target_ids.shape[0],):
        raise ValueError("target_complete must have [batch] shape")
    if blank_token_id < 0:
        raise ValueError("blank_token_id must be nonnegative")
    target_mask = target_mask.bool()
    target_complete = target_complete.bool()
    if target_mask.device.type == "cpu":
        active_after_gap = target_mask & (~target_mask).cumsum(dim=1).gt(0)
        if bool(active_after_gap.any()):
            raise ValueError("target_mask must be a contiguous active prefix")

    blank_tail = ~target_mask & (
        target_complete[:, None]
        | torch.full_like(target_mask, score_censored_tail)
    )
    labels = target_ids.masked_fill(blank_tail, blank_token_id)
    loss_mask = target_mask | blank_tail
    return labels, loss_mask


def whitespace_token_mask(
    tokenizer: CompactTokenizer, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return pure-whitespace text IDs, excluding every structural token."""

    structural = {
        tokenizer.PAD_ID,
        tokenizer.BOS_ID,
        tokenizer.EOS_ID,
        tokenizer.MASK_ID,
        tokenizer.USER_ID,
        tokenizer.ASSISTANT_ID,
    }
    if tokenizer.blank_token_id is not None:
        structural.add(tokenizer.blank_token_id)
    values = [
        token_id not in structural
        and bool(tokenizer.decode([token_id], show_roles=False).isspace())
        for token_id in range(tokenizer.vocab_size)
    ]
    return torch.tensor(values, dtype=torch.bool, device=device)


def whitespace_formation_penalties(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    whitespace_mask: torch.Tensor,
    content_target_mask: torch.Tensor | None = None,
    probabilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return content-omission and adjacent-whitespace probability penalties.

    Only scored target positions participate. Padding after the reference EOS
    is excluded, and legitimate isolated whitespace remains unpenalized.
    ``content_target_mask`` lets callers exclude EOS, BLANK, and structural
    symbols from the omission term instead of treating every non-whitespace
    symbol as lexical content.
    """

    if whitespace_mask.ndim != 1 or whitespace_mask.shape[0] != logits.shape[-1]:
        raise ValueError("whitespace_mask must have one entry per vocabulary item")
    if probabilities is None:
        probabilities = logits.softmax(dim=-1)
    elif probabilities.shape != logits.shape:
        raise ValueError("probabilities must match logits")
    whitespace_probability = probabilities[..., whitespace_mask].sum(dim=-1)
    target_whitespace = whitespace_mask[target_ids]

    if content_target_mask is None:
        non_whitespace_targets = target_mask & ~target_whitespace
    else:
        if content_target_mask.shape != target_mask.shape:
            raise ValueError("content_target_mask must match target_mask")
        non_whitespace_targets = target_mask & content_target_mask.bool()
    omission = (
        whitespace_probability * non_whitespace_targets
    ).sum() / non_whitespace_targets.sum().clamp_min(1)

    pair_mask = target_mask[:, :-1] & target_mask[:, 1:]
    legitimate_double_whitespace = (
        target_whitespace[:, :-1] & target_whitespace[:, 1:]
    )
    pair_mask = pair_mask & ~legitimate_double_whitespace
    adjacent = (
        whitespace_probability[:, :-1]
        * whitespace_probability[:, 1:]
        * pair_mask
    ).sum() / pair_mask.sum().clamp_min(1)
    return omission, adjacent


def stewardship_route_losses(
    ownership: torch.Tensor,
    route_uncertainty: torch.Tensor,
    route_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return familiar-route entropy and batch-marginal load balance losses.

    ``ownership`` is ``[B, R+1, C, T]`` and ``route_uncertainty`` is
    ``[B, R, T]``. Entropy pressure is strongest for familiar/confident work
    and vanishes as handoff uncertainty approaches one, allowing novel work to
    reopen a broad route beam. Balance acts only on the population marginal;
    individual tokens are never forced to visit every cell. Invalid padded
    context positions are excluded when ``route_valid`` is supplied.
    """

    if ownership.ndim != 4 or route_uncertainty.ndim != 3:
        raise ValueError("invalid stewardship diagnostic ranks")
    routes = ownership[:, 1:]
    if (
        routes.shape[:2] != route_uncertainty.shape[:2]
        or routes.shape[-1] != route_uncertainty.shape[-1]
    ):
        raise ValueError("ownership and uncertainty relay/token axes must match")
    cells = routes.shape[2]
    if route_valid is None:
        route_valid = torch.ones(
            routes.shape[0],
            routes.shape[-1],
            dtype=torch.bool,
            device=routes.device,
        )
    if route_valid.shape != (routes.shape[0], routes.shape[-1]):
        raise ValueError("route_valid must have [batch, tokens] shape")
    valid = route_valid[:, None].to(routes.dtype)
    entropy = -(
        routes * routes.clamp_min(1e-9).log()
    ).sum(2) / math.log(cells)
    familiarity = (
        (1.0 - route_uncertainty.detach()).clamp(0.0, 1.0) * valid
    )
    familiar_entropy = (entropy * familiarity).sum() / familiarity.sum().clamp_min(1.0)

    marginal_load = (routes * valid[:, :, None]).sum(dim=(0, 1, 3))
    marginal_load = marginal_load / (
        valid.sum() * routes.shape[1]
    ).clamp_min(1.0)
    balance = ((marginal_load * cells - 1.0) ** 2).mean()
    return familiar_entropy, balance


def stewardship_route_losses_from_statistics(
    route_entropy: torch.Tensor,
    route_cell_load: torch.Tensor,
    route_uncertainty: torch.Tensor,
    route_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute route losses from packed-native sufficient statistics.

    ``route_entropy`` and ``route_uncertainty`` use ``[B,R,T]`` layout.
    ``route_cell_load`` is ``[R,C]`` and contains each relay's mean ownership
    over valid batch/token positions. This representation avoids retaining a
    dense ``[B,R+1,C,T]`` ownership history during ordinary optimizer steps.
    """

    if (
        route_entropy.ndim != 3
        or route_uncertainty.shape != route_entropy.shape
        or route_cell_load.ndim != 2
        or route_cell_load.shape[0] != route_entropy.shape[1]
    ):
        raise ValueError("invalid packed route-loss statistic shapes")
    if route_valid.shape != (
        route_entropy.shape[0],
        route_entropy.shape[2],
    ):
        raise ValueError("route_valid must have [batch,tokens] shape")
    if route_cell_load.shape[1] <= 0:
        raise ValueError("route_cell_load must contain at least one cell")

    valid = route_valid[:, None].to(route_entropy.dtype)
    familiarity = (
        (1.0 - route_uncertainty.detach()).clamp(0.0, 1.0)
        * valid
    )
    familiar_entropy = (
        route_entropy * familiarity
    ).sum() / familiarity.sum().clamp_min(1.0)
    cells = route_cell_load.shape[1]
    marginal_load = route_cell_load.mean(0)
    balance = ((marginal_load * cells - 1.0) ** 2).mean()
    return familiar_entropy, balance
