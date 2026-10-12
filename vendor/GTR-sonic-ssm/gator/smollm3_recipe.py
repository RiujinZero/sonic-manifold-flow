"""Scaled SmolLM3 pretraining recipe and resumable HF streaming for GTR-ARW."""

from __future__ import annotations

import atexit
import gc
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .distractor_reasoning_source import (
    DISTRACTOR_REASONING_SOURCE_NAME,
    DistractorReasoningTextSource,
    load_chat_turn_pool,
)
from .tasks import LanguageBatch


def _process_rss_bytes() -> int:
    """Return current process RSS without requiring psutil."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (FileNotFoundError, IndexError, OSError, ValueError):
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return peak * (1 if sys.platform == "darwin" else 1024)
        except (ImportError, OSError, ValueError):
            return 0


TOTAL_TOKENS = 26_000_000_000
SEQUENCE_LENGTH = 4096
EFFECTIVE_BATCH_SEQUENCES = 32
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SEQUENCES
TOTAL_UPDATES = 198_365
WARMUP_UPDATES = 500
STAGE2_UPDATE = 144_992
STAGE3_UPDATE = 176_428
DECAY_UPDATES = 21_938
PEAK_LR = 2e-4


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo: str
    weight: float
    config: str | None = None
    split: str = "train"
    revision: str | None = None
    text_columns: tuple[str, ...] = ("text", "content")
    load_kwargs: tuple[tuple[str, Any], ...] = ()


def _source_identity(spec: SourceSpec) -> dict[str, Any]:
    """Reader identity excludes stage-local blend weight."""
    identity = asdict(spec)
    identity.pop("weight")
    return identity


# Public counterparts of the exact named streams in the official YAML pack.
# Language/code subsets remain separate mixture entries even when hosted in one repo.
_BASE = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", None),
    "dclm": ("mlfoundations/dclm-baseline-1.0", None),
    "pes2o": ("allenai/dolmino-mix-1124", "pes2o"),
    "wiki": ("allenai/dolmino-mix-1124", "wiki"),
    # The production shard is an aggregate StackExchange stream. The newer
    # HF 2025 mirror requires one site config per load, so use DOLMino's
    # public aggregate as the direct-HF counterpart.
    "stackexchange": ("allenai/dolmino-mix-1124", "stackexchange"),
    "infiwebmath": ("HuggingFaceTB/finemath", "infiwebmath-3plus"),
    "finemath": ("HuggingFaceTB/finemath", "finemath-3plus"),
    "infiwebmath-4plus": ("HuggingFaceTB/finemath", "infiwebmath-4plus"),
    "finemath-4plus": ("HuggingFaceTB/finemath", "finemath-4plus"),
}


def _spec(name: str, weight: float) -> SourceSpec:
    text_columns = ("text", "content")
    load_kwargs: tuple[tuple[str, Any], ...] = ()
    split = "train"
    if name == DISTRACTOR_REASONING_SOURCE_NAME:
        # No HF repo backs this source -- SmolLM3BatchProvider._build_source
        # special-cases the name before repo/config are ever used.
        return SourceSpec(name, "", weight)
    if name.startswith("fw2-"):
        language_configs = {"fra":"fra_Latn","spa":"spa_Latn","deu":"deu_Latn","ita":"ita_Latn","por":"por_Latn","cmn":"cmn_Hani","rus":"rus_Cyrl","fas":"fas_Arab","jpn":"jpn_Jpan","kor":"kor_Hang","hin":"hin_Deva","tha":"tha_Thai","vie":"vie_Latn","ell":"ell_Grek"}
        repo, config = "HuggingFaceFW/fineweb-2", language_configs[name.removeprefix("fw2-")]
    elif name.startswith(("stack-v2-", "stack-edu-")):
        # The SmolLM3 YAML points Nanotron at already-tokenized Stack v2
        # shards.  bigcode/the-stack-v2 itself contains SWH IDs, not source
        # text, so direct-HF mode uses the closest content-bearing BigCode
        # corpus. Exact data reproduction requires the original .ds shards.
        language = name.split("-", 2)[2]
        data_dirs = {
            "Python":"python", "Java":"java", "JavaScript":"javascript",
            "C":"c", "Cpp":"cpp", "C-Sharp":"c-sharp", "PHP":"php",
            "TypeScript":"typescript", "Swift":"swift", "SQL":"sql",
            "Ruby":"ruby", "Markdown":"markdown", "HTML":"html",
            "Rust":"rust", "Go":"go", "Shell":"shell",
        }
        if language == "Swift":
            # StarCoderData revision 9fc30b5 has no Swift directory. The
            # content-bearing deduplicated Stack corpus does expose Swift.
            repo, config = "bigcode/the-stack-dedup", None
            load_kwargs = (("data_dir", "data/swift"),)
        else:
            repo, config = "bigcode/starcoderdata", None
            load_kwargs = (("data_dir", data_dirs[language]),)
        text_columns = ("content", "code", "text")
    elif name in {"pull-requests", "kaggle", "jupyter-scripts", "github-issues"}:
        # The current public StarCoder2 extras revision no longer exposes
        # pull_requests or jupyter_scripts builders. Keep those tiny logical
        # mixture slots content-bearing with the closest available builders;
        # exact reproduction remains available through DataTrove .ds mode.
        extra_configs = {
            "pull-requests":"issues", "kaggle":"kaggle",
            "jupyter-scripts":"kaggle", "github-issues":"issues",
        }
        repo, config = "bigcode/starcoder2data-extras", extra_configs[name]
        text_columns = ("content", "text")
    elif name.startswith("megamath-"):
        # Public MegaMath rows are metadata pointers; the original run used
        # materialized shards. Keep this mixture slot content-bearing in raw
        # mode, while exact mode reads the original logical source folder.
        repo = "HuggingFaceTB/finemath"
        config = "infiwebmath-4plus" if name == "megamath-web-pro" else "finemath-4plus"
    elif name in _BASE:
        repo, config = _BASE[name]
    else:
        late = {
            "multilingual-wiki": ("allenai/dolmino-mix-1124", "wiki"),
            "cosmopedia2": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2"),
            "openmathinstruct-2": ("nvidia/OpenMathInstruct-2", None),
            "openmathreasoning": ("nvidia/OpenMathReasoning", "default"),
            "open-codereasoning": ("nvidia/OpenCodeReasoning", "default"),
            "natural-reasoning": ("facebook/natural_reasoning", None),
            # Both production names are views of TinyGSM-MIND. The public
            # release combines them into one content-bearing train split.
            "problem-solving": ("ostapeno/tinygsm-mind", None),
            "2students": ("ostapeno/tinygsm-mind", None),
            "dolmino-gsm": ("allenai/dolmino-mix-1124", "math"),
            "dolmino-basic": ("allenai/dolmino-mix-1124", "math"),
        }
        repo, config = late[name]
        if name == "openmathreasoning":
            split = "cot"
        elif name == "open-codereasoning":
            split = "split_0"
        if name in {
            "openmathinstruct-2", "openmathreasoning", "open-codereasoning",
            "natural-reasoning", "problem-solving", "2students",
        }:
            text_columns = ()
    return SourceSpec(
        name, repo, weight, config, split=split,
        text_columns=text_columns, load_kwargs=load_kwargs
    )


_LANGS = ("fra", "spa", "deu", "ita", "por", "cmn", "rus", "fas", "jpn", "kor", "hin", "tha", "vie", "ell")
_CODE = ("Python", "Java", "JavaScript", "C", "Cpp", "C-Sharp", "PHP", "TypeScript", "Swift", "SQL", "Ruby", "Markdown", "HTML", "Rust", "Go", "Shell")

_STAGE2_CODE = (
    "stack-edu-Python", "stack-edu-Java", "stack-edu-JavaScript",
    "stack-edu-C", "stack-edu-Cpp", "stack-edu-C-Sharp", "stack-edu-PHP",
    "stack-v2-TypeScript", "stack-v2-Swift", "stack-edu-SQL",
    "stack-v2-Ruby", "stack-edu-Markdown", "stack-v2-HTML",
    "stack-v2-Rust", "stack-v2-Go", "stack-v2-Shell",
)
_STAGE3_CODE = (
    "stack-edu-Python", "stack-edu-Java", "stack-edu-JavaScript",
    "stack-edu-C", "stack-edu-Cpp", "stack-edu-C-Sharp", "stack-edu-PHP",
    "stack-edu-TypeScript", "stack-edu-Swift", "stack-edu-SQL",
    "stack-edu-Ruby", "stack-edu-Markdown",
    # Public Stack-Edu has no HTML builder; use the public Stack v2 stream.
    "stack-v2-HTML",
    "stack-edu-Rust", "stack-edu-Go", "stack-edu-Shell",
)


def _stage(names: Sequence[str], weights: Sequence[float]) -> tuple[SourceSpec, ...]:
    if len(names) != len(weights):
        raise RuntimeError("recipe source/weight mismatch")
    return tuple(_spec(name, weight) for name, weight in zip(names, weights, strict=True))


_COMMON = ("fineweb-edu", "dclm", "pes2o", "wiki", "stackexchange")
_TAIL = ("pull-requests", "kaggle", "jupyter-scripts", "github-issues")
STAGE1_NAMES = _COMMON + tuple(f"fw2-{x}" for x in _LANGS) + ("infiwebmath", "finemath") + tuple(f"stack-v2-{x}" for x in _CODE) + _TAIL
STAGE1_WEIGHTS = (.333,.37,.02,.001,.004,.016,.02,.022,.0105,.01,.01,.01,.003,.00325,.00325,.00325,.00325,.00325,.00225,.01,.017,.025,.013,.013,.007,.018,.006,.006,.003,.001,.004,.0008,.005,.006,.0008,.0005,.0007,.006,.0005,.0055,.0032)
STAGE2_NAMES = _COMMON + tuple(f"fw2-{x}" for x in _LANGS) + ("infiwebmath","finemath","infiwebmath-4plus","finemath-4plus","megamath-web-pro","megamath-qa-qwen","megamath-text-code-block") + _STAGE2_CODE + _TAIL
STAGE2_WEIGHTS = (.3,.33,.016,.001,.002,.016,.02,.0232,.0105,.01,.01,.01,.002,.00325,.00325,.00325,.00325,.00005,.00225,.01,.01,.01,.02,.02,.0008,.02,.025,.01725,.01625,.007,.018,.006,.004,.003,.001,.006,.0002,.00611,.00614,.0008,.0005,.0001,.0114,.0005,.01,.004)
STAGE3_NAMES = ("fineweb-edu","dclm","pes2o","wiki","multilingual-wiki","cosmopedia2","stackexchange") + tuple(f"fw2-{x}" for x in _LANGS) + ("infiwebmath","finemath","infiwebmath-4plus","finemath-4plus","megamath-web-pro","megamath-qa-qwen","megamath-text-code-block") + _STAGE3_CODE + _TAIL + ("openmathinstruct-2","openmathreasoning","open-codereasoning","natural-reasoning","problem-solving","2students","dolmino-gsm","dolmino-basic")
STAGE3_WEIGHTS = (.2,.3,.002,.0002,.008,.004,.001,.018,.022,.023,.0125,.0045,.01,.01,.009,.0032,.0032,.0032,.0032,.00005,.0022,.002,.002,.02,.025,.014,.002,.05,.07,.018,.018,.008,.044,.006,.006,.003,.002,.013,.001,.005,.0102,.001,.0005,.006,.005,.0006,.012,.004,.005,.005,.0005,.001,.003,.003,.0004,.0002)
STAGES = (_stage(STAGE1_NAMES, STAGE1_WEIGHTS), _stage(STAGE2_NAMES, STAGE2_WEIGHTS), _stage(STAGE3_NAMES, STAGE3_WEIGHTS))

# Opt-in variants of stage 2/3 ("2b"/"3b") that interleave the
# distractor_reasoning synthetic generator suite against the regular stage
# 2/3 recipe (rescaled to sum to the remainder) rather than replacing it
# outright. Stage 1 is untouched -- these only apply once training reaches
# the stage-2/3 annealing portion of the recipe.
#
# 40%, not 60% (2026-08-23): the pregenerated distractor-reasoning shard
# pool targets ~100GB (~26.8B tokens). At 60% weight, the full 100B-token
# force_stage=1 run would need ~60B tokens from it -- ~2.24x repetition,
# over the "twice at most" bound the pregeneration was sized against. 40%
# needs ~40B tokens (~1.5x repetition), comfortably under, with more
# margin once the SmolTalk-reasoning/OpenThoughts/Tulu3 chat-turn-aside
# corpora (Runbooks/GTR_SONIC_SSM_PRODUCTION_JOURNAL.md) widen the pool
# further.
DISTRACTOR_REASONING_WEIGHT = 0.4


def _drop_prefix_and_rescale(
    names: Sequence[str], weights: Sequence[float], *, prefix: str,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Remove every source whose name starts with `prefix`, rescaling the
    remaining weights back up so they still sum to the original total."""
    pairs = tuple(zip(names, weights, strict=True))
    kept = [(name, weight) for name, weight in pairs if not name.startswith(prefix)]
    dropped_weight = sum(weight for name, weight in pairs if name.startswith(prefix))
    original_total = sum(weights)
    remaining_total = original_total - dropped_weight
    if remaining_total <= 0:
        raise ValueError(f"dropping prefix {prefix!r} leaves no weight to rescale")
    scale = original_total / remaining_total
    return tuple(name for name, _ in kept), tuple(weight * scale for _, weight in kept)


# fw2-* (2026-08-26): dropped from the stage-2b mixture, not just deweighted
# -- gtr_arw_english_50k is an English-only 50k-vocab tokenizer, and 8 of
# the 14 fw2-* languages aren't even Latin-script (Chinese, Russian,
# Persian, Japanese, Korean, Hindi, Thai, Greek), so that content was
# almost certainly tokenizing into heavy byte-fallback rather than
# meaningful subword units -- wasted token budget for a project whose
# whole thesis is a lean, English-focused edge-deployable model. Only
# applied to the stage-2b/3b (distractor-reasoning) variants actually in
# use for this run, not the plain stage 1-3 recipe, which stays a faithful
# SmolLM3-style baseline for comparison purposes.
STAGE2_ENGLISH_NAMES, STAGE2_ENGLISH_WEIGHTS = _drop_prefix_and_rescale(
    STAGE2_NAMES, STAGE2_WEIGHTS, prefix="fw2-",
)
STAGE3_ENGLISH_NAMES, STAGE3_ENGLISH_WEIGHTS = _drop_prefix_and_rescale(
    STAGE3_NAMES, STAGE3_WEIGHTS, prefix="fw2-",
)
STAGE2B_NAMES = STAGE2_ENGLISH_NAMES + (DISTRACTOR_REASONING_SOURCE_NAME,)
STAGE2B_WEIGHTS = tuple(w * (1.0 - DISTRACTOR_REASONING_WEIGHT) for w in STAGE2_ENGLISH_WEIGHTS) + (DISTRACTOR_REASONING_WEIGHT,)
STAGE3B_NAMES = STAGE3_ENGLISH_NAMES + (DISTRACTOR_REASONING_SOURCE_NAME,)
STAGE3B_WEIGHTS = tuple(w * (1.0 - DISTRACTOR_REASONING_WEIGHT) for w in STAGE3_ENGLISH_WEIGHTS) + (DISTRACTOR_REASONING_WEIGHT,)
STAGES_DISTRACTOR_REASONING = (
    _stage(STAGE1_NAMES, STAGE1_WEIGHTS),
    _stage(STAGE2B_NAMES, STAGE2B_WEIGHTS),
    _stage(STAGE3B_NAMES, STAGE3B_WEIGHTS),
)


def _scaled_update_count(
    base_updates: int,
    sequence_length: int,
    effective_batch_sequences: int = EFFECTIVE_BATCH_SEQUENCES,
) -> int:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if effective_batch_sequences <= 0:
        raise ValueError("effective_batch_sequences must be positive")
    base_tokens_per_update = SEQUENCE_LENGTH * EFFECTIVE_BATCH_SEQUENCES
    tokens_per_update = sequence_length * effective_batch_sequences
    return math.ceil(base_updates * base_tokens_per_update / tokens_per_update)


def total_updates_for_sequence_length(
    sequence_length: int,
    effective_batch_sequences: int = EFFECTIVE_BATCH_SEQUENCES,
) -> int:
    """Return updates needed to retain the 26B-token budget."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if effective_batch_sequences <= 0:
        raise ValueError("effective_batch_sequences must be positive")
    tokens_per_update = sequence_length * effective_batch_sequences
    return math.ceil(TOTAL_TOKENS / tokens_per_update)


def stage_for_update(
    update: int,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    effective_batch_sequences: int = EFFECTIVE_BATCH_SEQUENCES,
) -> int:
    """Return the token-equivalent zero-based SmolLM3 corpus stage."""
    if update < 0:
        raise ValueError("update cannot be negative")
    stage2 = _scaled_update_count(STAGE2_UPDATE, sequence_length, effective_batch_sequences)
    stage3 = _scaled_update_count(STAGE3_UPDATE, sequence_length, effective_batch_sequences)
    return 0 if update < stage2 - 1 else (1 if update < stage3 - 1 else 2)


def accumulation_steps_for_batch(
    batch_size: int,
    effective_batch_sequences: int = EFFECTIVE_BATCH_SEQUENCES,
) -> int:
    """Preserve the recipe's configured effective-batch formation count."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if effective_batch_sequences % batch_size:
        raise ValueError(
            "batch_size must divide the configured effective batch"
        )
    return effective_batch_sequences // batch_size


def decay_boundaries_for_updates(
    sequence_length: int = SEQUENCE_LENGTH,
    effective_batch_sequences: int = EFFECTIVE_BATCH_SEQUENCES,
    total_updates: int | None = None,
) -> tuple[int, int]:
    """Return (stage3_update, decay_updates) -- where decay starts and how
    long it lasts, in current-config recipe-update units.

    total_updates, when given, is the run's ACTUAL intended length (i.e.
    --steps) -- the base "smollm3-26b" recipe's own internal boundaries
    (sized for its own ~26B-token cycle) get rescaled to land at the same
    *fractional* position (~89% stable-until, ~11% decay) of that real
    length instead. Without this, a production run stretched well past one
    recipe cycle (--steps far larger than
    total_updates_for_sequence_length(...), e.g. scaling a 26B-token recipe
    out to a 100B-token target) hits the base recipe's boundaries at a
    small fraction of its real length and collapses to LR=0 there, wasting
    the rest of the token budget at a dead learning rate.
    """
    stage3_update = _scaled_update_count(STAGE3_UPDATE, sequence_length, effective_batch_sequences)
    decay_updates = _scaled_update_count(DECAY_UPDATES, sequence_length, effective_batch_sequences)
    if total_updates is not None:
        if total_updates <= 0:
            raise ValueError("total_updates must be positive")
        base_total = _scaled_update_count(TOTAL_UPDATES, sequence_length, effective_batch_sequences)
        if base_total <= 0:
            raise ValueError("scaled base recipe length must be positive")
        scale = total_updates / base_total
        stage3_update = round(stage3_update * scale)
        decay_updates = max(1, round(decay_updates * scale))
    return stage3_update, decay_updates


def learning_rate_for_update(
    update: int,
    *,
    peak_lr: float = PEAK_LR,
    sequence_length: int = SEQUENCE_LENGTH,
    effective_batch_sequences: int = EFFECTIVE_BATCH_SEQUENCES,
    total_updates: int | None = None,
) -> float:
    """LR used by one-based update, preserving token-based boundaries.

    See decay_boundaries_for_updates for what total_updates does. warmup
    deliberately stays unscaled regardless -- it's a fixed short
    stabilization window, not proportional to total run length.
    """
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")
    warmup_updates = _scaled_update_count(WARMUP_UPDATES, sequence_length, effective_batch_sequences)
    stage3_update, decay_updates = decay_boundaries_for_updates(
        sequence_length, effective_batch_sequences, total_updates,
    )
    if update <= 0:
        return 0.0
    if update <= warmup_updates:
        return peak_lr * update / warmup_updates
    if update < stage3_update:
        return peak_lr
    elapsed = update - stage3_update + 1
    return peak_lr * max(0.0, 1.0 - elapsed / decay_updates)


class HFStreamingTextSource:
    """One lazy HF stream with bounded RAM and asynchronous disk chunks."""

    FORMAT = "gator-hf-text-stream-v3"
    exact_state_supported = True

    def __init__(
        self,
        spec: SourceSpec,
        tokenizer: Any,
        *,
        eos_token_id: int,
        cache_root: str | Path,
        executor: ThreadPoolExecutor,
        chunk_tokens: int,
        prefetch_tokens: int,
        max_document_chars: int,
    ) -> None:
        if chunk_tokens <= 0 or not 0 <= prefetch_tokens < chunk_tokens or max_document_chars <= 0:
            raise ValueError("invalid SmolLM3 token chunk/prefetch sizes")
        self.spec, self.tokenizer, self.eos_token_id = spec, tokenizer, eos_token_id
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.name)
        self.cache_dir = Path(cache_root) / safe_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.executor = executor
        self.chunk_tokens = int(chunk_tokens)
        self.prefetch_tokens = int(prefetch_tokens)
        self.max_document_chars = int(max_document_chars)
        self._dataset = None
        self._iterator = None
        self._native_state: Mapping[str, Any] | None = None
        self._stream_lock = threading.Lock()
        self._document_tokens = np.empty(0, dtype=np.uint32)
        self._document_offset = 0
        self._current = np.empty(0, dtype=np.uint32)
        self._current_offset = 0
        self._restored_next = np.empty(0, dtype=np.uint32)
        self._prefetch: Future[Path] | None = None
        self._prefetched_path: Path | None = None
        self._chunk_serial = 0
        self.documents = 0
        self.truncated_documents = 0
        self._first_row_reported = False

    def _open(self) -> None:
        if self._dataset is None:
            opened = time.perf_counter()
            print(
                f"[smollm3:data] source={self.spec.name} event=open_start "
                f"repo={self.spec.repo} config={self.spec.config}",
                flush=True,
            )
            try:
                from datasets import load_dataset
            except ImportError as error:
                raise RuntimeError("SmolLM3 streaming requires `pip install datasets`") from error
            self._dataset = load_dataset(
                self.spec.repo,
                self.spec.config,
                split=self.spec.split,
                revision=self.spec.revision,
                streaming=True,
                **dict(self.spec.load_kwargs),
            )
            if self._native_state is not None:
                self._dataset.load_state_dict(self._native_state)
                self._native_state = None
            print(
                f"[smollm3:data] source={self.spec.name} event=open_done "
                f"seconds={time.perf_counter() - opened:.2f}",
                flush=True,
            )
        self._iterator = iter(self._dataset)

    @staticmethod
    def _semantic_parts(value: Any) -> list[str]:
        semantic_keys = {
            "text", "content", "code", "prompt", "instruction", "question",
            "problem", "reference_answer", "answer", "solution",
            "generated_solution", "response", "responses", "reasoning",
            "explanation", "output", "completion", "chosen", "messages",
            "conversation", "conversations", "openai_response", "func_code_string",
        }
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (list, tuple)):
            result: list[str] = []
            for item in value:
                result.extend(HFStreamingTextSource._semantic_parts(item))
            return result
        if isinstance(value, Mapping):
            result = []
            for key, item in value.items():
                if str(key).lower() in semantic_keys:
                    result.extend(HFStreamingTextSource._semantic_parts(item))
            return result
        return []

    def _text(self, row: Mapping[str, Any]) -> str:
        for key in self.spec.text_columns:
            value = row.get(key)
            if isinstance(value, str) and value:
                return value
        parts = self._semantic_parts(row)
        if parts:
            return "\n\n".join(parts)
        columns = tuple(str(key) for key in row)
        raise ValueError(
            f"{self.spec.name} ({self.spec.repo}/{self.spec.config}) has no usable text; "
            f"available columns={columns}. This source exposes metadata pointers "
            "and requires Nanotron/DataTrove tokenized shards."
        )

    def _next_row(self) -> Mapping[str, Any]:
        if self._iterator is None:
            self._open()
        try:
            row = next(self._iterator)
        except StopIteration:
            self._dataset = self._iterator = None
            self._open()
            row = next(self._iterator)
        if not self._first_row_reported:
            self._first_row_reported = True
            print(
                f"[smollm3:data] source={self.spec.name} event=first_row "
                f"columns={tuple(str(key) for key in row)}",
                flush=True,
            )
        return row

    def _fill_document_tokens(self) -> None:
        row = self._next_row()
        try:
            text = self._text(row)
        except Exception as error:
            raise RuntimeError(f"failed to materialize SmolLM3 source {self.spec.name}") from error
        truncated = False
        if len(text) > self.max_document_chars:
            text = text[:self.max_document_chars]
            truncated = True
        first_document = self.documents == 0
        tokenized = time.perf_counter()
        if first_document:
            print(
                f"[smollm3:data] source={self.spec.name} event=tokenize_start "
                f"characters={len(text)}",
                flush=True,
            )
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.chunk_tokens - 1,
        )
        if first_document:
            print(
                f"[smollm3:data] source={self.spec.name} event=tokenize_done "
                f"tokens={len(ids)} seconds={time.perf_counter() - tokenized:.2f}",
                flush=True,
            )
        if len(ids) >= self.chunk_tokens - 1:
            ids = ids[:self.chunk_tokens - 1]
            truncated = True
        ids.append(self.eos_token_id)
        self._document_tokens = np.asarray(ids, dtype=np.uint32)
        self._document_offset = 0
        self.documents += 1
        self.truncated_documents += int(truncated)

    def _build_chunk_to_disk(self, serial: int) -> Path:
        with self._stream_lock:
            started = time.perf_counter()
            report_progress = serial == 0
            next_report = 8_192
            if report_progress:
                print(
                    f"[smollm3:data] source={self.spec.name} event=chunk_start "
                    f"tokens={self.chunk_tokens}",
                    flush=True,
                )
            parts: list[np.ndarray] = []
            remaining = self.chunk_tokens
            while remaining:
                if self._document_offset == len(self._document_tokens):
                    self._fill_document_tokens()
                available = len(self._document_tokens) - self._document_offset
                take = min(remaining, available)
                parts.append(self._document_tokens[self._document_offset:self._document_offset + take])
                self._document_offset += take
                remaining -= take
                completed = self.chunk_tokens - remaining
                while report_progress and completed >= next_report:
                    print(
                        f"[smollm3:data] source={self.spec.name} "
                        f"event=chunk_progress tokens={next_report}/{self.chunk_tokens}",
                        flush=True,
                    )
                    next_report += 8_192
            chunk = np.concatenate(parts) if len(parts) > 1 else parts[0].copy()
            path = self.cache_dir / f"chunk_{serial:012d}.npy"
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, chunk, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            if report_progress:
                print(
                    f"[smollm3:data] source={self.spec.name} event=chunk_done "
                    f"tokens={len(chunk)} seconds={time.perf_counter() - started:.2f}",
                    flush=True,
                )
            return path

    def _schedule_prefetch(self) -> None:
        if self._prefetch is not None or self._prefetched_path is not None or len(self._restored_next):
            return
        serial = self._chunk_serial
        self._chunk_serial += 1
        self._prefetch = self.executor.submit(self._build_chunk_to_disk, serial)

    def _collect_prefetch(self) -> None:
        if self._prefetch is not None:
            future = self._prefetch
            self._prefetch = None
            try:
                self._prefetched_path = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"SmolLM3 prefetch failed for source {self.spec.name}"
                ) from error

    def _promote_next(self) -> None:
        if len(self._restored_next):
            self._current = self._restored_next
            self._restored_next = np.empty(0, dtype=np.uint32)
        else:
            if self._prefetch is None and self._prefetched_path is None:
                # The first chunk is immediately required, so build it on the
                # caller thread. Besides avoiding a pointless submit/wait,
                # this keeps HF/PyArrow initialization exceptions out of a
                # worker during interpreter shutdown.
                serial = self._chunk_serial
                self._chunk_serial += 1
                self._prefetched_path = self._build_chunk_to_disk(serial)
            else:
                self._collect_prefetch()
            assert self._prefetched_path is not None
            self._current = np.load(self._prefetched_path, allow_pickle=False)
            self._prefetched_path.unlink(missing_ok=True)
            self._prefetched_path = None
        self._current_offset = 0

    def _current_remaining(self) -> int:
        return len(self._current) - self._current_offset

    def take(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("token count must be positive")
        result: list[int] = []
        while len(result) < count:
            if self._current_remaining() == 0:
                self._promote_next()
            take = min(count - len(result), self._current_remaining())
            values = self._current[self._current_offset:self._current_offset + take]
            result.extend(int(value) for value in values)
            self._current_offset += take
            if self._current_remaining() <= self.prefetch_tokens:
                self._schedule_prefetch()
        return result

    @property
    def stream_open(self) -> bool:
        return self._dataset is not None

    @property
    def stream_resident(self) -> bool:
        """Whether this source owns or is about to own a live HF reader."""
        return self._dataset is not None or self._prefetch is not None

    def requires_stream(self, count: int) -> bool:
        """Whether satisfying the next take can require opening the HF reader."""
        buffered = self._current_remaining() + len(self._restored_next)
        if self._prefetched_path is not None or self._prefetch is not None:
            buffered += self.chunk_tokens
        return buffered < count

    def will_activate_stream(self, count: int) -> bool:
        """Predict a foreground open or low-water asynchronous prefetch."""
        if self.stream_resident:
            return False
        if self.requires_stream(count):
            return True
        return (
            not len(self._restored_next)
            and self._prefetched_path is None
            and self._current_remaining() - count <= self.prefetch_tokens
        )

    def release_stream(self) -> bool:
        """Evict heavyweight HF/PyArrow objects while retaining an exact cursor."""
        # A running prefetch may already have advanced the dataset cursor. Keep
        # its completed chunk and only then snapshot the native stream state.
        self._collect_prefetch()
        with self._stream_lock:
            if self._dataset is None:
                self._iterator = None
                return False
            try:
                self._native_state = self._dataset.state_dict()
            except Exception as error:
                raise RuntimeError(
                    f"cannot preserve cursor while evicting SmolLM3 source {self.spec.name}"
                ) from error
            self._iterator = None
            self._dataset = None
        print(
            f"[smollm3:data] source={self.spec.name} event=stream_evicted",
            flush=True,
        )
        return True

    def state_dict(self) -> dict[str, Any]:
        self._collect_prefetch()
        native = (
            self._native_state
            if self._dataset is None
            else self._dataset.state_dict()
        )
        prefetched = (
            np.load(self._prefetched_path, allow_pickle=False)
            if self._prefetched_path is not None
            else self._restored_next.copy()
        )
        return {
            "format": self.FORMAT, "identity": _source_identity(self.spec), "native": native,
            "document_tokens": self._document_tokens[self._document_offset:].copy(),
            "current": self._current[self._current_offset:].copy(),
            "prefetched": prefetched.copy(), "documents": self.documents,
            "truncated_documents": self.truncated_documents,
            "chunk_serial": self._chunk_serial, "chunk_tokens": self.chunk_tokens,
            "prefetch_tokens": self.prefetch_tokens,
            "max_document_chars": self.max_document_chars, "exact": True,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("format") != self.FORMAT
            or state.get("identity") != _source_identity(self.spec)
            or int(state.get("chunk_tokens", -1)) != self.chunk_tokens
            or int(state.get("prefetch_tokens", -1)) != self.prefetch_tokens
            or int(state.get("max_document_chars", -1)) != self.max_document_chars
        ):
            raise ValueError(f"HF stream identity changed for {self.spec.name}")
        self._document_tokens = np.asarray(state.get("document_tokens", ()), dtype=np.uint32)
        self._document_offset = 0
        self._current = np.asarray(state.get("current", ()), dtype=np.uint32)
        self._current_offset = 0
        self._restored_next = np.asarray(state.get("prefetched", ()), dtype=np.uint32)
        self.documents = int(state.get("documents", 0))
        self.truncated_documents = int(state.get("truncated_documents", 0))
        self._chunk_serial = int(state.get("chunk_serial", 0))
        self._dataset = self._iterator = None
        self._native_state = state.get("native")

    def close(self) -> None:
        future, self._prefetch = self._prefetch, None
        if future is not None:
            future.cancel()
            try:
                completed_path = future.result()
            except CancelledError:
                completed_path = None
            except Exception:
                completed_path = None
            if completed_path is not None:
                completed_path.unlink(missing_ok=True)
        if self._prefetched_path is not None:
            self._prefetched_path.unlink(missing_ok=True)
            self._prefetched_path = None
        for path in self.cache_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        for path in self.cache_dir.glob("*.npy"):
            path.unlink(missing_ok=True)
        self._iterator = None
        self._dataset = None


class DataTroveTokenizedSource:
    """Sequential, one-file-at-a-time reader for Nanotron `.ds` shards."""

    FORMAT = "gator-datatrove-token-stream-v1"
    exact_state_supported = True

    def __init__(self, spec: SourceSpec, data_root: str | Path) -> None:
        self.spec = spec
        root = Path(data_root)
        candidates = [root / spec.name]
        if spec.name.startswith("stack-v2-"):
            candidates.append(root / spec.name.replace("stack-v2-", "stack-edu-", 1))
        self.folder = next((path for path in candidates if path.is_dir()), candidates[0])
        self.files = tuple(sorted(self.folder.glob("*.ds")))
        if not self.files:
            raise FileNotFoundError(f"no DataTrove .ds shards found for {spec.name} under {self.folder}")
        if not any(path.stat().st_size for path in self.files):
            raise ValueError(f"all DataTrove shards are empty for {spec.name}")
        metadata = tuple(sorted(self.folder.glob("*.metadata")))
        self.token_size = 4
        if metadata:
            try:
                _, size = metadata[0].read_text(encoding="utf-8").splitlines()[0].split("|", 1)
                self.token_size = int(size)
            except (IndexError, ValueError) as error:
                raise ValueError(f"invalid DataTrove metadata in {metadata[0]}") from error
        if self.token_size not in {2, 4, 8}:
            raise ValueError(f"unsupported DataTrove token size {self.token_size}")
        self.dtype = np.dtype({2:"<u2", 4:"<u4", 8:"<u8"}[self.token_size])
        self.file_index = 0
        self.token_offset = 0
        self._handle = None

    def _open_current(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = self.files[self.file_index].open("rb")
        self._handle.seek(self.token_offset * self.token_size)

    def take(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("token count must be positive")
        result: list[int] = []
        while len(result) < count:
            if self._handle is None:
                self._open_current()
            needed = count - len(result)
            values = np.fromfile(self._handle, dtype=self.dtype, count=needed)
            result.extend(int(value) for value in values)
            self.token_offset += len(values)
            if len(values) < needed:
                self.file_index = (self.file_index + 1) % len(self.files)
                self.token_offset = 0
                self._open_current()
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT, "identity": _source_identity(self.spec),
            "files": tuple((path.name, path.stat().st_size) for path in self.files),
            "token_size": self.token_size, "file_index": self.file_index,
            "token_offset": self.token_offset, "exact": True,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        identity = tuple((path.name, path.stat().st_size) for path in self.files)
        if (
            state.get("format") != self.FORMAT or state.get("identity") != _source_identity(self.spec)
            or tuple(tuple(item) for item in state.get("files", ())) != identity
            or int(state.get("token_size", -1)) != self.token_size
        ):
            raise ValueError(f"DataTrove stream identity changed for {self.spec.name}")
        self.file_index = int(state["file_index"])
        self.token_offset = int(state["token_offset"])
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _nanotron_blend_choice(weights: Sequence[float], counts: Sequence[int], sample_index: int) -> int:
    """Match Megatron/Nanotron's largest-mixture-deficit schedule."""
    total = float(sum(weights))
    if not total > 0:
        raise ValueError("mixture weights must have positive sum")
    point = max(float(sample_index), 1.0)
    errors = [(float(weight) / total) * point - int(count) for weight, count in zip(weights, counts, strict=True)]
    return max(range(len(errors)), key=errors.__getitem__)


class SmolLM3BatchProvider:
    """Stage-aware exact-mixture stream producing GTR context/output windows."""
    FORMAT = "gator-smollm3-provider-v3"
    exact_state_supported = True

    def __init__(
        self, tokenizer: Any, *, context_tokens: int, output_slots: int,
        seed: int = 6, sources: Sequence[Sequence[SourceSpec]] = STAGES,
        tokenized_data_root: str | Path | None = None,
        stream_cache_dir: str | Path | None = None,
        chunk_tokens: int = 65_536, prefetch_tokens: int = 8_192,
        prefetch_workers: int = 2, max_document_chars: int = 262_144,
        max_open_streams: int = 2, max_host_ram_gb: float = 20.0,
        memory_log_every_seconds: float = 60.0,
        effective_batch_sequences: int = EFFECTIVE_BATCH_SEQUENCES,
        distractor_reasoning_chat_data: Sequence[str | Path] = (),
        distractor_reasoning_tokenizer: Any | None = None,
        force_stage: int | None = None,
    ) -> None:
        if prefetch_workers <= 0:
            raise ValueError("SmolLM3 prefetch workers must be positive")
        if max_open_streams <= 0:
            raise ValueError("SmolLM3 max open streams must be positive")
        if max_host_ram_gb < 0 or memory_log_every_seconds < 0:
            raise ValueError("SmolLM3 RAM limit and memory log interval cannot be negative")
        if effective_batch_sequences <= 0:
            raise ValueError("SmolLM3 effective_batch_sequences must be positive")
        if force_stage is not None and not 0 <= force_stage < len(sources):
            raise ValueError("SmolLM3 force_stage must index an existing recipe stage")
        self.force_stage = force_stage
        self.tokenizer, self.context_tokens, self.output_slots = tokenizer, context_tokens, output_slots
        # Loaded once regardless of whether any stage actually uses the
        # distractor-reasoning source, same as the streaming executor below
        # -- cheap when empty, and building it lazily per-source would mean
        # reparsing the same files for stage 2b and 3b.
        # Never the same object as `tokenizer` in real use -- HF fast
        # tokenizers mutate internal Rust state on every .encode() call,
        # unsafe to share with the background prefetch threads that call
        # tokenizer.encode() concurrently below (same hazard already fixed
        # for the fixed-prompt eval harness's own tokenizer). Falls back to
        # the shared instance only for callers (tests) that don't pass one
        # and never exercise real concurrency.
        self._distractor_reasoning_tokenizer = (
            distractor_reasoning_tokenizer if distractor_reasoning_tokenizer is not None else tokenizer
        )
        self._distractor_reasoning_chat_turn_pool = (
            load_chat_turn_pool(distractor_reasoning_chat_data)
            if distractor_reasoning_chat_data else ()
        )
        self.sequence_length = context_tokens + output_slots
        self.effective_batch_sequences = int(effective_batch_sequences)
        self.seed = int(seed)
        self._closed = False
        self.max_open_streams = int(max_open_streams)
        self.max_host_ram_bytes = int(float(max_host_ram_gb) * (1024 ** 3))
        self.memory_log_every_seconds = float(memory_log_every_seconds)
        self._last_memory_log = 0.0
        self._live_hf_sources: list[HFStreamingTextSource] = []
        self.stream_evictions = 0
        # Always allocate the streaming executor now, even when
        # tokenized_data_root is set -- mixed mode means some specs may
        # still fall back to live HF streaming (see _build_source below),
        # so a pure "local-only means no executor needed" assumption no
        # longer holds. The idle-thread-pool cost when every source ends
        # up local is negligible.
        self._executor = ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="smollm3-prefetch")
        if stream_cache_dir:
            cache_parent = Path(stream_cache_dir)
            cache_parent.mkdir(parents=True, exist_ok=True)
            self._cache_root = cache_parent / f"run_{os.getpid()}_{id(self):x}"
            self._cache_root.mkdir(parents=True, exist_ok=False)
        else:
            self._cache_root = Path(tempfile.mkdtemp(prefix="gator-smollm3-cache-"))

        def _build_source(spec: SourceSpec) -> Any:
            # Mixed mode: prefer local DataTrove .ds shards per-source when
            # tokenized_data_root is set and that specific source has them
            # prepared; fall back to live HF streaming (or, for the
            # distractor-reasoning pseudo-source, live generation) for any
            # source that doesn't -- rather than the old all-or-nothing
            # switch, which required every source in the mixture to have
            # local shards before any of them could be used. Applies to
            # distractor-reasoning too: live generation inside the full
            # training process (compile + threading + everything else
            # running concurrently) proved unstable in practice even
            # though the generator itself profiled as fast in isolation --
            # pregenerated .ds shards sidestep whatever that interaction
            # is, the same way local shards already do for the corpus
            # sources.
            if tokenized_data_root is not None:
                try:
                    source = DataTroveTokenizedSource(spec, tokenized_data_root)
                except (FileNotFoundError, ValueError):
                    pass
                else:
                    print(f"[smollm3:data] source={spec.name} event=mode mode=local_datatrove", flush=True)
                    return source
            if spec.name == DISTRACTOR_REASONING_SOURCE_NAME:
                print(f"[smollm3:data] source={spec.name} event=mode mode=distractor_reasoning", flush=True)
                return DistractorReasoningTextSource(
                    spec, self._distractor_reasoning_tokenizer, seed=self.seed,
                    chat_turn_pool=self._distractor_reasoning_chat_turn_pool,
                )
            print(f"[smollm3:data] source={spec.name} event=mode mode=hf_streaming", flush=True)
            return HFStreamingTextSource(
                spec, tokenizer, eos_token_id=tokenizer.eos_token_id,
                cache_root=self._cache_root, executor=self._executor,
                chunk_tokens=chunk_tokens, prefetch_tokens=prefetch_tokens,
                max_document_chars=max_document_chars,
            )

        # When force_stage pins sampling to one stage for the run's entire
        # lifetime, stages that will never be sampled don't need their
        # sources built at all -- eagerly constructing every stage's
        # sources regardless of force_stage was spawning network
        # connections/threads for ~2 extra stages' worth of sources (~100
        # unused sources) purely for nothing, real enough to exhaust file
        # descriptors and crash an otherwise-healthy source (dclm) with an
        # unrelated 'Bad file descriptor' error during a force_stage run.
        self.sources = tuple(
            tuple(_build_source(spec) for spec in stage)
            if force_stage is None or index == force_stage else ()
            for index, stage in enumerate(sources)
        )
        self.blend_positions = [0 for _ in self.sources]
        self.stage_counts = [[0 for _ in stage] for stage in self.sources]
        self.update = 0
        self.micro_step = 0
        self.counts: dict[str, int] = {}
        self.last_source_names: tuple[str, ...] = ()
        atexit.register(self.close)

    @property
    def cache_root(self) -> Path:
        return self._cache_root

    def _prune_live_streams(self) -> None:
        self._live_hf_sources = [
            source for source in self._live_hf_sources if source.stream_resident
        ]

    def _evict_stream(self, source: HFStreamingTextSource) -> None:
        if source in self._live_hf_sources:
            self._live_hf_sources.remove(source)
        if source.release_stream():
            self.stream_evictions += 1

    def _before_source_take(self, source: Any, count: int) -> None:
        if not isinstance(source, HFStreamingTextSource):
            return
        self._prune_live_streams()
        if not source.will_activate_stream(count):
            return
        while len(self._live_hf_sources) >= self.max_open_streams:
            self._evict_stream(self._live_hf_sources[0])
        gc.collect()

    def _after_source_take(self, source: Any) -> None:
        if not isinstance(source, HFStreamingTextSource) or not source.stream_resident:
            return
        if source in self._live_hf_sources:
            self._live_hf_sources.remove(source)
        self._live_hf_sources.append(source)
        while len(self._live_hf_sources) > self.max_open_streams:
            self._evict_stream(self._live_hf_sources[0])

    def _check_host_memory(self, *, force_log: bool = False) -> None:
        rss = _process_rss_bytes()
        now = time.monotonic()
        should_log = force_log or (
            self.memory_log_every_seconds > 0
            and now - self._last_memory_log >= self.memory_log_every_seconds
        )
        if should_log:
            self._last_memory_log = now
            print(
                f"[smollm3:memory] rss_gb={rss / (1024 ** 3):.2f} "
                f"open_streams={len(self._live_hf_sources)}/{self.max_open_streams} "
                f"evictions={self.stream_evictions}",
                flush=True,
            )
        if not self.max_host_ram_bytes or rss <= self.max_host_ram_bytes:
            return

        # Release every resumable reader before deciding the process is truly
        # above budget. Token chunks and exact cursors remain intact.
        while self._live_hf_sources:
            self._evict_stream(self._live_hf_sources[0])
        gc.collect()
        reduced_rss = _process_rss_bytes()
        print(
            f"[smollm3:memory] event=pressure rss_gb={rss / (1024 ** 3):.2f} "
            f"after_evict_gb={reduced_rss / (1024 ** 3):.2f} "
            f"limit_gb={self.max_host_ram_bytes / (1024 ** 3):.2f}",
            flush=True,
        )
        if reduced_rss > self.max_host_ram_bytes:
            raise MemoryError(
                "SmolLM3 direct-HF loader exceeded --smollm3-max-host-ram-gb "
                f"({reduced_rss / (1024 ** 3):.2f} GiB > "
                f"{self.max_host_ram_bytes / (1024 ** 3):.2f} GiB). "
                "Lower --smollm3-max-open-streams or use tokenized .ds shards."
            )

    def _carry_source_cursors(self, target_stage: int) -> None:
        """Continue reused logical sources across Nanotron recipe stages."""
        for target in self.sources[target_stage]:
            previous = None
            for prior_stage in range(target_stage - 1, -1, -1):
                previous = next(
                    (
                        source
                        for source in self.sources[prior_stage]
                        if source.spec.name == target.spec.name
                    ),
                    None,
                )
                if previous is not None:
                    break
            if previous is not None:
                target.load_state_dict(previous.state_dict())

    def _stage_for_update(self, update: int) -> int:
        if self.force_stage is not None:
            return self.force_stage
        return stage_for_update(
            update,
            sequence_length=self.sequence_length,
            effective_batch_sequences=self.effective_batch_sequences,
        )

    def set_progress(self, *, update: int, micro_step: int) -> None:
        old_stage = self._stage_for_update(self.update)
        new_stage = self._stage_for_update(update)
        if new_stage > old_stage:
            for target_stage in range(old_stage + 1, new_stage + 1):
                self._carry_source_cursors(target_stage)
        self.update, self.micro_step = update, micro_step

    def sample_cpu(self, batch_size: int) -> LanguageBatch:
        stage = self._stage_for_update(self.update)
        pool = self.sources[stage]
        picked = []
        weights = [source.spec.weight for source in pool]
        for _ in range(batch_size):
            index = _nanotron_blend_choice(weights, self.stage_counts[stage], self.blend_positions[stage])
            picked.append(pool[index])
            self.stage_counts[stage][index] += 1
            self.blend_positions[stage] += 1
        self.last_source_names = tuple(source.spec.name for source in picked)
        width = self.context_tokens + self.output_slots
        rows = []
        for source in picked:
            self._before_source_take(source, width)
            rows.append(source.take(width))
            self._after_source_take(source)
        self._check_host_memory()
        for source in picked:
            self.counts[source.spec.name] = self.counts.get(source.spec.name, 0) + 1
        values = torch.tensor(rows, dtype=torch.long)
        targets = values[:, self.context_tokens:]
        return LanguageBatch(
            values[:, :self.context_tokens],
            torch.ones((batch_size, self.context_tokens), dtype=torch.bool),
            targets,
            torch.ones((batch_size, self.output_slots), dtype=torch.bool),
            targets.eq(self.tokenizer.eos_token_id).any(dim=1),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT, "context_tokens": self.context_tokens,
            "output_slots": self.output_slots, "seed": self.seed,
            "update": self.update, "micro_step": self.micro_step,
            "blend_positions": tuple(self.blend_positions),
            "stage_counts": tuple(tuple(row) for row in self.stage_counts),
            "counts": dict(self.counts),
            "sources": tuple(tuple(source.state_dict() for source in stage) for stage in self.sources),
            "exact": True,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        allow_window_resize: bool = False,
    ) -> None:
        saved_context = int(state.get("context_tokens", -1))
        saved_output = int(state.get("output_slots", -1))
        topology_mismatch = (
            state.get("format") != self.FORMAT
            or saved_output != self.output_slots
            or saved_context != self.context_tokens
        )
        valid_growth = (
            allow_window_resize
            and state.get("format") == self.FORMAT
            and saved_output == self.output_slots
            and 0 <= saved_context < self.context_tokens
        )
        if topology_mismatch and not valid_growth:
            raise ValueError("SmolLM3 provider topology differs from checkpoint")
        nested = state.get("sources")
        if not isinstance(nested, (tuple, list)) or len(nested) != len(self.sources):
            raise ValueError("SmolLM3 stage topology differs from checkpoint")
        positions = state.get("blend_positions")
        stage_counts = state.get("stage_counts")
        if not isinstance(positions, (tuple, list)) or len(positions) != len(self.sources):
            raise ValueError("SmolLM3 blend positions differ from checkpoint")
        if not isinstance(stage_counts, (tuple, list)) or len(stage_counts) != len(self.sources):
            raise ValueError("SmolLM3 blend counts differ from checkpoint")
        # Source *cursors* are matched by name, not position: a mixture edit
        # that only removes a source (rather than just reweighting it)
        # changes a stage's source count, which used to hard-fail resume
        # even though every remaining source is otherwise identical to what
        # the checkpoint saved. A source still present just resumes its
        # exact read cursor as before; a source newly added since the
        # checkpoint was saved has no saved state to match and starts fresh.
        for current_stage, saved_stage in zip(self.sources, nested, strict=True):
            saved_by_name: dict[str, Mapping[str, Any]] = {
                name: saved
                for saved in saved_stage
                if (name := (saved.get("identity") or {}).get("name")) is not None
            }
            for source in current_stage:
                saved = saved_by_name.get(source.spec.name)
                if saved is not None:
                    source.load_state_dict(saved)
        self.blend_positions = [int(value) for value in positions]
        # stage_counts are deliberately NOT carried over from the checkpoint
        # -- they encode the *old* weight distribution's accumulated deficit,
        # and _nanotron_blend_choice only closes a deficit by ~1 per pick.
        # Any mixture edit this late into a long run (removing a source is
        # mathematically the same as bumping every survivor's weight)
        # produces an absolute deficit spike proportional to
        # blend_positions -- which is enormous after tens of millions of
        # picks -- and the largest-weight source then wins every single
        # pick for as long as it takes to linearly close that gap (tens of
        # thousands of picks, empirically, on this run). Rebasing every
        # source's count to its exact weight-proportional share of the
        # already-elapsed position sidesteps that entirely: it's a
        # near-no-op when weights didn't change (a healthy history is
        # already close to this target), and immediately consistent with
        # the current mixture when they did.
        rebased_stage_counts = []
        for current_stage, position in zip(self.sources, self.blend_positions, strict=True):
            stage_weight = sum(source.spec.weight for source in current_stage)
            rebased_stage_counts.append(
                [round((source.spec.weight / stage_weight) * position) for source in current_stage]
                if stage_weight > 0 else []
            )
        self.stage_counts = rebased_stage_counts
        self.update, self.micro_step = int(state.get("update", 0)), int(state.get("micro_step", 0))
        self.counts = {str(k): int(v) for k, v in state.get("counts", {}).items()}
        self._live_hf_sources.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        for stage in self.sources:
            for source in stage:
                source.close()
        shutil.rmtree(self._cache_root, ignore_errors=True)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def __enter__(self) -> "SmolLM3BatchProvider":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def adamw_parameter_groups(model: torch.nn.Module, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if "token_embedding" in name else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
