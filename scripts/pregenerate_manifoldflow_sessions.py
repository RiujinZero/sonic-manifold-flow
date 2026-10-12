#!/usr/bin/env python
"""Pregenerates ManifoldFlow production-training sessions to disk shards.

Must run ahead of training - see `session_builder.py`'s module docstring
for why (the 96-lane batch-lane probe's live-generation CPU bottleneck
earlier this session). Parallelized across worker processes, each writing
its own shard file so there's no cross-process coordination needed.

Usage:
    python scripts/pregenerate_manifoldflow_sessions.py \\
        --output-dir C:/path/to/manifoldflow_sessions \\
        --sessions 10000 --workers 8 --target-tokens 32768 --level 5
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))

_DEFAULT_TOKENIZER_DIR = _PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k"


def _worker(
    worker_index: int,
    *,
    sessions_for_worker: int,
    session_id_start: int,
    output_dir: Path,
    target_tokens: int,
    level: int,
    seed: int,
    sessions_per_shard: int,
    tokenizer_dir: Path,
    chat_data_root: Path | None,
) -> None:
    # Isolate this worker's HF Hub download cache before any transformers/
    # datasets import touches it - by default they all share one machine-
    # wide cache directory, and once the HF_TOKEN rate-limit fix let
    # workers download concurrently at real speed, that turned into a real
    # race: multiple processes renaming/opening the same cached parquet
    # file at once, surfaced as FileNotFoundError/PermissionError ("used
    # by another process" / "Access is denied") - confirmed as the actual
    # cause of a 20-of-28-worker crash cluster on a real run, not a
    # transient fluke. HF_HOME must be set before the first import that
    # might read it, since huggingface_hub resolves cache paths at import
    # time.
    #
    # Deliberately NOT nested under output_dir: HF's own cache layout is
    # already deep (hub/datasets--<org>--<name>/snapshots/<commit-hash>/
    # data/<split>/<file>.parquet), and nesting that under a long
    # output_dir pushed the full path past Windows' 260-char MAX_PATH -
    # confirmed as a real WinError 206 ("filename or extension is too
    # long") once this was tried nested. A short, drive-root-adjacent path
    # leaves headroom for HF's own nesting regardless of how deep
    # output_dir is.
    os.environ["HF_HOME"] = f"C:/_mf_hf_cache/w{worker_index}"

    from transformers import AutoTokenizer

    from manifoldflow.dataset.session_builder import (
        ChatSource,
        ChatTurnStream,
        _build_filler_source,
        build_session,
        pack_session,
        write_shard,
    )
    from manifoldflow.dataset.vocab_expansion import (
        expand_quantity_range,
        expand_vocabulary,
    )

    # Each worker process (spawned fresh on Windows - no shared state with
    # the parent) must patch its own copy of the vendored generator's
    # modules before the first _build_distractor_unit call.
    expand_vocabulary(seed=29)
    expand_quantity_range(max_quantity=9999)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    eos_token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
    )
    generator = random.Random(seed + worker_index)
    chat_sources = (
        [
            ChatSource(path=chat_data_root / "smoltalk_everyday.jsonl"),
            ChatSource(path=chat_data_root / "lmsys_chat_1m.jsonl"),
        ]
        if chat_data_root is not None
        else None  # ChatTurnStream resolves default_chat_sources() itself
    )
    chat_stream = ChatTurnStream(chat_sources, seed=seed + worker_index)
    filler_source = _build_filler_source(
        tokenizer,
        eos_token_id=eos_token_id,
        cache_root=output_dir / "_filler_cache" / f"worker{worker_index}",
        seed=seed + worker_index,
    )

    shard_sessions: list[dict] = []
    shard_index = 0
    start_time = time.monotonic()
    for offset in range(sessions_for_worker):
        session_id = session_id_start + offset
        # Progress fraction advances the query-distance curriculum
        # (session_builder._query_distance_is_mature) across the whole
        # pregenerated pool, not per-worker, so every worker should see
        # the same global progress at a given session_id.
        progress_fraction = session_id / max(1, sessions_for_worker * _worker_count_hint)
        session = build_session(
            session_id,
            generator,
            tokenizer,
            target_tokens=target_tokens,
            level=level,
            progress_fraction=min(1.0, progress_fraction),
            chat_stream=chat_stream,
            filler_source=filler_source,
        )
        shard_sessions.append(pack_session(session))
        if len(shard_sessions) >= sessions_per_shard:
            shard_path = output_dir / f"shard_w{worker_index:03d}_{shard_index:05d}.jsonl"
            write_shard(shard_sessions, shard_path)
            elapsed = time.monotonic() - start_time
            print(
                f"[worker {worker_index}] wrote {shard_path.name} "
                f"({offset + 1}/{sessions_for_worker} sessions, "
                f"{elapsed:.1f}s elapsed)",
                flush=True,
            )
            shard_sessions = []
            shard_index += 1

    if shard_sessions:
        shard_path = output_dir / f"shard_w{worker_index:03d}_{shard_index:05d}.jsonl"
        write_shard(shard_sessions, shard_path)
        print(f"[worker {worker_index}] wrote final {shard_path.name}", flush=True)


# Set once in main() before workers spawn; multiprocessing on Windows uses
# spawn, which re-imports this module fresh in each worker, so this must be
# a module-level global re-derivable from argv rather than a closure.
_worker_count_hint = 1


def main() -> None:
    global _worker_count_hint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--target-tokens", type=int, default=32_768)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sessions-per-shard", type=int, default=50)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument(
        "--chat-data-root",
        type=Path,
        default=None,
        help=(
            "Directory containing smoltalk_everyday.jsonl and "
            "lmsys_chat_1m.jsonl. Omit to auto-resolve the local G: drive "
            "export (see session_builder.default_chat_sources)."
        ),
    )
    args = parser.parse_args()

    _worker_count_hint = args.workers
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sessions_per_worker = args.sessions // args.workers
    remainder = args.sessions - sessions_per_worker * args.workers

    processes: list[tuple[int, int, multiprocessing.Process]] = []
    session_id_cursor = 0
    for worker_index in range(args.workers):
        count = sessions_per_worker + (1 if worker_index < remainder else 0)
        process = multiprocessing.Process(
            target=_worker,
            kwargs=dict(
                worker_index=worker_index,
                sessions_for_worker=count,
                session_id_start=session_id_cursor,
                output_dir=args.output_dir,
                target_tokens=args.target_tokens,
                level=args.level,
                seed=args.seed,
                sessions_per_shard=args.sessions_per_shard,
                tokenizer_dir=args.tokenizer_dir,
                chat_data_root=args.chat_data_root,
            ),
        )
        session_id_cursor += count
        processes.append((worker_index, count, process))
        process.start()

    for _, _, process in processes:
        process.join()

    # multiprocessing.Process.join() returns on ANY exit, crash included -
    # a worker dying partway through (see session_builder._local_chat_data_root's
    # docstring for a real instance: Google Drive dropping G: mid-run killed
    # ~13 of 28 workers at once) must not be reported as if every requested
    # session was actually produced.
    failed = [
        (worker_index, count)
        for worker_index, count, process in processes
        if process.exitcode != 0
    ]
    if failed:
        failed_sessions = sum(count for _, count in failed)
        print(
            f"WARNING: {len(failed)}/{args.workers} workers exited with a "
            f"non-zero exit code (assigned {failed_sessions} sessions total, "
            "likely undercounted - some may have written a partial shard "
            "before crashing). Check the log above for tracebacks. Failed "
            f"worker indices: {[index for index, _ in failed]}",
            flush=True,
        )
    print(
        f"done: {args.sessions - sum(c for _, c in failed)}/{args.sessions} "
        f"sessions requested across {args.workers} workers -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
