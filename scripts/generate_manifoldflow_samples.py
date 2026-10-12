#!/usr/bin/env python
"""Generates a handful of small, readable ManifoldFlow sessions and decodes
them back to text - for inspecting what the production recipe actually
trains on (the whole packed stream: chat, filler, and correction chains)
and what it's graded on (the later query/answer pairs), without wading
through a 32768-token production-sized session or raw token ids.

Same generator, vocabulary, and episode mix as
`pregenerate_manifoldflow_sessions.py` - just a small `--target-tokens`
default so a handful of sessions are easy to read end to end.

Usage:
    python scripts/generate_manifoldflow_samples.py --output samples.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))

_DEFAULT_TOKENIZER_DIR = _PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k"


def _decode_episodes(packed: dict, tokenizer) -> list[dict]:
    """Reduces one `pack_session()` result to plain, human-readable text -
    same episode classification `manifoldflow_production_train.build_window`
    uses (write vs. query), decoded instead of kept as token spans."""
    episodes_readable = []
    for ep in packed["episodes"]:
        start, end = ep["start"], ep["end"]
        entry = {
            "episode_type": ep["episode_type"],
            "text": tokenizer.decode(packed["token_ids"][start:end]),
            "num_tokens": end - start,
        }
        if ep["episode_type"] == "distractor":
            if "statement_spans" in ep:
                entry["role"] = "write"
                entry["target_statement_index"] = ep["target_statement_index"]
                entry["unit_id"] = ep["unit_id"]
                entry["statements"] = [
                    tokenizer.decode(packed["token_ids"][a:b])
                    for a, b in ep["statement_spans"]
                ]
            elif "query_span" in ep:
                entry["role"] = "query"
                entry["unit_id"] = ep["unit_id"]
                entry["target_kind"] = ep["target_kind"]
                qa, qb = ep["query_span"]
                aa, ab = ep["answer_span"]
                entry["question_text"] = tokenizer.decode(packed["token_ids"][qa:qb])
                entry["answer_text"] = tokenizer.decode(packed["token_ids"][aa:ab])
        episodes_readable.append(entry)
    return episodes_readable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--target-tokens", type=int, default=2500)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--progress-fraction", type=float, default=0.6)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument(
        "--chat-data-root",
        type=Path,
        default=None,
        help="Passed through to ChatSource resolution - omit to auto-resolve.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from manifoldflow.dataset.session_builder import (
        ChatSource,
        ChatTurnStream,
        _build_filler_source,
        build_session,
        pack_session,
    )
    from manifoldflow.dataset.vocab_expansion import expand_quantity_range, expand_vocabulary

    expand_vocabulary(seed=29)
    expand_quantity_range(max_quantity=9999)

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    chat_sources = (
        [
            ChatSource(path=args.chat_data_root / "smoltalk_everyday.jsonl"),
            ChatSource(path=args.chat_data_root / "lmsys_chat_1m.jsonl"),
        ]
        if args.chat_data_root is not None
        else None
    )
    chat_stream = ChatTurnStream(chat_sources, seed=args.seed)
    filler_source = _build_filler_source(
        tokenizer,
        eos_token_id=eos_token_id,
        cache_root=args.output.parent / "_filler_cache",
        seed=args.seed,
    )

    generator = random.Random(args.seed)
    sessions = []
    for session_id in range(args.sessions):
        session = build_session(
            session_id,
            generator,
            tokenizer,
            target_tokens=args.target_tokens,
            level=args.level,
            progress_fraction=args.progress_fraction,
            chat_stream=chat_stream,
            filler_source=filler_source,
        )
        packed = pack_session(session)
        sessions.append(
            {
                "session_id": session_id,
                "total_tokens": packed["total_tokens"],
                "episodes": _decode_episodes(packed, tokenizer),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    for s in sessions:
        print(f"session {s['session_id']}: {s['total_tokens']} tokens, {len(s['episodes'])} episodes")


if __name__ == "__main__":
    main()
