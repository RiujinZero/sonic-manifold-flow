"""Check design-v0.3.md Phase 0's exit gate for Sonic: "Identical text
produces stable keys and retrieval after restart."

Mirrors `qwen_deterministic_replay.py` exactly, against
`SonicRepresentationAdapter` instead: encodes a fixed set of event texts
twice in-process (same loaded model, repeat calls) and, when
`--compare-against` points at a previous run's saved output, once more
after a full process restart (fresh checkpoint load, fresh everything).
Sonic's recurrent SSM scan is a different numerical path from Qwen's
attention (an associative doubling scan, not a single fused kernel), so
this checks the same claim empirically rather than assuming Qwen's
bit-exact result transfers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from manifoldflow.sonic_adapter import (
    SonicRepresentationAdapter,
    SonicRepresentationConfig,
)

_DEFAULT_TEXTS = [
    "Neral owns the copper glider.",
    "Tovan placed the striped lantern in Vault 7.",
    "Mira moved from Box 4 to Box 9.",
    "The weather today is unusually clear and cold.",
]
_DEFAULT_CHECKPOINT = Path("checkpoints/sonic_ssm/sonic_ssm_128k_steps.pt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare-against", type=Path, default=None)
    parser.add_argument("--texts", nargs="+", default=_DEFAULT_TEXTS)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--pooling", default="mean", choices=["last_token", "mean"])
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.texts:
        raise ValueError("texts must be non-empty")

    torch.use_deterministic_algorithms(True, warn_only=True)

    config = SonicRepresentationConfig(
        checkpoint_path=args.checkpoint_path,
        pooling=args.pooling,
        projection_seed=args.projection_seed,
        output_dim=args.output_dim,
        device=args.device,
    )
    adapter = SonicRepresentationAdapter(config)

    keys: dict[str, Any] = {}
    for text in args.texts:
        first = adapter.encode_event(text)
        second = adapter.encode_event(text)
        in_process_max_abs_diff = (first - second).abs().max().item()
        keys[text] = {
            "key": first.tolist(),
            "in_process_repeat_max_abs_diff": in_process_max_abs_diff,
            "in_process_repeat_stable": in_process_max_abs_diff <= args.tolerance,
        }

    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": adapter.device,
        "tolerance": args.tolerance,
        "manifest": adapter.manifest(),
        "keys": keys,
    }

    if args.compare_against is not None:
        previous = json.loads(args.compare_against.read_text(encoding="utf-8"))
        manifest_matches = previous["manifest"] == result["manifest"]
        cross_restart: dict[str, Any] = {}
        for text, entry in keys.items():
            if text not in previous["keys"]:
                cross_restart[text] = {"error": "text absent from prior run"}
                continue
            prior_key = torch.tensor(previous["keys"][text]["key"])
            current_key = torch.tensor(entry["key"])
            max_abs_diff = (prior_key - current_key).abs().max().item()
            cross_restart[text] = {
                "max_abs_diff": max_abs_diff,
                "stable": max_abs_diff <= args.tolerance,
            }
        result["cross_restart_manifest_matches"] = manifest_matches
        result["cross_restart"] = cross_restart
        result["phase_0_exit_gate_passed"] = manifest_matches and all(
            entry.get("stable", False) for entry in cross_restart.values()
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    # See imagenet_stream_extract.py's main() for the confirmed root cause
    # (a leaked multiprocessing semaphore inside heavy optional dependencies
    # stalls normal interpreter shutdown) and why both outcomes force-exit.
    args = _parser().parse_args()
    try:
        metrics = run(args)
    except BaseException:
        sys.stdout.flush()
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    summary = {k: v for k, v in metrics.items() if k != "keys"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
