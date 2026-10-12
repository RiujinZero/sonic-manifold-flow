"""Phase 2 for Sonic: does the static basin bank retrieve payloads, and
does writing paraphrases of already-known facts grow the bank in a
controlled way? (design-v0.3.md SS9 Phase 2, exit gate: "The bank
retrieves correct payloads without temporal context and does not show
uncontrolled basin growth.")

Mirrors `qwen_static_bank_probe.py` exactly against
`SonicRepresentationAdapter` (Phase 1's selected 75%-depth mean-pooling
default) instead, reusing the identical Phase 1/2 fact/paraphrase dataset
so results are directly comparable to Qwen's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.experiments.qwen_representation_sweep import _generate_dataset
from manifoldflow.sonic_adapter import (
    SonicRepresentationAdapter,
    SonicRepresentationConfig,
)

_DEFAULT_CHECKPOINT = Path("checkpoints/sonic_ssm/sonic_ssm_128k_steps.pt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--pairs-per-family", type=int, default=8)
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--device", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    facts, paraphrases = _generate_dataset(args.dataset_seed, args.pairs_per_family)
    bank_size = len(facts)

    adapter = SonicRepresentationAdapter(
        SonicRepresentationConfig(
            checkpoint_path=args.checkpoint_path, device=args.device
        )
    )
    fact_keys = torch.stack([adapter.encode_event(text) for text in facts])
    paraphrase_keys = torch.stack(
        [adapter.encode_event(text) for text in paraphrases]
    )

    # --- Check 1: retrieval through the real write/read path ---
    retrieval_memory = BasinMemory(
        dimension=adapter.config.output_dim, novelty_threshold=0.0
    )
    for key, text in zip(fact_keys, facts, strict=True):
        retrieval_memory.write(key, text, Source("user"))
    if retrieval_memory.basin_count != bank_size:
        raise RuntimeError(
            f"expected one basin per fact ({bank_size}) with novelty_threshold=0, "
            f"got {retrieval_memory.basin_count}"
        )

    hits = 0
    per_query: dict[str, Any] = {}
    paired = zip(paraphrases, paraphrase_keys, facts, strict=True)
    for paraphrase, key, true_fact in paired:
        results = retrieval_memory.read(key, top_k=1)
        top_payloads = results[0].payloads if results else []
        matched = true_fact in top_payloads
        hits += matched
        per_query[paraphrase] = {"true_fact": true_fact, "matched": matched}
    retrieval_accuracy = hits / bank_size

    # --- Check 2: same-fact vs cross-fact distance gap for growth control ---
    same_fact_sq_distance = (paraphrase_keys - fact_keys).square().sum(dim=-1)
    cross_min_sq_distance = []
    for index in range(bank_size):
        others = torch.cat([fact_keys[:index], fact_keys[index + 1 :]])
        distances = (others - paraphrase_keys[index]).square().sum(dim=-1)
        cross_min_sq_distance.append(distances.min().item())
    cross_min_sq_distance_tensor = torch.tensor(cross_min_sq_distance)

    max_same_fact = float(same_fact_sq_distance.max().item())
    min_cross_fact = float(cross_min_sq_distance_tensor.min().item())
    clean_threshold_exists = max_same_fact < min_cross_fact
    growth_threshold = (
        (max_same_fact + min_cross_fact) / 2 if clean_threshold_exists else None
    )

    growth_result: dict[str, Any] = {
        "max_same_fact_squared_distance": max_same_fact,
        "min_cross_fact_squared_distance": min_cross_fact,
        "clean_separating_threshold_exists": clean_threshold_exists,
        "chosen_threshold": growth_threshold,
    }
    if clean_threshold_exists:
        assert growth_threshold is not None
        growth_memory = BasinMemory(
            dimension=adapter.config.output_dim, novelty_threshold=growth_threshold
        )
        for key, text in zip(fact_keys, facts, strict=True):
            growth_memory.write(key, text, Source("user"))
        basins_after_facts = growth_memory.basin_count
        for key, text in zip(paraphrase_keys, paraphrases, strict=True):
            growth_memory.write(key, text, Source("user"))
        basins_after_paraphrases = growth_memory.basin_count
        growth_result.update(
            {
                "basins_after_facts": basins_after_facts,
                "basins_after_paraphrases": basins_after_paraphrases,
                "controlled_growth": basins_after_paraphrases == basins_after_facts,
            }
        )

    result = {
        "bank_size": bank_size,
        "checkpoint_path": str(args.checkpoint_path),
        "retrieval_manifest": adapter.manifest(),
        "retrieval_accuracy_top1": retrieval_accuracy,
        "retrieval_per_query": per_query,
        "growth_control": growth_result,
        "phase_2_exit_gate_passed": (
            retrieval_accuracy == 1.0
            and growth_result.get("controlled_growth", False) is True
        ),
    }

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
    summary = {k: v for k, v in metrics.items() if k != "retrieval_per_query"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
