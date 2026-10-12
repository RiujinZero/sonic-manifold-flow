"""Phase 6: does consolidation scale, and does it preserve correctness and
foreground availability? (design-v0.3.md SS9 Phase 6, exit gate:
"Consolidation preserves evaluation accuracy within tolerance and
foreground reads remain available.")

Synthetic, not real-Qwen-encoded: at 10K-100K rows, encoding real text
through the model would cost hours (Phase 1-5's ~30s-per-60-facts rate
extrapolates to ~85 minutes at 10K and ~14 hours at 100K), infeasible for
what this benchmark actually needs to measure - the *systems* behavior of
`prune_expired_basins`/`merge_compatible_recent_basins`/`consolidate`, not
representation quality (already validated separately, Phase 1-5). This
mirrors the project's own established precedent
(`ann_scale_sweep.py`/`ann-routing-scale.md`): representation correctness
is tested small and real; scale is tested large and synthetic.

Builds the bank via `BasinMemory.from_bulk` (vectorized, not a `write()`
loop - see that method's docstring for why an uninterrupted `write()` loop
degrades toward O(n^2) at real scale), ages a fraction of basins into
staleness, and deliberately injects near-duplicate pairs so merge has
genuine work to do rather than measuring an empty no-op on data too
high-dimensional for random points to ever collide.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from manifoldflow.continuous_memory import BasinMemory, Source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scales", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--near_duplicate_fraction", type=float, default=0.05)
    parser.add_argument("--stale_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def _build_bank(
    scale: int, dimension: int, near_duplicate_fraction: float, seed: int
) -> BasinMemory:
    generator = torch.Generator().manual_seed(seed)
    base_count = int(scale * (1 - near_duplicate_fraction))
    duplicate_count = scale - base_count
    base_centers = F.normalize(
        torch.randn(base_count, dimension, generator=generator), dim=-1
    )
    # Near-duplicates: a small perturbation of a randomly chosen base
    # center, so merge has real close pairs to find - pure random points
    # on a high-dimensional sphere essentially never collide by chance.
    duplicate_sources = base_centers[
        torch.randint(0, base_count, (duplicate_count,), generator=generator)
    ]
    noise = 0.01 * torch.randn(duplicate_count, dimension, generator=generator)
    duplicate_centers = F.normalize(duplicate_sources + noise, dim=-1)
    centers = torch.cat([base_centers, duplicate_centers])
    payloads = [f"synthetic fact {i}" for i in range(centers.shape[0])]
    return BasinMemory.from_bulk(
        centers,
        payloads,
        Source("imported"),
        novelty_threshold=0.0,
        promotion_threshold=2,
        microbasin_max_age=1_000_000_000,
        merge_threshold=0.05,
    )


def _run_scale(
    scale: int, args: argparse.Namespace
) -> dict[str, Any]:
    memory = _build_bank(scale, args.dimension, args.near_duplicate_fraction, args.seed)
    initial_basin_count = memory.basin_count

    # Age a fraction of basins into staleness by setting their
    # last-touched step directly - equivalent in effect to a real stream
    # where those basins simply never got reinforced again. from_bulk
    # assigns last_touched_step sequentially by creation order, so every
    # *other* basin's step must also be pulled forward to "now" first -
    # otherwise a tight microbasin_max_age stales everything except the
    # very last basin created, not just the intended fraction.
    stale_count = int(initial_basin_count * args.stale_fraction)
    stale_ids = list(range(stale_count))
    stale_id_set = set(stale_ids)
    for basin_id in range(initial_basin_count):
        memory._last_touched_step[basin_id] = (
            0 if basin_id in stale_id_set else memory._step
        )
    memory.microbasin_max_age = 1

    # --- Foreground availability: a read right before consolidation ---
    probe_center = memory.centers[initial_basin_count - 1]
    pre_hits = memory.read(probe_center, top_k=1)
    expected_payload = [f"synthetic fact {initial_basin_count - 1}"]
    pre_correct = bool(pre_hits and pre_hits[0].payloads == expected_payload)

    started = time.perf_counter()
    expired = memory.expire_stale_microbasins()
    expire_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prune_mapping = memory.prune_expired_basins()
    prune_seconds = time.perf_counter() - started

    started = time.perf_counter()
    merge_mapping = memory.merge_compatible_recent_basins(0.05)
    merge_seconds = time.perf_counter() - started

    # --- Foreground availability: a read right after consolidation ---
    post_started = time.perf_counter()
    post_hits = memory.read(memory.centers[0], top_k=1)
    post_read_seconds = time.perf_counter() - post_started
    post_available = len(post_hits) == 1

    # --- Correctness: surviving, non-merged basins retrieve correctly ---
    # merge_mapping has an entry for every surviving basin, merged or not
    # (`_rebuild_from_groups` populates singleton groups too) - only new
    # ids that more than one old id maps to were an actual merge. Its keys
    # are in *post-prune* numbering, so original ids must be converted via
    # prune_mapping before checking membership - comparing raw original
    # ids against post-prune ids silently mismatches almost everything.
    new_id_counts = Counter(merge_mapping.values())
    actually_merged_post_prune_ids = {
        old_id
        for old_id, new_id in merge_mapping.items()
        if new_id_counts[new_id] > 1
    }
    surviving_original_ids = [
        i
        for i in range(initial_basin_count)
        if i not in stale_id_set
        and prune_mapping.get(i, i) not in actually_merged_post_prune_ids
    ]
    sample_size = min(200, len(surviving_original_ids))
    sample = torch.tensor(surviving_original_ids)[
        torch.randperm(len(surviving_original_ids))[:sample_size]
    ]
    correct = 0
    for original_id in sample.tolist():
        # merge operates on post-prune numbering, so both maps chain.
        after_prune = prune_mapping.get(original_id, original_id)
        after_merge = merge_mapping.get(after_prune, after_prune)
        query = memory.centers[after_merge]
        hits = memory.read(query, top_k=1)
        if hits and hits[0].payloads == [f"synthetic fact {original_id}"]:
            correct += 1
    accuracy = correct / sample_size if sample_size else 1.0

    merge_groups_combined = sum(1 for count in new_id_counts.values() if count > 1)

    return {
        "scale": scale,
        "initial_basin_count": initial_basin_count,
        "expired_count": len(expired),
        "final_basin_count": memory.basin_count,
        "merge_groups_combined": merge_groups_combined,
        "expire_seconds": expire_seconds,
        "prune_seconds": prune_seconds,
        "merge_seconds": merge_seconds,
        "foreground_read_available_pre_consolidation": pre_correct,
        "foreground_read_available_post_consolidation": post_available,
        "post_consolidation_read_seconds": post_read_seconds,
        "surviving_retrieval_accuracy_sample": accuracy,
        "surviving_retrieval_sample_size": sample_size,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = [_run_scale(scale, args) for scale in args.scales]
    exit_gate_passed = all(
        r["surviving_retrieval_accuracy_sample"] >= 0.99
        and r["foreground_read_available_pre_consolidation"]
        and r["foreground_read_available_post_consolidation"]
        for r in results
    )
    output = {"results": results, "phase_6_exit_gate_passed": exit_gate_passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    args = _parser().parse_args()
    try:
        metrics = run(args)
    except BaseException:
        sys.stdout.flush()
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
