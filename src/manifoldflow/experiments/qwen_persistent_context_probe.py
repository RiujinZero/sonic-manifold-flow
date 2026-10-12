"""Phase 3: does the vision-domain graded forgetting curve reproduce in
language? (design-v0.3.md SS9 Phase 3, exit gate: "A graded forgetting
curve appears above the stateless floor without context collapse.")

Directly mirrors `basin_traversal_probe.py`'s protocol - the vision-domain
probe this phase is meant to reproduce - adapted to language: write all 24
facts (Phase 1's dataset) into a static `BasinMemory` bank (`alpha=1`, one
basin per fact by construction), then for each target fact, each
`(alpha, decay)` from SS6.1's recommended starting range, and each
distractor count, route the fact's own key through
`persistent_context.PersistentBasinContext` (unmodified, reused as-is -
already modality-independent since it only takes centers and an index),
route N *other* facts' paraphrase keys as distraction, then route the
target's own paraphrase and check whether context measurably biases it
back toward the target's basin - re-association (same basin id) and
accuracy (does the routed basin's stored payload match the target fact).

Routing is pure tensor math over already-encoded keys, not further model
calls, so the entire sweep runs on the same 48 forward passes Phase 1/2
already needed - the model is only used once, up front, to encode every
fact and paraphrase.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.experiments.qwen_representation_sweep import _generate_dataset
from manifoldflow.persistent_context import PersistentBasinContext
from manifoldflow.qwen_adapter import (
    QwenRepresentationAdapter,
    QwenRepresentationConfig,
)

_DEFAULT_CONFIGS = [(1.0, 0.0), (0.7, 0.95), (0.6, 0.95), (0.5, 0.95), (0.5, 0.90)]
_DEFAULT_DISTRACTOR_COUNTS = [2, 5, 10, 15, 20]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--pairs-per-family", type=int, default=8)
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--sweep-seed", type=int, default=29)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--distractor-counts", type=int, nargs="+", default=_DEFAULT_DISTRACTOR_COUNTS
    )
    parser.add_argument(
        "--configs",
        action="append",
        dest="configs",
        metavar="ALPHA:DECAY",
        default=None,
    )
    return parser


def _parse_config(spec: str) -> tuple[float, float]:
    alpha_text, decay_text = spec.split(":")
    return float(alpha_text), float(decay_text)


def _run_stream(
    context: PersistentBasinContext,
    memory: BasinMemory,
    *,
    target_index: int,
    fact_key: Tensor,
    paraphrase_key: Tensor,
    distractor_keys: list[Tensor],
    target_fact_text: str,
) -> dict[str, float]:
    context.reset()
    visited_pre = {context.route(fact_key)}
    for distractor_key in distractor_keys:
        context.route(distractor_key)
    routed = context.route(paraphrase_key)
    return {
        "re_association": float(routed in visited_pre),
        "accuracy": float(target_fact_text in memory.payloads_for(routed)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    facts, paraphrases = _generate_dataset(args.dataset_seed, args.pairs_per_family)
    bank_size = len(facts)
    max_distractors = max(args.distractor_counts)
    if max_distractors > bank_size - 1:
        raise ValueError(
            f"max distractor count {max_distractors} needs {max_distractors + 1} "
            f"facts (one target plus that many distractors) but only "
            f"{bank_size} are available"
        )

    adapter = QwenRepresentationAdapter(
        QwenRepresentationConfig(model_name=args.model_name, device=args.device)
    )
    fact_keys = [adapter.encode_event(text) for text in facts]
    paraphrase_keys = [adapter.encode_event(text) for text in paraphrases]

    memory = BasinMemory(dimension=adapter.config.output_dim, novelty_threshold=0.0)
    for key, text in zip(fact_keys, facts, strict=True):
        memory.write(key, text, Source("user"))
    if memory.basin_count != bank_size:
        raise RuntimeError(
            f"expected one basin per fact ({bank_size}), got {memory.basin_count}"
        )

    configs = (
        [_parse_config(spec) for spec in args.configs]
        if args.configs
        else _DEFAULT_CONFIGS
    )
    generator = random.Random(args.sweep_seed)

    sweep: dict[str, Any] = {}
    for alpha, decay in configs:
        context = PersistentBasinContext(
            memory.centers, memory.index, alpha=alpha, decay=decay
        )
        by_distractor_count: dict[str, Any] = {}
        for distractor_count in args.distractor_counts:
            re_association_values = []
            accuracy_values = []
            for target_index in range(bank_size):
                other_indices = [i for i in range(bank_size) if i != target_index]
                distractor_indices = generator.sample(other_indices, distractor_count)
                distractor_keys = [paraphrase_keys[i] for i in distractor_indices]
                stats = _run_stream(
                    context,
                    memory,
                    target_index=target_index,
                    fact_key=fact_keys[target_index],
                    paraphrase_key=paraphrase_keys[target_index],
                    distractor_keys=distractor_keys,
                    target_fact_text=facts[target_index],
                )
                re_association_values.append(stats["re_association"])
                accuracy_values.append(stats["accuracy"])
            re_association_tensor = torch.tensor(re_association_values)
            accuracy_tensor = torch.tensor(accuracy_values)
            by_distractor_count[str(distractor_count)] = {
                "re_association_rate_mean": float(re_association_tensor.mean().item()),
                "accuracy_mean": float(accuracy_tensor.mean().item()),
            }
        sweep[f"alpha{alpha:g}_decay{decay:g}"] = {
            "alpha": alpha,
            "decay": decay,
            "by_distractor_count": by_distractor_count,
        }

    stateless_key = next(name for name, cfg in sweep.items() if cfg["alpha"] == 1.0)
    stateless_accuracy = {
        count: entry["accuracy_mean"]
        for count, entry in sweep[stateless_key]["by_distractor_count"].items()
    }
    collapse_detected: dict[str, bool] = {}
    graded_curve_detected: dict[str, bool] = {}
    for name, cfg in sweep.items():
        if cfg["alpha"] == 1.0:
            continue
        counts = args.distractor_counts
        re_assoc_by_count = [
            cfg["by_distractor_count"][str(c)]["re_association_rate_mean"]
            for c in counts
        ]
        accuracy_by_count = [
            cfg["by_distractor_count"][str(c)]["accuracy_mean"] for c in counts
        ]
        # Context-pinning collapse (SS1.2, SS8.3's formal detector): high,
        # flat re-association with accuracy that never exceeds the
        # stateless floor is the mechanism failing to track content, not
        # succeeding at retention - the same signature found in
        # basin-traversal-persistence-probe.md's alpha=0.3 vision result.
        max_accuracy_gain = max(
            acc - stateless_accuracy[str(c)]
            for acc, c in zip(accuracy_by_count, counts, strict=True)
        )
        collapse = min(re_assoc_by_count) > 0.9 and max_accuracy_gain < 0.02
        # A graded curve: re-association at the smallest distractor count
        # measurably exceeds the stateless floor and decreases (does not
        # stay flat) as distraction accumulates.
        stateless_re_assoc = sweep[stateless_key]["by_distractor_count"][
            str(counts[0])
        ]["re_association_rate_mean"]
        graded = (
            re_assoc_by_count[0] > stateless_re_assoc + 0.05
            and re_assoc_by_count[0] > re_assoc_by_count[-1]
            and not collapse
        )
        collapse_detected[name] = collapse
        graded_curve_detected[name] = graded

    phase_3_exit_gate_passed = any(graded_curve_detected.values())

    result = {
        "model_name": args.model_name,
        "bank_size": bank_size,
        "distractor_counts": args.distractor_counts,
        "retrieval_manifest": adapter.manifest(),
        "sweep": sweep,
        "collapse_detected": collapse_detected,
        "graded_curve_detected": graded_curve_detected,
        "phase_3_exit_gate_passed": phase_3_exit_gate_passed,
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
    print(json.dumps(metrics, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
