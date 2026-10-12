"""Phase 4: does the adaptive gate beat a fixed alpha? (design-v0.3.md SS9
Phase 4, exit gate: "Better retention/carryover tradeoff than fixed alpha
at matched current-input accuracy.")

Reuses the hard-negative dataset that finally gave Phase 3 real
discriminating power (`qwen_hard_negative_probe.py`'s
`_generate_hard_dataset`: 60 facts as 30 same-entity minimal pairs) and
compares `AdaptiveBasinContext` (`adaptive_context.py`) against the fixed
config that won Phase 3b (`alpha=0.5, decay=0.95`,
`qwen-phase3b-hard-negatives.md`) on two protocols:

1. Retention (Phase 3's own protocol): target fact, N distractors, then the
   target's own paraphrase - does context help recall it?
2. Carryover / current-input responsiveness (SS8.3's "false stale-memory
   carryover" and SS1.2's "retention is valid only while current-input
   responsiveness remains near the stateless baseline"): establish context
   on fact A, then *immediately* (zero distractors) query a paraphrase of a
   completely different fact B. A fixed alpha always blends in some of A's
   stale context even when B confidently conflicts with it; the adaptive
   gate is specifically meant to recognize a confident, conflicting
   observation and raise alpha toward 1 instead. This is the scenario fixed
   alpha cannot handle by construction, so it is the one the exit gate
   actually turns on.
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

from manifoldflow.adaptive_context import AdaptiveBasinContext
from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.experiments.qwen_hard_negative_probe import _generate_hard_dataset
from manifoldflow.persistent_context import PersistentBasinContext
from manifoldflow.qwen_adapter import (
    QwenRepresentationAdapter,
    QwenRepresentationConfig,
)

_FIXED_ALPHA = 0.5
_FIXED_DECAY = 0.95
_DISTRACTOR_COUNTS = [2, 5, 10, 15, 20]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--entity-count", type=int, default=30)
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--sweep-seed", type=int, default=29)
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-k", type=int, default=8)
    return parser


def _calibrate(
    fact_keys: list[Tensor], paraphrase_keys: list[Tensor], sibling_index: list[int]
) -> dict[str, float]:
    """Pick `rbf_bandwidth`/`novelty_threshold` from measured distances,
    the same discipline earlier phases used rather than guessing blind."""
    fact_matrix = torch.stack(fact_keys)
    paraphrase_matrix = torch.stack(paraphrase_keys)
    same_fact_sq_distance = (paraphrase_matrix - fact_matrix).square().sum(dim=-1)
    sibling_matrix = fact_matrix[torch.tensor(sibling_index)]
    sibling_sq_distance = (paraphrase_matrix - sibling_matrix).square().sum(dim=-1)
    # A bandwidth where a typical same-fact match still scores highly but a
    # typical sibling (hardest same-entity distractor) scores clearly lower.
    bandwidth = float(same_fact_sq_distance.mean().sqrt().item()) + 1e-6
    same_fact_score = torch.exp(-same_fact_sq_distance / (2 * bandwidth**2))
    sibling_score = torch.exp(-sibling_sq_distance / (2 * bandwidth**2))
    novelty_threshold = float(
        ((same_fact_score.mean() + sibling_score.mean()) / 2).item()
    )
    return {
        "rbf_bandwidth": bandwidth,
        "novelty_threshold": novelty_threshold,
        "mean_same_fact_score": float(same_fact_score.mean().item()),
        "mean_sibling_score": float(sibling_score.mean().item()),
    }


def _retention_trial(
    context: Any,
    memory: BasinMemory,
    *,
    fact_key: Tensor,
    paraphrase_key: Tensor,
    distractor_keys: list[Tensor],
    target_fact_text: str,
) -> dict[str, float]:
    context.reset()
    visited_pre = {_route_id(context.route(fact_key))}
    for distractor_key in distractor_keys:
        context.route(distractor_key)
    routed = _route_id(context.route(paraphrase_key))
    return {
        "re_association": float(routed in visited_pre),
        "accuracy": float(target_fact_text in memory.payloads_for(routed)),
    }


def _carryover_trial(
    context: Any,
    memory: BasinMemory,
    *,
    anchor_fact_key: Tensor,
    other_paraphrase_key: Tensor,
    other_fact_text: str,
) -> dict[str, float]:
    context.reset()
    context.route(anchor_fact_key)  # establish stale context on a different fact
    routed = _route_id(context.route(other_paraphrase_key))  # immediate topic switch
    correct = other_fact_text in memory.payloads_for(routed)
    return {"current_input_correct": float(correct)}


def _route_id(decision: Any) -> int:
    return decision.routed_id if hasattr(decision, "routed_id") else int(decision)


def run(args: argparse.Namespace) -> dict[str, Any]:
    facts, paraphrases, sibling_index = _generate_hard_dataset(
        args.dataset_seed, args.entity_count
    )
    bank_size = len(facts)

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

    calibration = _calibrate(fact_keys, paraphrase_keys, sibling_index)

    fixed = PersistentBasinContext(
        memory.centers, memory.index, alpha=_FIXED_ALPHA, decay=_FIXED_DECAY
    )
    adaptive = AdaptiveBasinContext(
        memory.centers,
        memory.index,
        alpha_base=_FIXED_ALPHA,
        decay=_FIXED_DECAY,
        rbf_bandwidth=calibration["rbf_bandwidth"],
        novelty_threshold=calibration["novelty_threshold"],
        candidate_k=args.candidate_k,
    )

    generator = random.Random(args.sweep_seed)

    retention: dict[str, dict[str, Any]] = {"fixed": {}, "adaptive": {}}
    for label, context in (("fixed", fixed), ("adaptive", adaptive)):
        for distractor_count in _DISTRACTOR_COUNTS:
            re_assoc_values = []
            accuracy_values = []
            for target_index in range(bank_size):
                other_indices = [i for i in range(bank_size) if i != target_index]
                distractor_indices = generator.sample(other_indices, distractor_count)
                distractor_keys = [paraphrase_keys[i] for i in distractor_indices]
                stats = _retention_trial(
                    context,
                    memory,
                    fact_key=fact_keys[target_index],
                    paraphrase_key=paraphrase_keys[target_index],
                    distractor_keys=distractor_keys,
                    target_fact_text=facts[target_index],
                )
                re_assoc_values.append(stats["re_association"])
                accuracy_values.append(stats["accuracy"])
            retention[label][str(distractor_count)] = {
                "re_association_rate_mean": sum(re_assoc_values) / len(re_assoc_values),
                "accuracy_mean": sum(accuracy_values) / len(accuracy_values),
            }

    carryover: dict[str, float] = {}
    for label, context in (("fixed", fixed), ("adaptive", adaptive)):
        correct_values = []
        for target_index in range(bank_size):
            other_candidates = [
                i
                for i in range(bank_size)
                if i != target_index and i != sibling_index[target_index]
            ]
            other_index = generator.choice(other_candidates)
            stats = _carryover_trial(
                context,
                memory,
                anchor_fact_key=fact_keys[target_index],
                other_paraphrase_key=paraphrase_keys[other_index],
                other_fact_text=facts[other_index],
            )
            correct_values.append(stats["current_input_correct"])
        carryover[label] = sum(correct_values) / len(correct_values)

    retention_not_worse = all(
        retention["adaptive"][str(n)]["accuracy_mean"]
        >= retention["fixed"][str(n)]["accuracy_mean"] - 0.05
        for n in _DISTRACTOR_COUNTS
    )
    carryover_improved = carryover["adaptive"] > carryover["fixed"]

    result = {
        "model_name": args.model_name,
        "bank_size": bank_size,
        "fixed_config": {"alpha": _FIXED_ALPHA, "decay": _FIXED_DECAY},
        "calibration": calibration,
        "retrieval_manifest": adapter.manifest(),
        "retention": retention,
        "carryover_current_input_accuracy": carryover,
        "retention_not_worse_than_fixed": retention_not_worse,
        "carryover_improved_over_fixed": carryover_improved,
        "phase_4_exit_gate_passed": retention_not_worse and carryover_improved,
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
