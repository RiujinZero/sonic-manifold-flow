"""Phase 3, retried with a dataset built to actually be confusable.

`qwen_persistent_context_probe.py` found every config - including the
`alpha=1.0` stateless control - scoring a perfect 1.000/1.000 at every
distraction depth (`docs/results/qwen-phase3-persistent-context.md`): the
Phase 1/2 dataset's distinct-entity facts are too cleanly separable to ever
create routing ambiguity, so there was no floor for persistence to lift
above. That script and its results doc are left as-is (already-published,
reproducible evidence of a real null result); this is a new script with a
genuinely different dataset, not a mutation of the old one.

The fix follows SS8.2's own "same-type hard negatives" episode family
literally: instead of `entity_count` facts about `entity_count` distinct
entities (trivially separable by entity name alone), generate
`entity_count` *pairs* of facts about the *same* entity, differing in only
one word (`"{entity} owns the {object_a}."` vs
`"{entity} owns the {object_b}."`) - a minimal pair that forces retrieval
to actually distinguish on the object, not the entity name. Before running
the (cheap, pure-tensor-math) persistence sweep, this script first checks
the *stateless* retrieval floor on the new dataset and refuses to proceed
if it is still at ceiling, so a second uninformative sweep is not spent
without knowing it in advance.
"""

from __future__ import annotations

import argparse
import itertools
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
from manifoldflow.persistent_context import PersistentBasinContext
from manifoldflow.qwen_adapter import (
    QwenRepresentationAdapter,
    QwenRepresentationConfig,
)
from manifoldflow.representation import retrieval_topk_accuracy

_SYLLABLES = [
    "Ne", "Ral", "Vek", "Tor", "Bri", "Dun", "Sol", "Kai", "Mir", "Zan",
    "Ash", "Ely", "Orn", "Ust", "Iva", "Cor", "Dre", "Fen", "Gol", "Hun",
]
_OBJECTS = [
    "copper glider", "striped lantern", "brass compass", "wooden crate",
    "silver whistle", "folded map", "iron key", "glass vial",
    "bronze medallion", "leather satchel",
]
_DEFAULT_CONFIGS = [(1.0, 0.0), (0.7, 0.95), (0.6, 0.95), (0.5, 0.95), (0.5, 0.90)]
_DEFAULT_DISTRACTOR_COUNTS = [2, 5, 10, 15, 20]


def _generate_entity_names(count: int, seed: int) -> list[str]:
    generator = random.Random(seed)
    combos = list(itertools.permutations(_SYLLABLES, 2))
    generator.shuffle(combos)
    if count > len(combos):
        raise ValueError(f"cannot generate {count} distinct entity names")
    return [f"{a}{b.lower()}" for a, b in combos[:count]]


def _generate_hard_dataset(
    seed: int, entity_count: int
) -> tuple[list[str], list[str], list[int]]:
    """Return `(facts, paraphrases, sibling_index)`: `entity_count` pairs of
    facts about the *same* entity, differing by one word (the object).
    `sibling_index[i]` is the index of fact `i`'s same-entity hard negative -
    the fact retrieval is most likely to be confused with."""
    generator = random.Random(seed)
    entities = _generate_entity_names(entity_count, seed)
    facts: list[str] = []
    paraphrases: list[str] = []
    sibling_index: list[int] = []
    for entity in entities:
        object_a, object_b = generator.sample(_OBJECTS, 2)
        base = len(facts)
        facts.append(f"{entity} owns the {object_a}.")
        paraphrases.append(f"The {object_a} belongs to {entity}.")
        facts.append(f"{entity} owns the {object_b}.")
        paraphrases.append(f"The {object_b} belongs to {entity}.")
        sibling_index.extend([base + 1, base])
    return facts, paraphrases, sibling_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--entity-count", type=int, default=30)
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
    parser.add_argument(
        "--skip-floor-check",
        action="store_true",
        help="run the sweep even if stateless top-1 retrieval is still 100%%",
    )
    return parser


def _parse_config(spec: str) -> tuple[float, float]:
    alpha_text, decay_text = spec.split(":")
    return float(alpha_text), float(decay_text)


def _run_stream(
    context: PersistentBasinContext,
    memory: BasinMemory,
    *,
    fact_key: Tensor,
    paraphrase_key: Tensor,
    distractor_keys: list[Tensor],
    target_fact_text: str,
    sibling_fact_text: str,
) -> dict[str, float]:
    context.reset()
    visited_pre = {context.route(fact_key)}
    for distractor_key in distractor_keys:
        context.route(distractor_key)
    routed = context.route(paraphrase_key)
    payloads = memory.payloads_for(routed)
    return {
        "re_association": float(routed in visited_pre),
        "accuracy": float(target_fact_text in payloads),
        "confused_with_sibling": float(sibling_fact_text in payloads),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    facts, paraphrases, sibling_index = _generate_hard_dataset(
        args.dataset_seed, args.entity_count
    )
    bank_size = len(facts)
    max_distractors = max(args.distractor_counts)
    if max_distractors > bank_size - 1:
        raise ValueError(
            f"max distractor count {max_distractors} needs {max_distractors + 1} "
            f"facts but only {bank_size} are available"
        )

    adapter = QwenRepresentationAdapter(
        QwenRepresentationConfig(model_name=args.model_name, device=args.device)
    )
    fact_keys = [adapter.encode_event(text) for text in facts]
    paraphrase_keys = [adapter.encode_event(text) for text in paraphrases]

    # --- Stateless floor check, before spending a full sweep ---
    fact_key_matrix = torch.stack(fact_keys)
    paraphrase_key_matrix = torch.stack(paraphrase_keys)
    true_index = torch.arange(bank_size)
    stateless_top1 = retrieval_topk_accuracy(
        paraphrase_key_matrix, fact_key_matrix, true_index, k=1
    )
    sibling_confusion_rate = 0.0
    if stateless_top1 < 1.0:
        scores = paraphrase_key_matrix @ fact_key_matrix.transpose(0, 1)
        nearest = scores.argmax(dim=1)
        misses = nearest != true_index
        sibling_tensor = torch.tensor(sibling_index)
        confused_with_sibling = misses & (nearest == sibling_tensor)
        miss_count = max(1, int(misses.sum().item()))
        sibling_confusion_rate = float(
            confused_with_sibling.float().sum().item() / miss_count
        )

    result: dict[str, Any] = {
        "model_name": args.model_name,
        "bank_size": bank_size,
        "stateless_top1_retrieval_accuracy": stateless_top1,
        "sibling_confusion_rate_among_misses": sibling_confusion_rate,
        "distractor_counts": args.distractor_counts,
        "retrieval_manifest": adapter.manifest(),
    }

    if stateless_top1 >= 1.0 and not args.skip_floor_check:
        result["sweep_skipped_reason"] = (
            "stateless top-1 retrieval is still 100% on this dataset; a "
            "persistence sweep here would repeat the same uninformative "
            "null result. Increase --entity-count or make objects/entities "
            "more similar before retrying."
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

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
            sibling_confusion_values = []
            for target_index in range(bank_size):
                other_indices = [i for i in range(bank_size) if i != target_index]
                distractor_indices = generator.sample(other_indices, distractor_count)
                distractor_keys = [paraphrase_keys[i] for i in distractor_indices]
                stats = _run_stream(
                    context,
                    memory,
                    fact_key=fact_keys[target_index],
                    paraphrase_key=paraphrase_keys[target_index],
                    distractor_keys=distractor_keys,
                    target_fact_text=facts[target_index],
                    sibling_fact_text=facts[sibling_index[target_index]],
                )
                re_association_values.append(stats["re_association"])
                accuracy_values.append(stats["accuracy"])
                sibling_confusion_values.append(stats["confused_with_sibling"])
            by_distractor_count[str(distractor_count)] = {
                "re_association_rate_mean": _mean(re_association_values),
                "accuracy_mean": _mean(accuracy_values),
                "sibling_confusion_rate": _mean(sibling_confusion_values),
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
    stateless_re_assoc = {
        count: entry["re_association_rate_mean"]
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
        max_accuracy_gain = max(
            acc - stateless_accuracy[str(c)]
            for acc, c in zip(accuracy_by_count, counts, strict=True)
        )
        collapse = min(re_assoc_by_count) > 0.9 and max_accuracy_gain < 0.02
        graded = (
            re_assoc_by_count[0] > stateless_re_assoc[str(counts[0])] + 0.05
            and re_assoc_by_count[0] > re_assoc_by_count[-1]
            and not collapse
        )
        collapse_detected[name] = collapse
        graded_curve_detected[name] = graded

    result.update(
        {
            "sweep": sweep,
            "collapse_detected": collapse_detected,
            "graded_curve_detected": graded_curve_detected,
            "phase_3_exit_gate_passed": any(graded_curve_detected.values()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


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
