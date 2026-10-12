"""Probe whether persistent basin context survives distraction.

Tests the mechanism `persistent_context.PersistentBasinContext` adds: does a
decaying trace of recently visited basin centers let a later, related
observation re-associate with an earlier one, despite unrelated observations
in between? Approximates the "truck reappears after 15 minutes" scenario
with real (unseen-by-the-bank) ImageNet-1K validation latents, using class
identity as a stand-in for instance identity, which ImageNet does not label:
a stream of `pre_count` images from one target class, then `distractor_count`
images from other, distinct classes, then `post_count` images from the
target class again. `alpha=1.0` is the stateless control - exactly today's
routing, with no context influence - so every other configuration is judged
against it, not against an assumed baseline.

Two signals are reported per configuration, averaged over many sampled
target classes: re-association rate (does a post-distraction query land on
one of the *same* basins the pre-distraction visits landed on - an
instance-level proxy, meaningful here because real occupancy is often
near-singleton) and class accuracy (does the routed basin's stored class
distribution still predict the right class - the standard, coarser signal).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.ann_benchmark import latency_distribution_ms
from manifoldflow.bank import BasinState
from manifoldflow.index import ExactIndex, HnswIndex
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.persistent_context import PersistentBasinContext
from manifoldflow.sharded_latents import load_sharded_latents
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument("--val-shard-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-classes", type=int, default=30)
    parser.add_argument("--pre-count", type=int, default=3)
    parser.add_argument("--post-count", type=int, default=3)
    parser.add_argument(
        "--distractor-counts", type=int, nargs="+", default=[5, 20, 50]
    )
    parser.add_argument(
        "--configs",
        action="append",
        dest="configs",
        metavar="ALPHA:DECAY",
        default=None,
        help="Repeatable alpha:decay pair. Defaults to a small fixed grid "
        "plus the alpha=1.0 stateless control.",
    )
    parser.add_argument("--min-examples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def _parse_config(spec: str) -> tuple[float, float]:
    alpha_text, decay_text = spec.split(":")
    return float(alpha_text), float(decay_text)


def _group_by_class(
    label: Tensor, *, min_examples: int, seed: int
) -> dict[int, list[int]]:
    generator = random.Random(seed)
    groups: dict[int, list[int]] = {}
    for index, value in enumerate(label.tolist()):
        groups.setdefault(value, []).append(index)
    eligible = {
        class_id: indices
        for class_id, indices in groups.items()
        if len(indices) >= min_examples
    }
    for indices in eligible.values():
        generator.shuffle(indices)
    return eligible


def _run_stream(
    context: PersistentBasinContext,
    latent: Tensor,
    *,
    target_class: int,
    target_indices: list[int],
    distractor_indices: list[int],
    pre_count: int,
    post_count: int,
    class_count: Tensor,
) -> dict[str, float]:
    context.reset()
    pre_ids = target_indices[:pre_count]
    post_ids = target_indices[pre_count : pre_count + post_count]

    visited_pre = set()
    for row in pre_ids:
        visited_pre.add(context.route(latent[row]))
    for row in distractor_indices:
        context.route(latent[row])

    re_associated = 0
    correct = 0
    for row in post_ids:
        routed = context.route(latent[row])
        if routed in visited_pre:
            re_associated += 1
        predicted_class = int(class_count[routed].argmax().item())
        if predicted_class == target_class:
            correct += 1

    return {
        "re_association_rate": re_associated / len(post_ids),
        "class_accuracy": correct / len(post_ids),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_classes <= 0 or args.pre_count <= 0 or args.post_count <= 0:
        raise ValueError("target_classes, pre_count, and post_count must be positive")
    if not args.distractor_counts or min(args.distractor_counts) < 0:
        raise ValueError("distractor_counts must be non-empty and non-negative")
    seed_everything(args.seed)

    bank_payload = torch.load(args.bank, map_location="cpu", weights_only=True)
    bank = BasinState.from_serialized(bank_payload)
    class_count = bank_payload["class_count"]
    normalizer_state = torch.load(
        args.normalizer, map_location="cpu", weights_only=True
    )
    normalizer = FrozenDiagonalWhitening(bank.mu.shape[1])
    normalizer.load_state_dict(normalizer_state)

    raw_latent, label = load_sharded_latents(args.val_shard_dir)
    latent = normalizer(raw_latent.float())

    groups = _group_by_class(
        label, min_examples=args.min_examples_per_class, seed=args.seed
    )
    max_distractors = max(args.distractor_counts)
    if len(groups) < args.target_classes + max_distractors:
        raise ValueError(
            f"only {len(groups)} classes have >= {args.min_examples_per_class} "
            f"examples; need at least target_classes + max(distractor_counts) "
            f"= {args.target_classes + max_distractors} distinct classes so "
            "every distractor is drawn from a class the query never targets"
        )
    generator = random.Random(args.seed)
    all_classes = sorted(groups)
    target_classes = generator.sample(all_classes, args.target_classes)

    configs = (
        [_parse_config(spec) for spec in args.configs]
        if args.configs
        else [(1.0, 0.0), (0.7, 0.9), (0.5, 0.9), (0.5, 0.95), (0.3, 0.95)]
    )

    exact_index = ExactIndex(bank.mu)
    sweep: dict[str, dict[str, Any]] = {}
    for alpha, decay in configs:
        context = PersistentBasinContext(bank.mu, exact_index, alpha=alpha, decay=decay)
        per_distractor: dict[str, Any] = {}
        for distractor_count in args.distractor_counts:
            re_association_rates = []
            class_accuracies = []
            for target_class in target_classes:
                target_indices = groups[target_class][
                    : args.pre_count + args.post_count
                ]
                distractor_classes = generator.sample(
                    [c for c in all_classes if c != target_class],
                    distractor_count,
                )
                distractor_pool = [groups[c][0] for c in distractor_classes]
                stats = _run_stream(
                    context,
                    latent,
                    target_class=target_class,
                    target_indices=target_indices,
                    distractor_indices=distractor_pool,
                    pre_count=args.pre_count,
                    post_count=args.post_count,
                    class_count=class_count,
                )
                re_association_rates.append(stats["re_association_rate"])
                class_accuracies.append(stats["class_accuracy"])
            tensor_re = torch.tensor(re_association_rates)
            tensor_acc = torch.tensor(class_accuracies)
            per_distractor[str(distractor_count)] = {
                "re_association_rate_mean": float(tensor_re.mean().item()),
                "re_association_rate_std": float(tensor_re.std().item()),
                "class_accuracy_mean": float(tensor_acc.mean().item()),
                "class_accuracy_std": float(tensor_acc.std().item()),
            }
        sweep[f"alpha{alpha:g}_decay{decay:g}"] = {
            "alpha": alpha,
            "decay": decay,
            "by_distractor_count": per_distractor,
        }

    # Confirm the mechanism stays ANN-routing-cheap, per the design's own
    # bounded-per-query-compute requirement, using the same HNSW settings
    # validated in imagenet-scale-evaluation.md.
    hnsw = HnswIndex(bank.mu, ef_search=512, seed=args.seed)
    ann_context = PersistentBasinContext(bank.mu, hnsw, alpha=0.5, decay=0.9)
    latency_query = latent[:200]

    def _timed_route(batch: Tensor) -> None:
        ann_context.route(batch.squeeze(0))

    latency = latency_distribution_ms(_timed_route, latency_query)

    metrics = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "seed": args.seed,
        "bank_basins": bank.mu.shape[0],
        "target_classes_sampled": args.target_classes,
        "pre_count": args.pre_count,
        "post_count": args.post_count,
        "distractor_counts": args.distractor_counts,
        "min_examples_per_class": args.min_examples_per_class,
        "eligible_classes": len(groups),
        "sweep": sweep,
        "ann_routing_latency_ms": latency,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    args = _parser().parse_args()
    started = time.perf_counter()
    metrics = run(args)
    metrics["wall_seconds"] = time.perf_counter() - started
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
