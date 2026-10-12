"""Consolidated real-ImageNet-1K evaluation: top-1/top-5, matched baselines,
routing recall, latency, and memory reported together at scale.

`imagenet_validation_eval.py` reported centroid/k-NN/probe top-1 accuracy.
This closes the remaining "Evaluation at scale" gap in
`imagenet-scale-readiness.md`: top-5 accuracy, the matched-memory VQ and RBF
baselines (not just centroid/k-NN/probe), ManifoldFlow's own exact and
ANN-routed bank accuracy side by side (so routing recall and classification
accuracy are visible together, not in separate reports), and per-method
latency and memory - all against the same real, whitened ImageNet-1K
latents and the same real basin bank, in one report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.ann_benchmark import latency_distribution_ms
from manifoldflow.bank import BasinState
from manifoldflow.baselines import (
    classification_accuracy,
    fit_linear_probe,
    fit_nearest_centroid,
    knn_class_scores,
    linear_probe_predict,
    top_k_accuracy,
)
from manifoldflow.index import ExactIndex, HnswIndex
from manifoldflow.prototype import (
    nearest_prototype_class_scores,
    rbf_prototype_class_scores,
)
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--rbf-top-k", type=int, default=4)
    parser.add_argument("--rbf-temperature", type=float, default=0.1)
    parser.add_argument("--ann-candidates", type=int, default=512)
    parser.add_argument("--ann-ef-search", type=int, default=1024)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser


def _tensor_bytes(*values: Tensor) -> int:
    return sum(value.numel() * value.element_size() for value in values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.eval_samples <= 0 or args.latency_samples <= 0:
        raise ValueError("eval_samples and latency_samples must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)

    payload = torch.load(args.latents, map_location="cpu", weights_only=True)
    train_latent = payload["train_latent"].float().to(device)
    train_label = payload["train_label"].long().to(device)
    test_latent = payload["test_latent"][: args.eval_samples].float().to(device)
    test_label = payload["test_label"][: args.eval_samples].long().to(device)
    latency_query = test_latent[: args.latency_samples]

    bank_payload = torch.load(args.bank, map_location="cpu", weights_only=True)
    bank = BasinState.from_serialized(bank_payload).mu.to(device)
    bank_full = BasinState.from_serialized(bank_payload)
    class_count = bank_payload["class_count"].to(device)
    classes = class_count.shape[1]

    results: dict[str, dict[str, Any]] = {}

    # --- Nearest centroid ---
    centroid = fit_nearest_centroid(train_latent, train_label)
    centroid_scores = -torch.cdist(test_latent, centroid.centers.to(device))
    results["nearest_centroid"] = {
        "top1_accuracy": classification_accuracy(
            centroid_scores.argmax(dim=-1), test_label
        ),
        "top5_accuracy": top_k_accuracy(centroid_scores, test_label, k=min(5, classes)),
        "memory_bytes": _tensor_bytes(centroid.centers),
        "latency_ms": latency_distribution_ms(
            lambda q: -torch.cdist(q, centroid.centers.to(device)), latency_query
        ),
    }

    # --- Exact k-NN ---
    knn_scores = knn_class_scores(
        train_latent, train_label, test_latent, k=args.knn_k, classes=classes
    )
    results["exact_knn"] = {
        "k": args.knn_k,
        "top1_accuracy": classification_accuracy(knn_scores.argmax(dim=-1), test_label),
        "top5_accuracy": top_k_accuracy(knn_scores, test_label, k=min(5, classes)),
        "memory_bytes": _tensor_bytes(train_latent) + train_label.numel() * 8,
        "latency_ms": latency_distribution_ms(
            lambda q: knn_class_scores(
                train_latent, train_label, q, k=args.knn_k, classes=classes
            ),
            latency_query,
        ),
    }

    # --- Linear probe ---
    probe = fit_linear_probe(
        train_latent,
        train_label,
        classes=classes,
        epochs=args.probe_epochs,
        batch_size=256,
        learning_rate=1e-2,
        seed=args.seed,
        device=device,
    ).model
    probe_scores = probe(test_latent)
    probe_bytes = sum(p.numel() * p.element_size() for p in probe.parameters())
    results["linear_probe"] = {
        "top1_accuracy": classification_accuracy(
            linear_probe_predict(probe, test_latent), test_label
        ),
        "top5_accuracy": top_k_accuracy(probe_scores, test_label, k=min(5, classes)),
        "memory_bytes": probe_bytes,
        "latency_ms": latency_distribution_ms(lambda q: probe(q), latency_query),
    }

    # --- ManifoldFlow bank: exact VQ (nearest-basin) and exact RBF ---
    exact_index = ExactIndex(bank)
    vq_scores = nearest_prototype_class_scores(
        test_latent, bank, class_count, index=exact_index
    )
    results["manifoldflow_vq_exact"] = {
        "top1_accuracy": classification_accuracy(vq_scores.argmax(dim=-1), test_label),
        "top5_accuracy": top_k_accuracy(vq_scores, test_label, k=min(5, classes)),
        "memory_bytes": _tensor_bytes(
            bank_full.mu, bank_full.raw_rate, bank_full.m2, class_count
        ),
        "latency_ms": latency_distribution_ms(
            lambda q: nearest_prototype_class_scores(
                q, bank, class_count, index=exact_index
            ),
            latency_query,
        ),
    }
    rbf_scores = rbf_prototype_class_scores(
        test_latent,
        bank,
        class_count,
        top_k=args.rbf_top_k,
        temperature=args.rbf_temperature,
        index=exact_index,
    )
    results["manifoldflow_rbf_exact"] = {
        "top1_accuracy": classification_accuracy(rbf_scores.argmax(dim=-1), test_label),
        "top5_accuracy": top_k_accuracy(rbf_scores, test_label, k=min(5, classes)),
        "memory_bytes": results["manifoldflow_vq_exact"]["memory_bytes"],
        "latency_ms": latency_distribution_ms(
            lambda q: rbf_prototype_class_scores(
                q,
                bank,
                class_count,
                top_k=args.rbf_top_k,
                temperature=args.rbf_temperature,
                index=exact_index,
            ),
            latency_query,
        ),
    }

    # --- ManifoldFlow bank: ANN-routed VQ, with recall against the exact winner ---
    ann_candidates = min(args.ann_candidates, bank.shape[0])
    hnsw = HnswIndex(bank, ef_search=args.ann_ef_search, seed=args.seed)
    exact_winner = exact_index.query(test_latent, 1).squeeze(1)
    ann_candidate_ids = hnsw.query(test_latent, ann_candidates)
    ann_recall = (ann_candidate_ids == exact_winner[:, None]).any(dim=-1).float().mean()
    ann_winner = ann_candidate_ids[:, 0]
    ann_vq_scores = class_count[ann_winner].float()
    results["manifoldflow_vq_ann"] = {
        "candidates": ann_candidates,
        "ef_search": args.ann_ef_search,
        "routing_recall_vs_exact": float(ann_recall.item()),
        "top1_accuracy": classification_accuracy(
            ann_vq_scores.argmax(dim=-1), test_label
        ),
        "top5_accuracy": top_k_accuracy(ann_vq_scores, test_label, k=min(5, classes)),
        "memory_bytes": results["manifoldflow_vq_exact"]["memory_bytes"],
        "latency_ms": latency_distribution_ms(
            lambda q: class_count[hnsw.query(q, ann_candidates)[:, 0]], latency_query
        ),
    }

    metrics = {
        "latents": str(args.latents),
        "bank": str(args.bank),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "seed": args.seed,
        "classes": classes,
        "train_samples": train_latent.shape[0],
        "eval_samples": test_latent.shape[0],
        "latency_samples": latency_query.shape[0],
        "bank_basins": bank.shape[0],
        "results": results,
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
