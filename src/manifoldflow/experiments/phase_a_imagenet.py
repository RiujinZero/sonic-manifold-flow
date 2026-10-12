"""Frozen 224px backbone entry point for ImageFolder-style large-image streams.

Mirrors `phase_a.py`'s reproducible-artifact pattern for the ImageNet-shaped
entry path described in `imagenet-scale-readiness.md`: a frozen
classification backbone with a fixed PCA projection instead of a trained
paired autoencoder, ImageFolder-style ingestion instead of a built-in
torchvision dataset, and sharded/restartable latent extraction instead of one
monolithic tensor. This proves the ingest-encode-evaluate path; it does not
build or route a basin bank (`initialize_bank.py`/`ann_benchmark.py` already
accept any `{train,test}_latent/label` artifact shaped like this script's
output).
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

from manifoldflow.backbone import Frozen224Backbone, imagenet_eval_transform
from manifoldflow.baselines import (
    classification_accuracy,
    exact_knn_predict,
    fit_linear_probe,
    fit_nearest_centroid,
    linear_probe_predict,
)
from manifoldflow.data import build_imagefolder_dataloaders
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.sharded_latents import extract_sharded_latents, load_sharded_latents
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--projected-dim", type=int, default=256)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/phase_a_imagenet")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet-pretrained backbone weights (default).",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Use a randomly initialized backbone (network-free smoke runs).",
    )
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@torch.no_grad()
def _collect_calibration_features(
    backbone: Frozen224Backbone,
    loader: Any,
    *,
    device: torch.device,
    samples: int,
) -> torch.Tensor:
    parts = []
    seen = 0
    for image, _ in loader:
        if seen >= samples:
            break
        image = image[: samples - seen].to(device)
        parts.append(backbone.extract_features(image).cpu())
        seen += image.shape[0]
    if not parts:
        raise ValueError("calibration loader produced no samples")
    return torch.cat(parts)[:samples]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Extract, normalize, and evaluate frozen 224px latents end to end."""
    if args.projected_dim <= 0 or args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("projected_dim, batch_size, and shard_size must be positive")
    seed_everything(args.seed)
    device = _device(args.device)

    transform = imagenet_eval_transform()
    train_loader, val_loader, classes = build_imagefolder_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        transform=transform,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    backbone = Frozen224Backbone(
        projected_dim=args.projected_dim, pretrained=args.pretrained
    )
    backbone.to(device)
    calibration = _collect_calibration_features(
        backbone, train_loader, device=device, samples=args.calibration_samples
    )
    backbone.fit_projection(calibration.to(device))

    output = args.output_root / f"d{args.projected_dim}" / f"seed{args.seed}"
    started = time.perf_counter()
    train_manifest = extract_sharded_latents(
        backbone,
        train_loader,
        output_dir=output / "train_shards",
        shard_size=args.shard_size,
        device=device,
    )
    val_manifest = extract_sharded_latents(
        backbone,
        val_loader,
        output_dir=output / "val_shards",
        shard_size=args.shard_size,
        device=device,
    )
    extraction_seconds = time.perf_counter() - started

    train_latent, train_label = load_sharded_latents(output / "train_shards")
    val_latent, val_label = load_sharded_latents(output / "val_shards")
    if args.max_eval_samples is not None:
        val_latent = val_latent[: args.max_eval_samples]
        val_label = val_label[: args.max_eval_samples]

    normalizer = FrozenDiagonalWhitening(args.projected_dim)
    normalizer.fit(train_latent)
    normalized_train = normalizer(train_latent)
    normalized_val = normalizer(val_latent)

    centroid = fit_nearest_centroid(normalized_train, train_label)
    centroid_accuracy = classification_accuracy(
        centroid.predict(normalized_val), val_label
    )

    knn_k = min(args.knn_k, normalized_train.shape[0])
    knn_prediction = exact_knn_predict(
        normalized_train,
        train_label,
        normalized_val,
        k=knn_k,
        query_chunk_size=512,
        reference_chunk_size=8192,
    )
    knn_accuracy = classification_accuracy(knn_prediction, val_label)

    probe = fit_linear_probe(
        normalized_train,
        train_label,
        classes=classes,
        epochs=args.probe_epochs,
        batch_size=256,
        learning_rate=1e-2,
        seed=args.seed,
        device=device,
    )
    probe_accuracy = classification_accuracy(
        linear_probe_predict(probe.model, normalized_val), val_label
    )

    elapsed = time.perf_counter() - started
    metrics = {
        "data_root": str(args.data_root),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "seed": args.seed,
        "pretrained": args.pretrained,
        "classes": classes,
        "feature_dim": backbone.feature_dim,
        "projected_dim": args.projected_dim,
        "train_shard_manifest": train_manifest,
        "val_shard_manifest": val_manifest,
        "train_samples": normalized_train.shape[0],
        "eval_samples": normalized_val.shape[0],
        "extraction_seconds": extraction_seconds,
        "nearest_centroid_accuracy": centroid_accuracy,
        "exact_knn_accuracy": knn_accuracy,
        "exact_knn_k": knn_k,
        "linear_probe_accuracy": probe_accuracy,
        "linear_probe_epoch_loss": list(probe.loss),
        "elapsed_seconds": elapsed,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "normalizer": normalizer.state_dict(),
            "linear_probe": probe.model.cpu().state_dict(),
        },
        output / "checkpoint.pt",
    )
    return metrics


def main() -> None:
    args = _parser().parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
