"""Evaluate the streamed ImageNet-1K train latents against real validation data.

`imagenet_stream_extract.py` streams and encodes a bounded ImageNet-1K train
subset without persisting raw images. This script encodes the (separately
downloaded, local) validation parquet shards through the *same* fitted
backbone checkpoint - not a freshly re-fit one, so train and validation
latents are directly comparable - and reports the same matched baselines
`phase_a.py` reports for the small-image MVP (nearest centroid, exact k-NN,
linear probe), giving the first real-data accuracy numbers for the frozen
224px entry path rather than just a plumbing smoke test.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time
import traceback
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
from manifoldflow.huggingface_stream import batches_from_rows
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.sharded_latents import extract_sharded_latents, load_sharded_latents
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard-dir", required=True, type=Path)
    parser.add_argument("--validation-parquet-glob", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_eval_samples <= 0 or args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError(
            "max_eval_samples, batch_size, and shard_size must be positive"
        )
    try:
        import datasets
    except ImportError as error:
        raise RuntimeError(
            "Local parquet ingestion requires the `datasets` package: "
            "python -m pip install manifold-flow[stream]"
        ) from error

    seed_everything(args.seed)
    device = torch.device(args.device)

    checkpoint = torch.load(
        args.train_shard_dir / "checkpoint.pt", map_location=device, weights_only=True
    )
    backbone = Frozen224Backbone(
        projected_dim=int(checkpoint["projected_dim"]), pretrained=False
    )
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.to(device)
    if not bool(backbone.is_fitted):
        raise RuntimeError("loaded checkpoint's projection was never fitted")

    files = sorted(glob.glob(args.validation_parquet_glob))
    if not files:
        raise FileNotFoundError(
            f"no parquet files matched {args.validation_parquet_glob!r}"
        )
    validation_rows = datasets.load_dataset("parquet", data_files=files, split="train")

    val_batches = batches_from_rows(
        validation_rows, transform=imagenet_eval_transform(), batch_size=args.batch_size
    )
    started = time.perf_counter()
    val_manifest = extract_sharded_latents(
        backbone,
        val_batches,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        device=device,
    )
    extraction_seconds = time.perf_counter() - started

    train_latent, train_label = load_sharded_latents(args.train_shard_dir)
    val_latent, val_label = load_sharded_latents(args.output_dir)
    val_latent = val_latent[: args.max_eval_samples]
    val_label = val_label[: args.max_eval_samples]

    normalizer = FrozenDiagonalWhitening(train_latent.shape[1])
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
        query_chunk_size=256,
        reference_chunk_size=8192,
    )
    knn_accuracy = classification_accuracy(knn_prediction, val_label)

    classes = int(train_label.max().item()) + 1
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

    metrics = {
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "seed": args.seed,
        "classes": classes,
        "train_samples": normalized_train.shape[0],
        "eval_samples": normalized_val.shape[0],
        "validation_extraction_seconds": extraction_seconds,
        "validation_manifest": val_manifest,
        "nearest_centroid_accuracy": centroid_accuracy,
        "exact_knn_accuracy": knn_accuracy,
        "exact_knn_k": knn_k,
        "linear_probe_accuracy": probe_accuracy,
        "linear_probe_epoch_loss": list(probe.loss),
    }
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    # See imagenet_stream_extract.py's main() for the confirmed root cause
    # (a leaked multiprocessing semaphore inside datasets/pyarrow streaming
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
