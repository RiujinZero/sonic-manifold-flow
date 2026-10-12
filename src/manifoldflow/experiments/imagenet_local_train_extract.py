"""Extract ImageNet-1K train latents from downloaded parquet shards, not streaming.

`imagenet_stream_extract.py` streams train rows directly from the Hub. That
path has proven unreliable for large pulls in practice: resuming via
`skip()` re-reads every already-consumed shard from the start on each call,
and after sustained use this repo's CDN starts returning HTTP 429
(rate-limited) on the very first shard, every time - a `datasets`/`fsspec`
streaming issue, not a resumability issue, since the shard/manifest state
itself always stayed intact.

This instead downloads whole shard files with `huggingface_hub.snapshot_download`
(the same robust, resumable path already used for the validation split) to a
scratch directory, reads them locally and non-streaming - matching
`imagenet_validation_eval.py`'s pattern - and deletes the raw parquet once
latent extraction succeeds, so only the compact shards remain on disk.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from manifoldflow.backbone import Frozen224Backbone, imagenet_eval_transform
from manifoldflow.huggingface_stream import batches_from_rows
from manifoldflow.sharded_latents import extract_sharded_latents
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-cache-dir", required=True, type=Path)
    parser.add_argument("--projected-dim", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repo-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Skip deleting the downloaded parquet shards after extraction.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_samples <= 0 or args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("max_samples, batch_size, and shard_size must be positive")
    if args.calibration_samples <= 0 or args.shard_count <= 0:
        raise ValueError("calibration_samples and shard_count must be positive")
    token = os.environ.get(args.hf_token_env)
    if not token:
        raise RuntimeError(
            f"set {args.hf_token_env} to a Hugging Face access token with "
            f"access to {args.repo_id!r}"
        )
    try:
        import datasets
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "This script requires the `datasets` package: "
            "python -m pip install manifold-flow[stream]"
        ) from error

    seed_everything(args.seed)
    device = torch.device(args.device)
    transform = imagenet_eval_transform()

    shard_names = [
        f"data/train-{index:05d}-of-00294.parquet" for index in range(args.shard_count)
    ]
    download_started = time.perf_counter()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.raw_cache_dir),
        allow_patterns=shard_names,
        token=token,
    )
    download_seconds = time.perf_counter() - download_started

    files = sorted(glob.glob(str(args.raw_cache_dir / "data" / "train-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no train shards downloaded to {args.raw_cache_dir}")
    rows = datasets.load_dataset("parquet", data_files=files, split="train")

    backbone = Frozen224Backbone(projected_dim=args.projected_dim, pretrained=True)
    backbone.to(device)

    calibration_parts = []
    with torch.no_grad():
        for image, _ in batches_from_rows(
            rows.select(range(min(args.calibration_samples, len(rows)))),
            transform=transform,
            batch_size=args.batch_size,
        ):
            calibration_parts.append(backbone.extract_features(image.to(device)).cpu())
    if not calibration_parts:
        raise ValueError("calibration produced no rows")
    calibration = torch.cat(calibration_parts)[: args.calibration_samples]
    backbone.fit_projection(calibration.to(device))

    bounded_rows = rows.select(range(min(args.max_samples, len(rows))))
    train_batches = batches_from_rows(
        bounded_rows, transform=transform, batch_size=args.batch_size
    )
    extraction_started = time.perf_counter()
    manifest = extract_sharded_latents(
        backbone,
        train_batches,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        device=device,
    )
    extraction_seconds = time.perf_counter() - extraction_started

    if not args.keep_raw:
        shutil.rmtree(args.raw_cache_dir, ignore_errors=True)

    metrics = {
        "repo_id": args.repo_id,
        "shard_count_downloaded": args.shard_count,
        "rows_available_in_shards": len(rows),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "seed": args.seed,
        "projected_dim": args.projected_dim,
        "manifest": manifest,
        "download_seconds": download_seconds,
        "extraction_seconds": extraction_seconds,
        "raw_deleted": not args.keep_raw,
    }
    (args.output_dir / "extraction_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {"backbone": backbone.state_dict(), "projected_dim": args.projected_dim},
        args.output_dir / "checkpoint.pt",
    )
    return metrics


def main() -> None:
    # See imagenet_stream_extract.py's main() for why both outcomes force-exit:
    # datasets/pyarrow has been observed to leave a leaked multiprocessing
    # semaphore that stalls normal interpreter shutdown in this environment.
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
