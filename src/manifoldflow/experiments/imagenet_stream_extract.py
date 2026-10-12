"""Extract bounded ImageNet-1K train latents via streaming, no raw download.

Full ImageNet-1K train is ~146GB (`imagenet-scale-readiness.md`), more than
this project's environment can safely persist locally. This streams a
bounded number of rows directly from the Hugging Face Hub through the frozen
224px backbone (`backbone.py`) and writes only the resulting compact latent
shards (`sharded_latents.py`) to disk - raw images are never persisted.
Requires an authenticated Hugging Face token with access to the gated
`ILSVRC/imagenet-1k` dataset (`huggingface_stream.py`).

Reruns against the same `--output-dir` resume from the existing manifest's
row count via the stream's own `.skip()`, so an interrupted run does not
re-download already-covered rows before continuing. A large `skip()` against
this gated dataset has been observed to hang indefinitely with no error
(see `huggingface_stream.with_timeout`'s docstring); `--stream-timeout-seconds`
turns a stall into a loud `TimeoutError` instead, and `main()` force-exits
after a successful run since a lingering native background thread has been
observed to prevent clean interpreter shutdown even after all real work
(including writing every output file) is done.
"""

from __future__ import annotations

import argparse
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
from manifoldflow.huggingface_stream import streaming_imagenet_batches, with_timeout
from manifoldflow.sharded_latents import extract_sharded_latents
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--projected-dim", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="train")
    parser.add_argument("--repo-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--stream-timeout-seconds", type=float, default=300.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_samples <= 0 or args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("max_samples, batch_size, and shard_size must be positive")
    if args.calibration_samples <= 0:
        raise ValueError("calibration_samples must be positive")
    token = os.environ.get(args.hf_token_env)
    if not token:
        raise RuntimeError(
            f"set {args.hf_token_env} to a Hugging Face access token with "
            f"access to {args.repo_id!r}"
        )

    seed_everything(args.seed)
    device = torch.device(args.device)
    transform = imagenet_eval_transform()

    backbone = Frozen224Backbone(projected_dim=args.projected_dim, pretrained=True)
    backbone.to(device)

    calibration_parts = []
    seen = 0
    calibration_stream = with_timeout(
        streaming_imagenet_batches(
            split=args.split,
            transform=transform,
            batch_size=args.batch_size,
            token=token,
            take=args.calibration_samples,
            repo_id=args.repo_id,
        ),
        seconds=args.stream_timeout_seconds,
    )
    with torch.no_grad():
        for image, _ in calibration_stream:
            calibration_parts.append(backbone.extract_features(image.to(device)).cpu())
            seen += image.shape[0]
    if not calibration_parts:
        raise ValueError("calibration stream produced no rows")
    calibration = torch.cat(calibration_parts)[: args.calibration_samples]
    backbone.fit_projection(calibration.to(device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    resume_skip = 0
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resume_skip = int(prior_manifest["total_rows"])
    remaining = args.max_samples - resume_skip

    started = time.perf_counter()
    if remaining > 0:
        batches = with_timeout(
            streaming_imagenet_batches(
                split=args.split,
                transform=transform,
                batch_size=args.batch_size,
                token=token,
                skip=resume_skip,
                take=remaining,
                repo_id=args.repo_id,
            ),
            seconds=args.stream_timeout_seconds,
        )
        manifest = extract_sharded_latents(
            backbone,
            batches,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            device=device,
        )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - started

    metrics = {
        "repo_id": args.repo_id,
        "split": args.split,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "seed": args.seed,
        "projected_dim": args.projected_dim,
        "requested_max_samples": args.max_samples,
        "resumed_from_rows": resume_skip,
        "manifest": manifest,
        "extraction_seconds": elapsed,
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
    # Root cause (confirmed): `datasets`/`pyarrow` streaming has leaked a
    # multiprocessing semaphore in this environment (visible as a
    # `resource_tracker: ... leaked semaphore` warning), which stalls
    # Python's normal interpreter shutdown - a separate OS process, not a
    # Python thread, so it never showed up in `threading.enumerate()`.
    # `os._exit` bypasses that cleanup entirely via the OS `_exit` syscall.
    # This must cover both outcomes: an exception path that only prints a
    # traceback and returns still hangs identically to the success path.
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
