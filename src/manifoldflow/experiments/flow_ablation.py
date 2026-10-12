"""Evaluate matched no-flow, hard-flow, and soft-flow image ablations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from manifoldflow.calibration import evaluate_flow_ablation
from manifoldflow.data import build_image_dataloaders
from manifoldflow.experiments.flow_common import (
    load_bank,
    load_representation,
    read_config,
    resolve_device,
)
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("FashionMNIST", "CIFAR10"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--image-noise-std", type=float)
    parser.add_argument("--times", nargs="+", type=float)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    if seed < 0:
        raise ValueError("seed must be non-negative")
    settings = config["flow_ablation"]
    seed_everything(seed)
    device = resolve_device(args.device)
    model, normalizer, probe, spec = load_representation(
        args.checkpoint, dataset=args.dataset
    )
    bank = load_bank(args.bank, expected_dimension=model.latent_dim)
    _, test_loader, _ = build_image_dataloaders(
        args.dataset,
        root=args.data_root,
        batch_size=int(settings["batch_size"]),
        num_workers=0 if args.smoke else int(settings["num_workers"]),
        download=args.download,
        seed=seed,
    )
    max_samples = (
        128
        if args.smoke
        else (
            args.max_samples
            if args.max_samples is not None
            else int(settings["max_samples"])
        )
    )
    times = (
        tuple(args.times)
        if args.times is not None
        else tuple(float(value) for value in settings["times"])
    )
    image_noise_std = (
        args.image_noise_std
        if args.image_noise_std is not None
        else float(settings["image_noise_std"])
    )
    started = time.perf_counter()
    ablation = evaluate_flow_ablation(
        model,
        normalizer,
        probe,
        bank,
        test_loader,
        device=device,
        times=times,
        candidates=int(config["flow"]["candidate_count"]),
        soft_top_k=int(config["flow"]["top_k"]),
        temperature=float(config["flow"]["temperature"]),
        image_noise_std=image_noise_std,
        max_samples=max_samples,
        seed=seed,
    )
    metrics = {
        "dataset": spec.name,
        "latent_width": model.latent_dim,
        "basins": bank.mu.shape[0],
        "seed": seed,
        "device": str(device),
        "precision": str(next(model.parameters()).dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "batch_size": int(settings["batch_size"]),
        "eval_samples": max_samples,
        "candidate_count": int(config["flow"]["candidate_count"]),
        "soft_top_k": int(config["flow"]["top_k"]),
        "temperature": float(config["flow"]["temperature"]),
        "times": list(times),
        "image_noise_std": image_noise_std,
        "ablation": ablation,
        "elapsed_seconds": time.perf_counter() - started,
        "smoke": args.smoke,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    metrics = run(_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
