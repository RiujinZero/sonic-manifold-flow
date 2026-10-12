"""Calibrate a frozen-backbone decoder for normalized and flowed latents."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from manifoldflow.calibration import calibrate_decoder
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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    if seed < 0:
        raise ValueError("seed must be non-negative")
    settings = config["decoder_calibration"]
    seed_everything(seed)
    device = resolve_device(args.device)
    model, normalizer, probe, spec = load_representation(
        args.checkpoint, dataset=args.dataset
    )
    bank = load_bank(args.bank, expected_dimension=model.latent_dim)
    train_loader, _, _ = build_image_dataloaders(
        args.dataset,
        root=args.data_root,
        batch_size=int(settings["batch_size"]),
        num_workers=0 if args.smoke else int(settings["num_workers"]),
        download=args.download,
        seed=seed,
    )
    epochs = (
        1
        if args.smoke
        else (args.epochs if args.epochs is not None else int(settings["epochs"]))
    )
    max_batches = 2 if args.smoke else args.max_batches
    started = time.perf_counter()
    result = calibrate_decoder(
        model,
        normalizer,
        bank,
        train_loader,
        device=device,
        epochs=epochs,
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        flow_time=float(settings["flow_time"]),
        latent_noise_std=float(settings["latent_noise_std"]),
        candidates=int(config["flow"]["candidate_count"]),
        normalized_weight=float(settings["normalized_weight"]),
        flowed_weight=float(settings["flowed_weight"]),
        max_batches=max_batches,
    )
    elapsed = time.perf_counter() - started
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
        "epochs": epochs,
        "batch_size": int(settings["batch_size"]),
        "flow_time": float(settings["flow_time"]),
        "latent_noise_std": float(settings["latent_noise_std"]),
        "normalized_weight": float(settings["normalized_weight"]),
        "flowed_weight": float(settings["flowed_weight"]),
        "total_epoch_mse": list(result.total_loss),
        "normalized_epoch_mse": list(result.normalized_loss),
        "flowed_epoch_mse": list(result.flowed_loss),
        "elapsed_seconds": elapsed,
        "smoke": args.smoke,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "dataset": spec.name,
            "latent_width": model.latent_dim,
            "seed": seed,
            "model": model.cpu().state_dict(),
            "normalizer": normalizer.cpu().state_dict(),
            "linear_probe": probe.cpu().state_dict(),
            "config": config,
            "source_checkpoint": str(args.checkpoint),
            "source_bank": str(args.bank),
        },
        args.output / "checkpoint.pt",
    )
    return metrics


def main() -> None:
    metrics = run(_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
