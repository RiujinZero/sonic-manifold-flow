"""Calibrate one decoder with balanced flow, prototype, and RBF exposure."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from manifoldflow.calibration import calibrate_neutral_decoder
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
    parser.add_argument("--flow-bank", required=True, type=Path)
    parser.add_argument("--prototype-bank", required=True, type=Path)
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
    settings = config["neutral_calibration"]
    flow = config["flow"]
    matched = config["matched_baselines"]
    seed_everything(seed)
    device = resolve_device(args.device)
    model, normalizer, probe, spec = load_representation(
        args.checkpoint, dataset=args.dataset
    )
    flow_bank = load_bank(args.flow_bank, expected_dimension=model.latent_dim)
    prototype_bank = load_bank(args.prototype_bank, expected_dimension=model.latent_dim)
    flow_vector_values = flow_bank.mu.numel() + flow_bank.raw_rate.numel()
    prototype_vector_values = prototype_bank.mu.numel()
    if flow_vector_values != prototype_vector_values:
        raise ValueError(
            "prototype center values must match flow center-plus-rate values"
        )
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
    flow_times = tuple(float(value) for value in settings["flow_times"])
    strengths = tuple(float(value) for value in settings["prototype_strengths"])
    branch_count = len(flow_times) + 2 * len(strengths)
    max_batches = branch_count if args.smoke else args.max_batches
    started = time.perf_counter()
    result = calibrate_neutral_decoder(
        model,
        normalizer,
        flow_bank,
        prototype_bank.mu,
        train_loader,
        device=device,
        epochs=epochs,
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        latent_noise_std=float(settings["latent_noise_std"]),
        flow_times=flow_times,
        prototype_strengths=strengths,
        candidates=int(flow["candidate_count"]),
        flow_top_k=int(flow["top_k"]),
        flow_temperature=float(flow["temperature"]),
        rbf_top_k=int(matched["rbf_top_k"]),
        rbf_temperature=float(matched["rbf_temperature"]),
        normalized_weight=float(settings["normalized_weight"]),
        transformed_weight=float(settings["transformed_weight"]),
        max_batches=max_batches,
    )
    metrics = {
        "dataset": spec.name,
        "latent_width": model.latent_dim,
        "flow_basins": flow_bank.mu.shape[0],
        "prototype_basins": prototype_bank.mu.shape[0],
        "flow_vector_values": flow_vector_values,
        "prototype_vector_values": prototype_vector_values,
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
        "flow_times": list(flow_times),
        "prototype_strengths": list(strengths),
        "total_epoch_mse": list(result.total_loss),
        "normalized_epoch_mse": list(result.normalized_loss),
        "transformed_epoch_mse": list(result.transformed_loss),
        "branch_epoch_mse": {
            name: list(values) for name, values in result.branch_loss.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
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
            "source_flow_bank": str(args.flow_bank),
            "source_prototype_bank": str(args.prototype_bank),
        },
        args.output / "checkpoint.pt",
    )
    return metrics


def main() -> None:
    metrics = run(_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
