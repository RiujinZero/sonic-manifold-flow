"""Evaluate local basin adaptation on an ordered non-IID latent stream."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from manifoldflow.bank import BasinState
from manifoldflow.continual import ContinualStreamSpec, evaluate_continual_latent_stream
from manifoldflow.experiments.flow_common import read_config
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--prototype-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stream-samples-per-class", type=int)
    parser.add_argument("--shift-scale", type=float)
    parser.add_argument("--latent-noise-std", type=float)
    parser.add_argument("--novelty-quantile", type=float)
    parser.add_argument("--novelty-persistence", type=int)
    parser.add_argument("--novelty-support-multiplier", type=float)
    parser.add_argument("--basin-growth-budget", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    seed_everything(seed)
    settings = config["continual"]
    learning = config["learning"]
    flow = config["flow"]
    latent_payload = torch.load(args.latents, map_location="cpu", weights_only=True)
    required_latent = {"train_latent", "train_label", "test_latent", "test_label"}
    missing = required_latent.difference(latent_payload)
    if missing:
        raise ValueError(f"latent artifact is missing keys: {sorted(missing)}")
    bank_payload = torch.load(args.bank, map_location="cpu", weights_only=True)
    if not isinstance(bank_payload, dict):
        raise ValueError("bank artifact must contain a mapping")
    bank = BasinState.from_serialized(bank_payload)
    class_count = bank_payload.get("class_count")
    if not isinstance(class_count, torch.Tensor):
        raise ValueError("bank artifact must contain class_count")
    variance_scale = float(bank_payload.get("variance_scale", 1.0))
    prototype_payload = torch.load(
        args.prototype_bank, map_location="cpu", weights_only=True
    )
    if not isinstance(prototype_payload, dict):
        raise ValueError("prototype bank artifact must contain a mapping")
    prototype_bank = BasinState.from_serialized(prototype_payload)
    prototype_class_count = prototype_payload.get("class_count")
    if not isinstance(prototype_class_count, torch.Tensor):
        raise ValueError("prototype bank artifact must contain class_count")
    samples_per_class = (
        10
        if args.smoke
        else (
            args.stream_samples_per_class
            if args.stream_samples_per_class is not None
            else int(settings["stream_samples_per_class"])
        )
    )
    spec = ContinualStreamSpec(
        classes_per_task=int(settings["classes_per_task"]),
        stream_samples_per_class=samples_per_class,
        shift_scale=(
            args.shift_scale
            if args.shift_scale is not None
            else float(settings["shift_scale"])
        ),
        latent_noise_std=(
            args.latent_noise_std
            if args.latent_noise_std is not None
            else float(settings["latent_noise_std"])
        ),
        novelty_quantile=(
            args.novelty_quantile
            if args.novelty_quantile is not None
            else float(learning["novelty_distance_quantile"])
        ),
        novelty_persistence=(
            args.novelty_persistence
            if args.novelty_persistence is not None
            else int(learning["novelty_persistence"])
        ),
        novelty_support_multiplier=(
            args.novelty_support_multiplier
            if args.novelty_support_multiplier is not None
            else float(settings["novelty_support_multiplier"])
        ),
        basin_growth_budget=(
            args.basin_growth_budget
            if args.basin_growth_budget is not None
            else int(settings["basin_growth_budget"])
        ),
        seed=seed,
    )
    started = time.perf_counter()
    continual = evaluate_continual_latent_stream(
        bank,
        class_count.long(),
        prototype_bank,
        prototype_class_count.long(),
        latent_payload["train_latent"].float(),
        latent_payload["train_label"].long(),
        latent_payload["test_latent"].float(),
        latent_payload["test_label"].long(),
        spec=spec,
        variance_scale=variance_scale,
        variance_epsilon=float(config["initialization"]["variance_epsilon"]),
        rate_min=float(flow["rate_min"]),
        rate_max=float(flow["rate_max"]),
        rbf_top_k=int(config["matched_baselines"]["rbf_top_k"]),
        rbf_temperature=float(config["matched_baselines"]["rbf_temperature"]),
    )
    metrics = {
        "seed": seed,
        "dimension": bank.mu.shape[1],
        "initial_basins": bank.mu.shape[0],
        "prototype_basins": prototype_bank.mu.shape[0],
        "train_samples": latent_payload["train_latent"].shape[0],
        "test_samples": latent_payload["test_latent"].shape[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "elapsed_seconds": time.perf_counter() - started,
        "smoke": args.smoke,
        "continual": continual,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
