"""Initialize a populated basin bank from a Phase A latent artifact."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from manifoldflow.baselines import classification_accuracy
from manifoldflow.flow import positive_rate
from manifoldflow.initialization import initialize_basin_bank
from manifoldflow.training import seed_everything

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--basins", required=True, type=int)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mini-batch-steps", type=int)
    parser.add_argument("--refinement-steps", type=int)
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _tensor_bytes(*values: torch.Tensor) -> int:
    return sum(value.numel() * value.element_size() for value in values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build, validate, and serialize one exact-routing basin initialization."""
    config = _read_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    if seed < 0:
        raise ValueError("seed must be non-negative")
    initialization = config["initialization"]
    flow = config["flow"]
    seed_everything(seed)
    device = _device(args.device)

    payload = torch.load(args.latents, map_location="cpu", weights_only=True)
    required = {"train_latent", "train_label", "test_latent", "test_label"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"latent artifact is missing keys: {sorted(missing)}")
    train_latent = payload["train_latent"].float().to(device)
    train_label = payload["train_label"].long().to(device)
    test_latent = payload["test_latent"].float().to(device)
    test_label = payload["test_label"].long().to(device)
    if args.basins > train_latent.shape[0]:
        raise ValueError("basins cannot exceed the number of training latents")

    mini_batch_steps = (
        args.mini_batch_steps
        if args.mini_batch_steps is not None
        else int(initialization["mini_batch_steps"])
    )
    refinement_steps = (
        args.refinement_steps
        if args.refinement_steps is not None
        else int(initialization["refinement_steps"])
    )
    started = time.perf_counter()
    bank = initialize_basin_bank(
        train_latent,
        train_label,
        basins=args.basins,
        classes=args.classes,
        mini_batch_size=int(initialization["mini_batch_size"]),
        mini_batch_steps=mini_batch_steps,
        refinement_steps=refinement_steps,
        seed=seed,
        query_chunk_size=int(initialization["query_chunk_size"]),
        center_chunk_size=int(initialization["center_chunk_size"]),
        variance_scale=None,
        target_median_rate=float(initialization["target_median_rate"]),
        variance_epsilon=float(initialization["variance_epsilon"]),
        rate_min=float(flow["rate_min"]),
        rate_max=float(flow["rate_max"]),
        bias_bound=float(initialization["bias_bound"]),
    )
    elapsed = time.perf_counter() - started

    query_chunk_size = int(initialization["query_chunk_size"])
    center_chunk_size = int(initialization["center_chunk_size"])
    train_prediction = bank.predict_nearest_value(
        train_latent,
        query_chunk_size=query_chunk_size,
        center_chunk_size=center_chunk_size,
    )
    test_prediction = bank.predict_nearest_value(
        test_latent,
        query_chunk_size=query_chunk_size,
        center_chunk_size=center_chunk_size,
    )
    rate = positive_rate(
        bank.state.raw_rate,
        rate_min=float(flow["rate_min"]),
        rate_max=float(flow["rate_max"]),
    )
    configured_rate_max = float(flow["rate_max"])
    core_bytes = _tensor_bytes(
        bank.state.mu,
        bank.state.raw_rate,
        bank.state.bias,
        bank.state.count,
        bank.state.m2,
        bank.state.usage,
        bank.class_count,
    )
    occupancy = bank.state.count.float()
    metrics = {
        "source_latents": str(args.latents),
        "device": str(device),
        "precision": str(train_latent.dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "seed": seed,
        "dimension": train_latent.shape[1],
        "train_samples": train_latent.shape[0],
        "test_samples": test_latent.shape[0],
        "requested_basins": args.basins,
        "active_basins": bank.active_basins,
        "pruned_empty_basins": args.basins - bank.active_basins,
        "mini_batch_steps": mini_batch_steps,
        "mini_batch_size": int(initialization["mini_batch_size"]),
        "refinement_steps": bank.refinement_steps,
        "variance_scale": bank.variance_scale,
        "inertia": bank.inertia,
        "occupancy_min": int(occupancy.min().item()),
        "occupancy_median": float(occupancy.median().item()),
        "occupancy_max": int(occupancy.max().item()),
        "rate_min_observed": float(rate.min().item()),
        "rate_median_observed": float(rate.median().item()),
        "rate_max_observed": float(rate.max().item()),
        "rate_max_saturation_fraction": float(
            (rate >= configured_rate_max - 1e-6).float().mean().item()
        ),
        "nearest_basin_train_accuracy": classification_accuracy(
            train_prediction, train_label
        ),
        "nearest_basin_test_accuracy": classification_accuracy(
            test_prediction, test_label
        ),
        "core_bank_bytes": core_bytes,
        "initialization_seconds": elapsed,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    serialized = bank.serializable()
    serialized["seed"] = seed
    torch.save(serialized, args.output / "bank.pt")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = _parser().parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
