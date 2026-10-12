"""Train and evaluate the Phase A small-image representation baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from manifoldflow.baselines import (
    classification_accuracy,
    exact_knn_predict,
    fit_linear_probe,
    fit_nearest_centroid,
    linear_probe_predict,
    neighborhood_preservation,
)
from manifoldflow.data import build_image_dataloaders
from manifoldflow.models import CompactImageAutoencoder, trainable_parameter_count
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.training import (
    collect_flattened_inputs,
    collect_latents,
    evaluate_reconstruction,
    seed_everything,
    train_autoencoder,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("FashionMNIST", "CIFAR10"))
    parser.add_argument("--latent-width", required=True, type=int, choices=(64, 128))
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/phase_a"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one dataset/latent-width run and persist reproducible artifacts."""
    config = _read_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    if seed < 0:
        raise ValueError("seed must be non-negative")
    seed_everything(seed)
    device = _device(args.device)
    training = config["training"]
    baselines = config["baselines"]

    epochs = 1 if args.smoke else int(training["epochs"])
    max_batches = 2 if args.smoke else None
    max_train_samples = 128 if args.smoke else args.max_train_samples
    max_eval_samples = 128 if args.smoke else args.max_eval_samples
    for name, value in (
        ("max_train_samples", max_train_samples),
        ("max_eval_samples", max_eval_samples),
    ):
        if value is not None and value < 2:
            raise ValueError(f"{name} must be at least two")

    train_loader, test_loader, spec = build_image_dataloaders(
        args.dataset,
        root=args.data_root,
        batch_size=int(training["batch_size"]),
        num_workers=0 if args.smoke else int(training["num_workers"]),
        download=args.download,
        seed=seed,
    )
    model = CompactImageAutoencoder(
        input_channels=spec.channels,
        input_size=spec.image_size,
        latent_dim=args.latent_width,
    )
    parameter_count = trainable_parameter_count(model)

    started = time.perf_counter()
    autoencoder_loss = train_autoencoder(
        model,
        train_loader,
        device=device,
        epochs=epochs,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_batches=max_batches,
    )
    no_flow_mse = evaluate_reconstruction(
        model,
        test_loader,
        device=device,
        max_samples=max_eval_samples,
    )
    train_latent = collect_latents(
        model,
        train_loader,
        device=device,
        max_samples=max_train_samples,
    )
    test_latent = collect_latents(
        model,
        test_loader,
        device=device,
        max_samples=max_eval_samples,
    )
    model.freeze()

    normalizer = FrozenDiagonalWhitening(args.latent_width)
    normalizer.fit(train_latent.latent)
    normalized_train = normalizer(train_latent.latent)
    normalized_test = normalizer(test_latent.latent)

    neighborhood_samples = min(
        int(baselines["neighborhood_samples"]), normalized_test.shape[0]
    )
    neighborhood_input = collect_flattened_inputs(
        test_loader, max_samples=neighborhood_samples
    )
    neighborhood_k = min(int(baselines["neighborhood_k"]), neighborhood_samples - 1)
    neighbor_overlap = neighborhood_preservation(
        neighborhood_input,
        normalized_test[:neighborhood_samples],
        k=neighborhood_k,
    )

    centroid = fit_nearest_centroid(normalized_train, train_latent.label)
    centroid_prediction = centroid.predict(normalized_test)
    centroid_accuracy = classification_accuracy(centroid_prediction, test_latent.label)

    knn_k = min(int(baselines["knn_k"]), normalized_train.shape[0])
    knn_prediction = exact_knn_predict(
        normalized_train,
        train_latent.label,
        normalized_test,
        k=knn_k,
        query_chunk_size=int(baselines["knn_query_chunk_size"]),
        reference_chunk_size=int(baselines["knn_reference_chunk_size"]),
    )
    knn_accuracy = classification_accuracy(knn_prediction, test_latent.label)

    probe_epochs = 2 if args.smoke else int(baselines["linear_probe_epochs"])
    probe = fit_linear_probe(
        normalized_train,
        train_latent.label,
        classes=spec.classes,
        epochs=probe_epochs,
        batch_size=int(baselines["linear_probe_batch_size"]),
        learning_rate=float(baselines["linear_probe_learning_rate"]),
        seed=seed,
        device=device,
    )
    probe_prediction = linear_probe_predict(probe.model, normalized_test)
    probe_accuracy = classification_accuracy(probe_prediction, test_latent.label)

    elapsed = time.perf_counter() - started
    output = args.output_root / spec.name / f"d{args.latent_width}" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": spec.name,
        "latent_width": args.latent_width,
        "seed": seed,
        "device": str(device),
        "precision": str(next(model.parameters()).dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "epochs": epochs,
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "num_workers": 0 if args.smoke else int(training["num_workers"]),
        "train_samples": normalized_train.shape[0],
        "eval_samples": normalized_test.shape[0],
        "trainable_parameters": parameter_count,
        "autoencoder_epoch_mse": list(autoencoder_loss),
        "no_flow_reconstruction_mse": no_flow_mse,
        "no_flow_reconstruction_psnr_db": 10.0
        * math.log10(1.0 / max(no_flow_mse, 1e-12)),
        "input_latent_neighbor_overlap": neighbor_overlap,
        "neighborhood_samples": neighborhood_samples,
        "neighborhood_k": neighborhood_k,
        "nearest_centroid_accuracy": centroid_accuracy,
        "exact_knn_accuracy": knn_accuracy,
        "exact_knn_k": knn_k,
        "linear_probe_accuracy": probe_accuracy,
        "linear_probe_epoch_loss": list(probe.loss),
        "elapsed_seconds": elapsed,
        "smoke": args.smoke,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "dataset": spec.name,
            "latent_width": args.latent_width,
            "seed": seed,
            "model": model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "linear_probe": probe.model.cpu().state_dict(),
            "config": config,
        },
        output / "checkpoint.pt",
    )
    torch.save(
        {
            "train_latent": normalized_train,
            "train_label": train_latent.label,
            "test_latent": normalized_test,
            "test_label": test_latent.label,
        },
        output / "latents.pt",
    )
    return metrics


def main() -> None:
    args = _parser().parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
