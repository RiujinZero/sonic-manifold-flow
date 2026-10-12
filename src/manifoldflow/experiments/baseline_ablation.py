"""Compare analytic flow with matched-memory prototype denoisers."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from manifoldflow.baselines import fit_nearest_centroid
from manifoldflow.conditional import (
    calibrate_conditional_gate,
    calibrate_conditional_gate_grid,
    semantic_gate_name,
)
from manifoldflow.data import build_image_dataloaders
from manifoldflow.experiments.flow_common import (
    load_bank,
    load_representation,
    load_valued_bank,
    read_config,
    resolve_device,
)
from manifoldflow.matched_baselines import evaluate_matched_baselines
from manifoldflow.training import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("FashionMNIST", "CIFAR10"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--flow-bank", required=True, type=Path)
    parser.add_argument("--prototype-bank", required=True, type=Path)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--image-noise-std", type=float)
    parser.add_argument("--flow-times", nargs="+", type=float)
    parser.add_argument("--strengths", nargs="+", type=float)
    parser.add_argument("--diagnostic-rbf-strength", type=float)
    parser.add_argument("--semantic-grid", action="store_true")
    parser.add_argument("--semantic-confidence", type=float)
    parser.add_argument("--semantic-distance-quantile", type=float)
    parser.add_argument("--semantic-margin-threshold", type=float)
    parser.add_argument(
        "--semantic-hybrid",
        action="store_true",
        help="Use diagnostic RBF except where each semantic flow gate activates",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    if seed < 0:
        raise ValueError("seed must be non-negative")
    settings = config["matched_baselines"]
    flow = config["flow"]
    seed_everything(seed)
    device = resolve_device(args.device)
    model, normalizer, probe, spec = load_representation(
        args.checkpoint, dataset=args.dataset
    )
    flow_bank, flow_class_probability = load_valued_bank(
        args.flow_bank,
        expected_dimension=model.latent_dim,
        expected_classes=spec.classes,
    )
    assert flow_class_probability is not None
    prototype_bank = load_bank(args.prototype_bank, expected_dimension=model.latent_dim)
    flow_vector_values = flow_bank.mu.numel() + flow_bank.raw_rate.numel()
    prototype_vector_values = prototype_bank.mu.numel()
    if flow_vector_values != prototype_vector_values:
        raise ValueError(
            "prototype center values must match flow center-plus-rate values"
        )

    latent_payload = torch.load(args.latents, map_location="cpu", weights_only=True)
    if not {"train_latent", "train_label"}.issubset(latent_payload):
        raise ValueError("latent artifact must contain train_latent and train_label")
    centroid = fit_nearest_centroid(
        latent_payload["train_latent"].float(),
        latent_payload["train_label"].long(),
    )
    semantic_values = (
        args.semantic_confidence,
        args.semantic_distance_quantile,
        args.semantic_margin_threshold,
    )
    if args.semantic_grid and any(value is not None for value in semantic_values):
        raise ValueError("--semantic-grid cannot be combined with single-gate values")
    if args.semantic_hybrid and not (
        args.semantic_grid or all(value is not None for value in semantic_values)
    ):
        raise ValueError("--semantic-hybrid requires a semantic grid or single gate")
    if not args.semantic_grid and any(value is not None for value in semantic_values):
        if any(value is None for value in semantic_values):
            raise ValueError("all three semantic gate values must be supplied")
        confidence, distance_quantile, semantic_margin = semantic_values
        assert confidence is not None
        assert distance_quantile is not None
        assert semantic_margin is not None
        calibrations = (
            calibrate_conditional_gate(
                latent_payload["train_latent"].float(),
                flow_bank,
                confidence_threshold=confidence,
                distance_quantile=distance_quantile,
                semantic_margin_threshold=semantic_margin,
                candidates=int(flow["candidate_count"]),
                top_k=int(flow["top_k"]),
                flow_time=float(settings["conditional_flow_time"]),
                temperature=float(flow["temperature"]),
                chunk_size=int(settings["gate_calibration_chunk_size"]),
                max_samples=int(settings["gate_calibration_samples"]),
                class_probability=flow_class_probability,
                name=semantic_gate_name(
                    confidence,
                    distance_quantile,
                    semantic_margin,
                    float(settings["conditional_flow_time"]),
                ),
            ),
        )
    elif args.semantic_grid:
        calibrations = calibrate_conditional_gate_grid(
            latent_payload["train_latent"].float(),
            flow_bank,
            flow_class_probability,
            confidence_thresholds=tuple(
                float(value) for value in settings["semantic_confidence_thresholds"]
            ),
            distance_quantiles=tuple(
                float(value) for value in settings["semantic_distance_quantiles"]
            ),
            semantic_margin_thresholds=tuple(
                float(value) for value in settings["semantic_margin_thresholds"]
            ),
            candidates=int(flow["candidate_count"]),
            top_k=int(flow["top_k"]),
            flow_time=float(settings["conditional_flow_time"]),
            temperature=float(flow["temperature"]),
            chunk_size=int(settings["gate_calibration_chunk_size"]),
            max_samples=int(settings["gate_calibration_samples"]),
        )
    else:
        calibrations = (
            calibrate_conditional_gate(
                latent_payload["train_latent"].float(),
                flow_bank,
                confidence_threshold=float(
                    settings["conditional_confidence_threshold"]
                ),
                distance_quantile=float(settings["conditional_distance_quantile"]),
                candidates=int(flow["candidate_count"]),
                top_k=int(flow["top_k"]),
                flow_time=float(settings["conditional_flow_time"]),
                temperature=float(flow["temperature"]),
                chunk_size=int(settings["gate_calibration_chunk_size"]),
                max_samples=int(settings["gate_calibration_samples"]),
            ),
        )
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
    flow_times = (
        tuple(args.flow_times)
        if args.flow_times is not None
        else tuple(float(value) for value in settings["flow_times"])
    )
    strengths = (
        tuple(args.strengths)
        if args.strengths is not None
        else tuple(float(value) for value in settings["strengths"])
    )
    image_noise_std = (
        args.image_noise_std
        if args.image_noise_std is not None
        else float(settings["image_noise_std"])
    )
    started = time.perf_counter()
    ablation = evaluate_matched_baselines(
        model,
        normalizer,
        probe,
        flow_bank,
        prototype_bank.mu,
        centroid.centers,
        test_loader,
        device=device,
        flow_times=flow_times,
        strengths=strengths,
        candidates=int(flow["candidate_count"]),
        flow_top_k=int(flow["top_k"]),
        flow_temperature=float(flow["temperature"]),
        rbf_top_k=int(settings["rbf_top_k"]),
        rbf_temperature=float(settings["rbf_temperature"]),
        flow_class_probability=flow_class_probability,
        conditional_gates=tuple(calibration.spec for calibration in calibrations),
        conditional_flow_time=float(settings["conditional_flow_time"]),
        diagnostic_rbf_strength=(
            args.diagnostic_rbf_strength
            if args.diagnostic_rbf_strength is not None
            else float(settings["diagnostic_rbf_strength"])
        ),
        include_semantic_hybrids=args.semantic_hybrid,
        image_noise_std=image_noise_std,
        skip_samples=args.skip_samples,
        max_samples=max_samples,
        seed=seed,
    )
    metrics = {
        "dataset": spec.name,
        "latent_width": model.latent_dim,
        "basins": flow_bank.mu.shape[0],
        "prototype_basins": prototype_bank.mu.shape[0],
        "flow_vector_values": flow_vector_values,
        "prototype_vector_values": prototype_vector_values,
        "class_centroids": centroid.centers.shape[0],
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
        "skip_samples": args.skip_samples,
        "candidate_count": int(flow["candidate_count"]),
        "soft_top_k": int(flow["top_k"]),
        "temperature": float(flow["temperature"]),
        "times": list(flow_times),
        "strengths": list(strengths),
        "rbf_top_k": int(settings["rbf_top_k"]),
        "rbf_temperature": float(settings["rbf_temperature"]),
        "conditional_gate_mode": (
            "semantic_grid"
            if args.semantic_grid
            else (
                "semantic_single"
                if any(value is not None for value in semantic_values)
                else "legacy"
            )
        ),
        "semantic_hybrid": args.semantic_hybrid,
        "conditional_gate_calibrations": [
            {
                "name": calibration.name,
                "confidence_threshold": calibration.confidence_threshold,
                "distance_quantile": calibration.distance_quantile,
                "distance_threshold": calibration.distance_threshold,
                "semantic_margin_threshold": calibration.semantic_margin_threshold,
                "samples": calibration.samples,
                "clean_confident_fraction": calibration.confident_fraction,
                "clean_semantic_eligible_fraction": (
                    calibration.semantic_eligible_fraction
                ),
                "clean_activation_fraction": calibration.activation_fraction,
            }
            for calibration in calibrations
        ],
        "conditional_calibration_samples": calibrations[0].samples,
        "conditional_flow_time": float(settings["conditional_flow_time"]),
        "diagnostic_rbf_strength": (
            args.diagnostic_rbf_strength
            if args.diagnostic_rbf_strength is not None
            else float(settings["diagnostic_rbf_strength"])
        ),
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
