"""Validation-only operating-point selection for robustness experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_GROUPS = {
    "semantic_hybrid": "semantic_hybrid_",
    "semantic_flow": "semantic_flow_",
    "conditional_flow": "conditional_flow_",
    "flow": "flow_",
    "prototype": "prototype_",
    "rbf": "rbf_",
    "class_centroid": "class_centroid_",
    "vq": "vq_",
}


def _mean_metric(
    summary: Mapping[str, Any], variant: str, condition: str, metric: str
) -> float:
    try:
        value = summary[variant][condition][metric]["mean"]
    except (KeyError, TypeError) as error:
        raise ValueError("aggregate summary has an invalid metric structure") from error
    if not isinstance(value, (int, float)):
        raise ValueError("aggregate metric means must be numeric")
    return float(value)


def select_operating_points(
    aggregate: Mapping[str, Any],
    *,
    max_clean_mse_relative_increase: float = 0.05,
    max_clean_accuracy_drop: float = 0.005,
) -> dict[str, Any]:
    """Select each method family's lowest-noisy-MSE feasible variant.

    Feasibility is determined only by clean-quality constraints relative to
    `t0`. Ties favor higher noisy accuracy and then lower clean MSE.
    """
    if max_clean_mse_relative_increase < 0 or max_clean_accuracy_drop < 0:
        raise ValueError("selection constraints must be non-negative")
    summary = aggregate.get("summary")
    if not isinstance(summary, Mapping) or "t0" not in summary:
        raise ValueError("aggregate must contain a t0 summary")
    baseline_clean_mse = _mean_metric(summary, "t0", "clean", "reconstruction_mse")
    baseline_clean_accuracy = _mean_metric(
        summary, "t0", "clean", "classification_accuracy"
    )
    baseline_noisy_mse = _mean_metric(summary, "t0", "noisy", "reconstruction_mse")
    baseline_noisy_accuracy = _mean_metric(
        summary, "t0", "noisy", "classification_accuracy"
    )
    result: dict[str, Any] = {}
    for group, prefix in _GROUPS.items():
        feasible = []
        for variant in summary:
            if not variant.startswith(prefix):
                continue
            clean_mse = _mean_metric(summary, variant, "clean", "reconstruction_mse")
            clean_accuracy = _mean_metric(
                summary, variant, "clean", "classification_accuracy"
            )
            if clean_mse > baseline_clean_mse * (1 + max_clean_mse_relative_increase):
                continue
            if clean_accuracy < baseline_clean_accuracy - max_clean_accuracy_drop:
                continue
            noisy_mse = _mean_metric(summary, variant, "noisy", "reconstruction_mse")
            noisy_accuracy = _mean_metric(
                summary, variant, "noisy", "classification_accuracy"
            )
            feasible.append(
                (noisy_mse, -noisy_accuracy, clean_mse, variant, clean_accuracy)
            )
        if not feasible:
            result[group] = None
            continue
        noisy_mse, negative_noisy_accuracy, clean_mse, variant, clean_accuracy = min(
            feasible
        )
        noisy_accuracy = -negative_noisy_accuracy
        result[group] = {
            "variant": variant,
            "clean_reconstruction_mse": clean_mse,
            "clean_classification_accuracy": clean_accuracy,
            "noisy_reconstruction_mse": noisy_mse,
            "noisy_classification_accuracy": noisy_accuracy,
            "clean_mse_relative_change": clean_mse / baseline_clean_mse - 1,
            "clean_accuracy_change": clean_accuracy - baseline_clean_accuracy,
            "noisy_mse_relative_change": noisy_mse / baseline_noisy_mse - 1,
            "noisy_accuracy_change": noisy_accuracy - baseline_noisy_accuracy,
        }
    return {
        "validation_seeds": aggregate.get("seeds"),
        "constraints": {
            "max_clean_mse_relative_increase": max_clean_mse_relative_increase,
            "max_clean_accuracy_drop": max_clean_accuracy_drop,
        },
        "baseline": {
            "clean_reconstruction_mse": baseline_clean_mse,
            "clean_classification_accuracy": baseline_clean_accuracy,
            "noisy_reconstruction_mse": baseline_noisy_mse,
            "noisy_classification_accuracy": baseline_noisy_accuracy,
        },
        "selected": result,
    }
