"""Validation and aggregation for repeated flow-ablation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean, stdev
from typing import Any

_MATCHED_FIELDS = (
    "dataset",
    "latent_width",
    "basins",
    "eval_samples",
    "candidate_count",
    "soft_top_k",
    "temperature",
    "times",
    "image_noise_std",
)

_OPTIONAL_MATCHED_FIELDS = (
    "prototype_basins",
    "flow_vector_values",
    "prototype_vector_values",
    "class_centroids",
    "strengths",
    "rbf_top_k",
    "rbf_temperature",
    "skip_samples",
    "conditional_confidence_threshold",
    "conditional_distance_quantile",
    "conditional_calibration_samples",
    "conditional_flow_time",
    "diagnostic_rbf_strength",
    "conditional_gate_mode",
    "semantic_hybrid",
)


def aggregate_flow_metrics(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate compatible ablations with sample standard deviations."""
    if len(runs) < 2:
        raise ValueError("at least two flow-ablation runs are required")
    reference = runs[0]
    for field in _MATCHED_FIELDS:
        if field not in reference:
            raise ValueError(f"run is missing compatibility field {field!r}")
        if any(run.get(field) != reference[field] for run in runs[1:]):
            raise ValueError(f"flow-ablation runs disagree on {field!r}")
    optional_fields = [
        field for field in _OPTIONAL_MATCHED_FIELDS if field in reference
    ]
    for field in optional_fields:
        if any(run.get(field) != reference[field] for run in runs[1:]):
            raise ValueError(f"flow-ablation runs disagree on {field!r}")

    seeds = [run.get("seed") for run in runs]
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("every flow-ablation run must contain an integer seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("flow-ablation seeds must be unique")

    ablations = [run.get("ablation") for run in runs]
    if any(not isinstance(value, Mapping) for value in ablations):
        raise ValueError("every run must contain an ablation mapping")
    reference_ablation = ablations[0]
    variants = set(reference_ablation)
    if any(set(ablation) != variants for ablation in ablations[1:]):
        raise ValueError("flow-ablation runs must contain identical variants")

    summary: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for variant in sorted(variants):
        reference_conditions = reference_ablation[variant]
        if not isinstance(reference_conditions, Mapping):
            raise ValueError("ablation variants must contain condition mappings")
        conditions = set(reference_conditions)
        if any(
            not isinstance(ablation[variant], Mapping)
            or set(ablation[variant]) != conditions
            for ablation in ablations[1:]
        ):
            raise ValueError("flow-ablation variants must contain matched conditions")
        summary[variant] = {}
        for condition in sorted(conditions):
            reference_metrics = reference_conditions[condition]
            if not isinstance(reference_metrics, Mapping):
                raise ValueError("ablation conditions must contain metric mappings")
            metric_names = set(reference_metrics)
            summary[variant][condition] = {}
            for metric_name in sorted(metric_names):
                values = []
                for ablation in ablations:
                    metrics = ablation[variant][condition]
                    if not isinstance(metrics, Mapping) or set(metrics) != metric_names:
                        raise ValueError("flow-ablation metric sets must match")
                    value = metrics[metric_name]
                    if not isinstance(value, (int, float)):
                        raise ValueError("flow-ablation metric values must be numeric")
                    values.append(float(value))
                summary[variant][condition][metric_name] = {
                    "mean": fmean(values),
                    "sample_std": stdev(values),
                    "by_seed": {
                        str(seed): value
                        for seed, value in zip(seeds, values, strict=True)
                    },
                }

    return {
        **{field: reference[field] for field in (*_MATCHED_FIELDS, *optional_fields)},
        "seeds": seeds,
        "run_count": len(runs),
        "summary": summary,
    }
