"""Compatibility validation and aggregation for continual-stream runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean, stdev
from typing import Any

_TOP_LEVEL_FIELDS = (
    "dimension",
    "initial_basins",
    "prototype_basins",
    "train_samples",
    "test_samples",
)

_SPEC_FIELDS = (
    "classes_per_task",
    "stream_samples_per_class",
    "shift_scale",
    "latent_noise_std",
    "novelty_quantile",
    "novelty_persistence",
    "novelty_support_multiplier",
    "basin_growth_budget",
    "rbf_top_k",
    "rbf_temperature",
    "manifold_vector_values",
    "prototype_vector_values",
)


def _require_mapping(value: object, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def aggregate_continual_metrics(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate compatible continual runs with means and sample deviations."""
    if len(runs) < 2:
        raise ValueError("at least two continual runs are required")
    reference = runs[0]
    for field in _TOP_LEVEL_FIELDS:
        if field not in reference:
            raise ValueError(f"run is missing compatibility field {field!r}")
        if any(run.get(field) != reference[field] for run in runs[1:]):
            raise ValueError(f"continual runs disagree on {field!r}")
    seeds = [run.get("seed") for run in runs]
    if any(not isinstance(seed, int) for seed in seeds) or len(set(seeds)) != len(
        seeds
    ):
        raise ValueError("continual runs require unique integer seeds")
    continual = [
        _require_mapping(run.get("continual"), "run is missing continual metrics")
        for run in runs
    ]
    reference_spec = _require_mapping(
        continual[0].get("spec"), "continual metrics are missing spec"
    )
    for field in _SPEC_FIELDS:
        if field not in reference_spec:
            raise ValueError(f"continual spec is missing {field!r}")
        for value in continual[1:]:
            spec = _require_mapping(value.get("spec"), "continual spec is invalid")
            if spec.get(field) != reference_spec[field]:
                raise ValueError(f"continual runs disagree on spec field {field!r}")
    tasks = continual[0].get("tasks")
    if any(value.get("tasks") != tasks for value in continual[1:]):
        raise ValueError("continual runs disagree on task order")
    method_maps = [
        _require_mapping(value.get("methods"), "continual methods are invalid")
        for value in continual
    ]
    methods = set(method_maps[0])
    if any(set(value) != methods for value in method_maps[1:]):
        raise ValueError("continual runs contain different methods")
    summary: dict[str, dict[str, Any]] = {}
    for method in sorted(methods):
        method_summaries = [
            _require_mapping(
                value[method].get("summary"), "continual method summary is invalid"
            )
            for value in method_maps
        ]
        metric_names = set(method_summaries[0])
        if any(set(value) != metric_names for value in method_summaries[1:]):
            raise ValueError("continual method summaries contain different metrics")
        summary[method] = {}
        for metric in sorted(metric_names):
            values = [value[metric] for value in method_summaries]
            if all(isinstance(value, (int, float)) for value in values):
                numeric = [float(value) for value in values]
                summary[method][metric] = {
                    "mean": fmean(numeric),
                    "sample_std": stdev(numeric),
                    "by_seed": {
                        str(seed): value
                        for seed, value in zip(seeds, numeric, strict=True)
                    },
                }
            elif all(isinstance(value, list) for value in values):
                widths = {len(value) for value in values}
                if len(widths) != 1 or any(
                    not all(isinstance(item, (int, float)) for item in value)
                    for value in values
                ):
                    raise ValueError(
                        "continual list metrics must be matched numeric lists"
                    )
                summary[method][metric] = {
                    "mean": [
                        fmean(float(value[index]) for value in values)
                        for index in range(len(values[0]))
                    ],
                    "by_seed": {
                        str(seed): value
                        for seed, value in zip(seeds, values, strict=True)
                    },
                }
            else:
                raise ValueError("continual summary metrics must be numeric")
    frozen = summary["frozen"]
    comparisons = {}
    for method, metrics in summary.items():
        comparisons[method] = {
            "clean_accuracy_change_vs_frozen": metrics["final_clean_accuracy"]["mean"]
            - frozen["final_clean_accuracy"]["mean"],
            "shifted_accuracy_change_vs_frozen": metrics["final_shifted_accuracy"][
                "mean"
            ]
            - frozen["final_shifted_accuracy"]["mean"],
            "forgetting_change_vs_frozen": metrics["mean_forgetting"]["mean"]
            - frozen["mean_forgetting"]["mean"],
        }
    return {
        **{field: reference[field] for field in _TOP_LEVEL_FIELDS},
        "seeds": seeds,
        "run_count": len(runs),
        "tasks": tasks,
        "spec": {field: reference_spec[field] for field in _SPEC_FIELDS},
        "summary": summary,
        "comparisons": comparisons,
    }
