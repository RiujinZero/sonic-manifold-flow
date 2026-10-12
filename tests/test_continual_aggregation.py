import copy

import pytest

from manifoldflow import aggregate_continual_metrics


def _run(seed: int, shifted: float) -> dict:
    summary = {
        "initial_clean_accuracy": 0.8,
        "final_clean_accuracy": 0.79,
        "clean_accuracy_change": -0.01,
        "final_shifted_accuracy": shifted,
        "mean_task_adaptation": 0.02,
        "mean_forgetting": 0.01,
        "initial_basins": 2,
        "final_basins": 2,
        "basins_created": 0,
        "final_delta_size": 1,
        "final_pending_novelty": 0,
        "final_updated_rows": 1,
        "stored_vector_values": 8,
        "stored_value_scalars": 4,
        "novel_observations": 0,
        "created_observations": 0,
        "capacity_rejections": 0,
        "task_adaptation": [0.02, 0.01],
    }
    spec = {
        "classes_per_task": 1,
        "stream_samples_per_class": 2,
        "shift_scale": 0.2,
        "latent_noise_std": 0.01,
        "novelty_quantile": 0.95,
        "novelty_persistence": 3,
        "novelty_support_multiplier": 1.5,
        "basin_growth_budget": 2,
        "rbf_top_k": 2,
        "rbf_temperature": 0.1,
        "manifold_vector_values": 8,
        "prototype_vector_values": 8,
    }
    return {
        "seed": seed,
        "dimension": 2,
        "initial_basins": 2,
        "prototype_basins": 4,
        "train_samples": 6,
        "test_samples": 4,
        "continual": {
            "spec": {**spec, "seed": seed},
            "tasks": [[0], [1]],
            "methods": {
                "frozen": {"summary": {**summary, "final_shifted_accuracy": 0.7}},
                "local_fixed": {"summary": summary},
            },
        },
    }


def test_continual_aggregation_reports_comparisons() -> None:
    result = aggregate_continual_metrics([_run(1, 0.72), _run(2, 0.74)])

    assert result["summary"]["local_fixed"]["final_shifted_accuracy"]["mean"] == 0.73
    assert result["comparisons"]["local_fixed"][
        "shifted_accuracy_change_vs_frozen"
    ] == pytest.approx(0.03)
    assert result["summary"]["local_fixed"]["task_adaptation"]["mean"] == [
        0.02,
        0.01,
    ]


def test_continual_aggregation_rejects_protocol_mismatch() -> None:
    first = _run(1, 0.72)
    second = copy.deepcopy(_run(2, 0.74))
    second["continual"]["spec"]["rbf_top_k"] = 3

    with pytest.raises(ValueError, match="rbf_top_k"):
        aggregate_continual_metrics([first, second])
