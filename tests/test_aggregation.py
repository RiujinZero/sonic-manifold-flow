import copy

import pytest

from manifoldflow.aggregation import aggregate_flow_metrics


def _run(seed: int, mse: float, accuracy: float) -> dict[str, object]:
    return {
        "dataset": "FashionMNIST",
        "latent_width": 128,
        "basins": 1000,
        "eval_samples": 2000,
        "candidate_count": 64,
        "soft_top_k": 4,
        "temperature": 0.1,
        "times": [0.1],
        "image_noise_std": 0.2,
        "seed": seed,
        "ablation": {
            "t0": {
                "noisy": {
                    "reconstruction_mse": mse,
                    "classification_accuracy": accuracy,
                }
            }
        },
    }


def test_aggregate_flow_metrics_reports_mean_std_and_seed_values() -> None:
    result = aggregate_flow_metrics(
        [_run(17, 0.02, 0.70), _run(18, 0.01, 0.72), _run(19, 0.03, 0.71)]
    )
    mse = result["summary"]["t0"]["noisy"]["reconstruction_mse"]

    assert result["seeds"] == [17, 18, 19]
    assert mse["mean"] == pytest.approx(0.02)
    assert mse["sample_std"] == pytest.approx(0.01)
    assert mse["by_seed"] == {"17": 0.02, "18": 0.01, "19": 0.03}


def test_aggregate_flow_metrics_rejects_mismatched_protocols() -> None:
    first = _run(17, 0.02, 0.70)
    second = copy.deepcopy(_run(18, 0.01, 0.72))
    second["image_noise_std"] = 0.3

    with pytest.raises(ValueError, match="image_noise_std"):
        aggregate_flow_metrics([first, second])


def test_aggregate_flow_metrics_checks_optional_baseline_protocol() -> None:
    first = _run(17, 0.02, 0.70)
    second = _run(18, 0.01, 0.72)
    first["prototype_basins"] = 2000
    second["prototype_basins"] = 1000

    with pytest.raises(ValueError, match="prototype_basins"):
        aggregate_flow_metrics([first, second])


def test_aggregate_flow_metrics_checks_diagnostic_reference() -> None:
    first = _run(17, 0.02, 0.70)
    second = _run(18, 0.01, 0.72)
    first["diagnostic_rbf_strength"] = 0.1
    second["diagnostic_rbf_strength"] = 0.2

    with pytest.raises(ValueError, match="diagnostic_rbf_strength"):
        aggregate_flow_metrics([first, second])
