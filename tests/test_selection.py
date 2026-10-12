from manifoldflow.selection import select_operating_points


def _metric(mean: float) -> dict[str, float]:
    return {"mean": mean, "sample_std": 0.0}


def _variant(clean_mse: float, clean_acc: float, noisy_mse: float, noisy_acc: float):
    return {
        "clean": {
            "reconstruction_mse": _metric(clean_mse),
            "classification_accuracy": _metric(clean_acc),
        },
        "noisy": {
            "reconstruction_mse": _metric(noisy_mse),
            "classification_accuracy": _metric(noisy_acc),
        },
    }


def test_selector_uses_clean_constraints_before_noisy_mse() -> None:
    aggregate = {
        "seeds": [17, 18, 19],
        "summary": {
            "t0": _variant(1.0, 0.90, 2.0, 0.70),
            "rbf_a0.1": _variant(1.02, 0.90, 1.6, 0.75),
            "rbf_a0.2": _variant(1.10, 0.91, 1.2, 0.80),
            "flow_soft_t0.1": _variant(1.01, 0.897, 1.7, 0.73),
            "conditional_flow_t0.1": _variant(1.00, 0.90, 1.5, 0.76),
            "semantic_flow_c0.8_q0.9_m0.4_t0.1": _variant(1.01, 0.90, 1.4, 0.77),
            "semantic_hybrid_rbf0.1_c0.8_q0.9_m0.4_t0.1": _variant(
                1.02, 0.90, 1.3, 0.78
            ),
        },
    }
    result = select_operating_points(
        aggregate,
        max_clean_mse_relative_increase=0.05,
        max_clean_accuracy_drop=0.005,
    )

    assert result["selected"]["rbf"]["variant"] == "rbf_a0.1"
    assert result["selected"]["flow"]["variant"] == "flow_soft_t0.1"
    assert result["selected"]["conditional_flow"]["variant"] == "conditional_flow_t0.1"
    assert (
        result["selected"]["semantic_flow"]["variant"]
        == "semantic_flow_c0.8_q0.9_m0.4_t0.1"
    )
    assert (
        result["selected"]["semantic_hybrid"]["variant"]
        == "semantic_hybrid_rbf0.1_c0.8_q0.9_m0.4_t0.1"
    )
    assert result["selected"]["prototype"] is None
