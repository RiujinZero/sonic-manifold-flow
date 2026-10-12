import torch

from manifoldflow.bank import BasinState
from manifoldflow.conditional import (
    calibrate_conditional_gate,
    calibrate_conditional_gate_grid,
    conditional_flow,
)


def test_conditional_flow_requires_confident_and_far_route() -> None:
    bank = BasinState.from_centers(
        torch.tensor([[0.0, 0.0], [10.0, 0.0]]), initial_rate=1.0
    )
    latent = torch.tensor(
        [
            [2.0, 0.0],  # confident destination, unusually far
            [0.1, 0.0],  # confident destination, already clean
            [5.0, 0.0],  # far, but ambiguous between basins
        ]
    )
    result = conditional_flow(
        latent,
        bank,
        confidence_threshold=0.8,
        distance_threshold=1.0,
        candidates=2,
        top_k=2,
        flow_time=0.1,
        temperature=0.1,
    )

    assert result.active.tolist() == [True, False, False]
    assert not torch.equal(result.latent[0], latent[0])
    torch.testing.assert_close(result.latent[1:], latent[1:])


def test_conditional_gate_calibrates_far_threshold_from_clean_latents() -> None:
    bank = BasinState.from_centers(
        torch.tensor([[0.0, 0.0], [10.0, 0.0]]), initial_rate=1.0
    )
    clean = torch.tensor([[0.1, 0.0], [0.2, 0.0], [9.8, 0.0], [9.6, 0.0]])
    calibration = calibrate_conditional_gate(
        clean,
        bank,
        confidence_threshold=0.8,
        distance_quantile=0.5,
        candidates=2,
        top_k=2,
        chunk_size=2,
    )

    assert calibration.samples == 4
    assert calibration.confident_fraction == 1.0
    assert calibration.distance_threshold > 0
    assert 0 < calibration.activation_fraction <= 1


def test_semantic_margin_blocks_a_confident_geometric_route() -> None:
    bank = BasinState.from_centers(
        torch.tensor([[0.0, 0.0], [10.0, 0.0]]), initial_rate=1.0
    )
    latent = torch.tensor([[2.0, 0.0]])
    ambiguous_value = torch.tensor([[0.55, 0.45], [0.0, 1.0]])
    clear_value = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    blocked = conditional_flow(
        latent,
        bank,
        confidence_threshold=0.8,
        distance_threshold=1.0,
        semantic_margin_threshold=0.5,
        candidates=2,
        top_k=2,
        flow_time=0.1,
        temperature=0.1,
        class_probability=ambiguous_value,
    )
    active = conditional_flow(
        latent,
        bank,
        confidence_threshold=0.8,
        distance_threshold=1.0,
        semantic_margin_threshold=0.5,
        candidates=2,
        top_k=2,
        flow_time=0.1,
        temperature=0.1,
        class_probability=clear_value,
    )

    assert not blocked.active.item()
    assert active.active.item()
    assert blocked.semantic_margin is not None
    assert active.semantic_margin is not None
    assert blocked.semantic_margin.item() < active.semantic_margin.item()


def test_semantic_gate_grid_reuses_clean_route_evidence() -> None:
    bank = BasinState.from_centers(
        torch.tensor([[0.0, 0.0], [10.0, 0.0]]), initial_rate=1.0
    )
    clean = torch.tensor([[0.1, 0.0], [0.2, 0.0], [9.8, 0.0], [9.6, 0.0]])
    class_probability = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    calibrations = calibrate_conditional_gate_grid(
        clean,
        bank,
        class_probability,
        confidence_thresholds=(0.5, 0.8),
        distance_quantiles=(0.5, 0.75),
        semantic_margin_thresholds=(0.2, 0.6),
        candidates=2,
        top_k=2,
        chunk_size=2,
    )

    assert len(calibrations) == 8
    assert len({calibration.name for calibration in calibrations}) == 8
    assert all(
        calibration.semantic_margin_threshold is not None
        for calibration in calibrations
    )
    assert all(
        calibration.spec.name == calibration.name for calibration in calibrations
    )
