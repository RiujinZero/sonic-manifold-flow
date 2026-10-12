import pytest
import torch
from torch import nn

from manifoldflow.bank import BasinState
from manifoldflow.conditional import ConditionalGateSpec
from manifoldflow.matched_baselines import evaluate_matched_baselines
from manifoldflow.models import CompactImageAutoencoder
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.prototype import (
    nearest_prototype_class_scores,
    nearest_prototype_contraction,
    rbf_prototype_class_scores,
    rbf_prototype_contraction,
)
from manifoldflow.training import batches_from_tensors


def test_nearest_prototype_class_scores_returns_nearest_center_row() -> None:
    latent = torch.tensor([[0.1, 0.0], [1.9, 0.0]])
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    class_count = torch.tensor([[5, 1], [0, 4]])

    scores = nearest_prototype_class_scores(latent, centers, class_count)

    torch.testing.assert_close(scores, torch.tensor([[5.0, 1.0], [0.0, 4.0]]))


def test_rbf_prototype_class_scores_blends_nearby_centers() -> None:
    latent = torch.tensor([[1.0, 0.0]])
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    class_count = torch.tensor([[10, 0], [0, 10]])

    scores = rbf_prototype_class_scores(
        latent, centers, class_count, top_k=2, temperature=0.1
    )

    # Equidistant from both centers with identical, opposite class weight:
    # the mixture should land at 50/50, not favor either class.
    torch.testing.assert_close(scores, torch.tensor([[0.5, 0.5]]))


def test_prototype_class_scores_reject_mismatched_class_count_rows() -> None:
    latent = torch.tensor([[0.0, 0.0]])
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    bad_class_count = torch.tensor([[1, 0]])  # only 1 row for 2 centers

    with pytest.raises(ValueError):
        nearest_prototype_class_scores(latent, centers, bad_class_count)
    with pytest.raises(ValueError):
        rbf_prototype_class_scores(latent, centers, bad_class_count)


def test_nearest_prototype_contraction_interpolates_by_strength() -> None:
    latent = torch.tensor([[0.5, 0.0]])
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    result = nearest_prototype_contraction(latent, centers, strength=0.25)

    torch.testing.assert_close(result, torch.tensor([[0.375, 0.0]]))


def test_rbf_prototype_contraction_uses_soft_local_target() -> None:
    latent = torch.tensor([[1.0, 0.0]])
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    result = rbf_prototype_contraction(
        latent, centers, strength=1.0, top_k=2, temperature=0.1
    )

    torch.testing.assert_close(result, latent)


def test_matched_baseline_evaluator_reports_every_transform_family() -> None:
    generator = torch.Generator().manual_seed(9)
    image = torch.rand(4, 1, 16, 16, generator=generator)
    label = torch.tensor([0, 1, 0, 1])
    batches = list(batches_from_tensors(image, label, batch_size=2))
    model = CompactImageAutoencoder(input_channels=1, input_size=16, latent_dim=4)
    with torch.no_grad():
        raw_latent = model.encode(image)
    normalizer = FrozenDiagonalWhitening(4)
    normalizer.fit(raw_latent)
    normalized = normalizer(raw_latent)
    flow_bank = BasinState.from_centers(normalized[:2], initial_rate=2.0)
    probe = nn.Linear(4, 2)

    result = evaluate_matched_baselines(
        model,
        normalizer,
        probe,
        flow_bank,
        normalized,
        torch.stack((normalized[label == 0].mean(0), normalized[label == 1].mean(0))),
        batches,
        device="cpu",
        flow_times=(0.1,),
        strengths=(0.2,),
        candidates=2,
        flow_top_k=2,
        rbf_top_k=2,
        conditional_confidence_threshold=0.0,
        conditional_distance_threshold=0.0,
        conditional_flow_time=0.1,
        diagnostic_rbf_strength=0.2,
        max_samples=3,
    )

    assert set(result) == {
        "t0",
        "flow_soft_t0.1",
        "vq_projection",
        "prototype_a0.2",
        "rbf_a0.2",
        "class_centroid_a0.2",
        "conditional_flow_t0.1",
    }
    for conditions in result.values():
        for metrics in conditions.values():
            assert metrics["reconstruction_mse"] >= 0
            assert 0 <= metrics["classification_accuracy"] <= 1
    conditional = result["conditional_flow_t0.1"]["noisy"]
    assert conditional["activation_count"] == 3
    assert "rbf_correct_to_wrong_count" in conditional
    assert "rbf_wrong_to_correct_count" in conditional
    assert "rbf_correct_to_wrong_count" in result["flow_soft_t0.1"]["noisy"]


def test_matched_evaluator_reports_semantic_gate_variant() -> None:
    generator = torch.Generator().manual_seed(12)
    image = torch.rand(3, 1, 16, 16, generator=generator)
    label = torch.tensor([0, 1, 0])
    batches = list(batches_from_tensors(image, label, batch_size=3))
    model = CompactImageAutoencoder(input_channels=1, input_size=16, latent_dim=4)
    with torch.no_grad():
        raw_latent = model.encode(image)
    normalizer = FrozenDiagonalWhitening(4)
    normalizer.fit(raw_latent)
    normalized = normalizer(raw_latent)
    flow_bank = BasinState.from_centers(normalized[:2], initial_rate=2.0)
    probe = nn.Linear(4, 2)
    name = "semantic_flow_c0_q0.5_m0_t0.1"

    result = evaluate_matched_baselines(
        model,
        normalizer,
        probe,
        flow_bank,
        normalized,
        torch.stack((normalized[label == 0].mean(0), normalized[label == 1].mean(0))),
        batches,
        device="cpu",
        flow_times=(0.1,),
        strengths=(0.2,),
        candidates=2,
        flow_top_k=2,
        rbf_top_k=2,
        flow_class_probability=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        conditional_gates=(
            ConditionalGateSpec(
                name=name,
                confidence_threshold=0.0,
                distance_threshold=0.0,
                semantic_margin_threshold=0.0,
            ),
        ),
        conditional_flow_time=0.1,
        diagnostic_rbf_strength=0.2,
        max_samples=2,
    )

    assert result[name]["noisy"]["activation_count"] == 2
    assert "rbf_correct_to_wrong_count" in result[name]["noisy"]


def test_semantic_hybrid_uses_flow_only_on_active_gate() -> None:
    generator = torch.Generator().manual_seed(21)
    image = torch.rand(3, 1, 16, 16, generator=generator)
    label = torch.tensor([0, 1, 0])
    batches = list(batches_from_tensors(image, label, batch_size=3))
    model = CompactImageAutoencoder(input_channels=1, input_size=16, latent_dim=4)
    with torch.no_grad():
        raw_latent = model.encode(image)
    normalizer = FrozenDiagonalWhitening(4)
    normalizer.fit(raw_latent)
    normalized = normalizer(raw_latent)
    flow_bank = BasinState.from_centers(normalized[:2], initial_rate=2.0)
    probe = nn.Linear(4, 2)
    active_name = "semantic_flow_active"
    inactive_name = "semantic_flow_inactive"

    result = evaluate_matched_baselines(
        model,
        normalizer,
        probe,
        flow_bank,
        normalized,
        torch.stack((normalized[label == 0].mean(0), normalized[label == 1].mean(0))),
        batches,
        device="cpu",
        flow_times=(0.1,),
        strengths=(0.2,),
        candidates=2,
        flow_top_k=2,
        rbf_top_k=2,
        flow_class_probability=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        conditional_gates=(
            ConditionalGateSpec(active_name, 0.0, 0.0, 0.0),
            ConditionalGateSpec(inactive_name, 1.0, float("inf"), 1.0),
        ),
        conditional_flow_time=0.1,
        diagnostic_rbf_strength=0.2,
        include_semantic_hybrids=True,
        image_noise_std=0.0,
        max_samples=3,
    )

    active_hybrid = result["semantic_hybrid_rbf0.2_active"]
    inactive_hybrid = result["semantic_hybrid_rbf0.2_inactive"]
    for condition in ("clean", "noisy"):
        assert active_hybrid[condition] == result[active_name][condition]
        assert (
            inactive_hybrid[condition]["reconstruction_mse"]
            == result["rbf_a0.2"][condition]["reconstruction_mse"]
        )
        assert inactive_hybrid[condition]["activation_count"] == 0
