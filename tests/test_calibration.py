import copy

import torch
from torch import nn

from manifoldflow.bank import BasinState
from manifoldflow.calibration import (
    calibrate_decoder,
    calibrate_neutral_decoder,
    evaluate_flow_ablation,
)
from manifoldflow.models import CompactImageAutoencoder
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.training import batches_from_tensors


def _fixture() -> tuple[
    CompactImageAutoencoder,
    FrozenDiagonalWhitening,
    BasinState,
    list[tuple[torch.Tensor, torch.Tensor]],
]:
    generator = torch.Generator().manual_seed(8)
    image = torch.rand(6, 1, 16, 16, generator=generator)
    label = torch.tensor([0, 1, 0, 1, 0, 1])
    batches = list(batches_from_tensors(image, label, batch_size=3))
    model = CompactImageAutoencoder(input_channels=1, input_size=16, latent_dim=4)
    with torch.no_grad():
        raw_latent = model.encode(image)
    normalizer = FrozenDiagonalWhitening(4)
    normalizer.fit(raw_latent)
    normalized = normalizer(raw_latent)
    bank = BasinState.from_centers(normalized[:3], initial_rate=2.0)
    return model, normalizer, bank, batches


def test_calibration_updates_decoder_but_not_encoder() -> None:
    model, normalizer, bank, batches = _fixture()
    encoder_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("encoder_")
    }
    decoder_before = copy.deepcopy(model.decoder_projection.state_dict())

    result = calibrate_decoder(
        model,
        normalizer,
        bank,
        batches,
        device="cpu",
        epochs=1,
        candidates=3,
        max_batches=1,
    )

    assert len(result.total_loss) == 1
    assert result.total_loss[0] > 0
    for name, expected in encoder_before.items():
        torch.testing.assert_close(model.state_dict()[name], expected)
    assert any(
        not torch.equal(value, decoder_before[name])
        for name, value in model.decoder_projection.state_dict().items()
    )
    assert all(
        not parameter.requires_grad
        for module in (model.encoder_conv, model.encoder_projection)
        for parameter in module.parameters()
    )


def test_flow_ablation_returns_matched_clean_and_noisy_metrics() -> None:
    model, normalizer, bank, batches = _fixture()
    probe = nn.Linear(4, 2)
    result = evaluate_flow_ablation(
        model,
        normalizer,
        probe,
        bank,
        batches,
        device="cpu",
        times=(0.25,),
        candidates=3,
        soft_top_k=2,
        max_samples=4,
    )

    assert set(result) == {"t0", "hard_t0.25", "soft_t0.25"}
    for variant in result.values():
        assert set(variant) == {"clean", "noisy"}
        for metrics in variant.values():
            assert metrics["reconstruction_mse"] >= 0
            assert 0 <= metrics["classification_accuracy"] <= 1
            assert metrics["mean_latent_displacement"] >= 0
    assert result["t0"]["clean"]["mean_latent_displacement"] == 0


def test_neutral_calibration_exposes_every_transform_branch() -> None:
    generator = torch.Generator().manual_seed(12)
    image = torch.rand(12, 1, 16, 16, generator=generator)
    label = torch.arange(12) % 2
    batches = list(batches_from_tensors(image, label, batch_size=2))
    model = CompactImageAutoencoder(input_channels=1, input_size=16, latent_dim=4)
    with torch.no_grad():
        raw_latent = model.encode(image)
    normalizer = FrozenDiagonalWhitening(4)
    normalizer.fit(raw_latent)
    normalized = normalizer(raw_latent)
    flow_bank = BasinState.from_centers(normalized[:2], initial_rate=2.0)

    result = calibrate_neutral_decoder(
        model,
        normalizer,
        flow_bank,
        normalized[:4],
        batches,
        device="cpu",
        epochs=1,
        flow_times=(0.05, 0.1),
        prototype_strengths=(0.1, 0.2),
        candidates=2,
        flow_top_k=2,
        rbf_top_k=2,
    )

    assert set(result.branch_loss) == {
        "flow_t0.05",
        "flow_t0.1",
        "prototype_a0.1",
        "prototype_a0.2",
        "rbf_a0.1",
        "rbf_a0.2",
    }
    assert all(len(history) == 1 for history in result.branch_loss.values())
