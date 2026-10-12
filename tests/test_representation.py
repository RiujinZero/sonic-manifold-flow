import torch

from manifoldflow.data import image_dataset_spec
from manifoldflow.models import CompactImageAutoencoder, trainable_parameter_count
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.training import (
    batches_from_tensors,
    collect_latents,
    evaluate_reconstruction,
    train_autoencoder,
)


def test_dataset_specs_cover_both_mvp_domains() -> None:
    fashion = image_dataset_spec("fashion-mnist")
    cifar = image_dataset_spec("CIFAR_10")
    assert (fashion.channels, fashion.image_size, fashion.classes) == (1, 28, 10)
    assert (cifar.channels, cifar.image_size, cifar.classes) == (3, 32, 10)


@torch.no_grad()
def test_paired_autoencoder_shapes_for_mvp_datasets() -> None:
    for channels, size in ((1, 28), (3, 32)):
        model = CompactImageAutoencoder(
            input_channels=channels, input_size=size, latent_dim=64
        )
        image = torch.rand(2, channels, size, size)
        reconstruction, latent = model(image)
        assert reconstruction.shape == image.shape
        assert latent.shape == (2, 64)
        assert torch.all((0 <= reconstruction) & (reconstruction <= 1))


def test_freeze_removes_all_trainable_parameters() -> None:
    model = CompactImageAutoencoder(input_channels=1, input_size=28, latent_dim=64)
    assert trainable_parameter_count(model) > 0
    model.freeze()
    assert trainable_parameter_count(model) == 0
    assert not model.training


def test_diagonal_whitening_fits_and_roundtrips_state() -> None:
    latent = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0], [7.0, 10.0]])
    normalizer = FrozenDiagonalWhitening(2, l2_normalize=False)
    normalizer.fit(latent)
    normalized = normalizer(latent)

    torch.testing.assert_close(normalized.mean(0), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        normalized.var(0, unbiased=False), torch.ones(2), atol=1e-5, rtol=1e-5
    )
    restored = FrozenDiagonalWhitening(2, l2_normalize=False)
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored(latent), normalized)


def test_training_and_collection_smoke_path() -> None:
    generator = torch.Generator().manual_seed(4)
    image = torch.rand(4, 1, 28, 28, generator=generator)
    label = torch.tensor([0, 1, 0, 1])
    batches = batches_from_tensors(image, label, batch_size=2)
    model = CompactImageAutoencoder(input_channels=1, input_size=28, latent_dim=8)

    history = train_autoencoder(model, batches, device="cpu", epochs=1)
    reconstruction_mse = evaluate_reconstruction(model, batches, device="cpu")
    collected = collect_latents(model, batches, device="cpu", max_samples=3)

    assert len(history) == 1
    assert history[0] > 0
    assert reconstruction_mse > 0
    assert collected.latent.shape == (3, 8)
    assert collected.label.tolist() == [0, 1, 0]
