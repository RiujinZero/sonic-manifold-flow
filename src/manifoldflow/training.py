"""Training and representation extraction utilities for Phase A."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from manifoldflow.models import CompactImageAutoencoder


@dataclass(frozen=True)
class LatentDataset:
    """CPU-resident latent/label pair collected from a frozen encoder."""

    latent: Tensor
    label: Tensor


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch without forcing globally deterministic kernels."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_autoencoder(
    model: CompactImageAutoencoder,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str,
    epochs: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_batches: int | None = None,
) -> tuple[float, ...]:
    """Train the paired model and return mean reconstruction loss per epoch."""
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    for _ in range(epochs):
        loss_sum = 0.0
        samples = 0
        for batch_index, (image, _) in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            image = image.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(image)
            loss = F.mse_loss(reconstruction, image)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * image.shape[0]
            samples += image.shape[0]
        if samples == 0:
            raise ValueError("training batches produced no samples")
        history.append(loss_sum / samples)
    return tuple(history)


@torch.no_grad()
def evaluate_reconstruction(
    model: CompactImageAutoencoder,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str,
    max_samples: int | None = None,
) -> float:
    """Return no-flow mean squared reconstruction error."""
    model.to(device)
    model.eval()
    loss_sum = 0.0
    samples = 0
    for image, _ in batches:
        if max_samples is not None:
            remaining = max_samples - samples
            if remaining <= 0:
                break
            image = image[:remaining]
        image = image.to(device)
        reconstruction, _ = model(image)
        loss_sum += float(F.mse_loss(reconstruction, image)) * image.shape[0]
        samples += image.shape[0]
    if samples == 0:
        raise ValueError("evaluation batches produced no samples")
    return loss_sum / samples


@torch.no_grad()
def collect_latents(
    model: CompactImageAutoencoder,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str,
    max_samples: int | None = None,
) -> LatentDataset:
    """Encode a bounded dataset and return contiguous CPU tensors."""
    model.to(device)
    model.eval()
    latent_parts = []
    label_parts = []
    samples = 0
    for image, label in batches:
        if max_samples is not None:
            remaining = max_samples - samples
            if remaining <= 0:
                break
            image = image[:remaining]
            label = label[:remaining]
        latent_parts.append(model.encode(image.to(device)).cpu())
        label_parts.append(label.to(dtype=torch.long, device="cpu"))
        samples += image.shape[0]
    if not latent_parts:
        raise ValueError("latent batches produced no samples")
    return LatentDataset(
        latent=torch.cat(latent_parts).contiguous(),
        label=torch.cat(label_parts).contiguous(),
    )


@torch.no_grad()
def collect_flattened_inputs(
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    max_samples: int,
) -> Tensor:
    """Collect a bounded, ordered input sample for neighborhood evaluation."""
    if max_samples < 2:
        raise ValueError("max_samples must be at least two")
    image_parts = []
    samples = 0
    for image, _ in batches:
        remaining = max_samples - samples
        if remaining <= 0:
            break
        image = image[:remaining]
        image_parts.append(image.flatten(start_dim=1).cpu())
        samples += image.shape[0]
    if samples < 2:
        raise ValueError("input batches produced fewer than two samples")
    return torch.cat(image_parts).contiguous()


def batches_from_tensors(
    image: Tensor,
    label: Tensor,
    *,
    batch_size: int,
) -> list[tuple[Tensor, Tensor]]:
    """Small synthetic-data helper used by smoke tests and examples."""
    if image.shape[0] != label.shape[0] or batch_size <= 0:
        raise ValueError(
            "image/label sample counts must match and batch_size be positive"
        )
    return [
        (image[start : start + batch_size], label[start : start + batch_size])
        for start in range(0, image.shape[0], batch_size)
    ]
