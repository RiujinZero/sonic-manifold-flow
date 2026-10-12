"""Shared artifact loading for the first routed-flow experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from manifoldflow.bank import BasinState
from manifoldflow.data import ImageDatasetSpec, image_dataset_spec
from manifoldflow.models import CompactImageAutoencoder
from manifoldflow.normalization import FrozenDiagonalWhitening

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_representation(
    checkpoint_path: Path,
    *,
    dataset: str,
) -> tuple[
    CompactImageAutoencoder, FrozenDiagonalWhitening, nn.Linear, ImageDatasetSpec
]:
    """Restore the Phase A model, normalizer, and frozen linear probe."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"dataset", "latent_width", "model", "normalizer", "linear_probe"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    spec = image_dataset_spec(dataset)
    if payload["dataset"] != spec.name:
        raise ValueError(
            f"checkpoint dataset {payload['dataset']!r} does not match {spec.name!r}"
        )
    latent_width = int(payload["latent_width"])
    model = CompactImageAutoencoder(
        input_channels=spec.channels,
        input_size=spec.image_size,
        latent_dim=latent_width,
    )
    model.load_state_dict(payload["model"])
    normalizer = FrozenDiagonalWhitening(latent_width)
    normalizer.load_state_dict(payload["normalizer"])
    probe = nn.Linear(latent_width, spec.classes)
    probe.load_state_dict(payload["linear_probe"])
    probe.requires_grad_(False)
    return model, normalizer, probe, spec


def load_bank(path: Path, *, expected_dimension: int) -> BasinState:
    bank, _ = load_valued_bank(path, expected_dimension=expected_dimension)
    return bank


def load_valued_bank(
    path: Path,
    *,
    expected_dimension: int,
    expected_classes: int | None = None,
) -> tuple[BasinState, torch.Tensor | None]:
    """Load core basin state and an optional validated class distribution."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("bank artifact must contain a mapping")
    bank = BasinState.from_serialized(payload)
    if bank.mu.shape[1] != expected_dimension:
        raise ValueError("bank dimension does not match the representation")
    class_count = payload.get("class_count")
    if class_count is None:
        if expected_classes is not None:
            raise ValueError("bank artifact is missing class_count")
        return bank, None
    if not isinstance(class_count, torch.Tensor) or class_count.ndim != 2:
        raise ValueError("bank class_count must have shape [basins, classes]")
    if class_count.shape[0] != bank.mu.shape[0]:
        raise ValueError("bank class_count rows must match basin count")
    if expected_classes is not None and class_count.shape[1] != expected_classes:
        raise ValueError("bank class_count does not match the dataset classes")
    if torch.any(class_count < 0) or torch.any(class_count.sum(dim=1) <= 0):
        raise ValueError("every basin must have non-negative, non-empty class counts")
    class_probability = class_count.float()
    class_probability /= class_probability.sum(dim=1, keepdim=True)
    return bank, class_probability
