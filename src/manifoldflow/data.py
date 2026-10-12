"""Small-image dataset definitions and reproducible loader construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


@dataclass(frozen=True)
class ImageDatasetSpec:
    """Shape and label metadata needed to build a paired image model."""

    name: str
    channels: int
    image_size: int
    classes: int


_SPECS = {
    "fashionmnist": ImageDatasetSpec("FashionMNIST", 1, 28, 10),
    "cifar10": ImageDatasetSpec("CIFAR10", 3, 32, 10),
}


def image_dataset_spec(name: str) -> ImageDatasetSpec:
    """Resolve a supported dataset name without importing torchvision."""
    key = name.lower().replace("-", "").replace("_", "")
    try:
        return _SPECS[key]
    except KeyError as error:
        choices = ", ".join(spec.name for spec in _SPECS.values())
        message = f"unsupported dataset {name!r}; choose one of {choices}"
        raise ValueError(message) from error


def build_image_dataloaders(
    name: str,
    *,
    root: str | Path,
    batch_size: int,
    num_workers: int = 0,
    download: bool = False,
    seed: int = 17,
) -> tuple[DataLoader[Any], DataLoader[Any], ImageDatasetSpec]:
    """Create deterministic train/test loaders for a supported small dataset.

    Torchvision is imported lazily so the mathematical core remains usable
    without the optional vision dependency.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError(
            "Phase A datasets require the vision extra: "
            "python -m pip install -e '.[vision]'"
        ) from error

    spec = image_dataset_spec(name)
    dataset_class = getattr(datasets, spec.name)
    transform = transforms.ToTensor()
    root = Path(root).expanduser()
    train_dataset = dataset_class(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = dataset_class(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, test_loader, spec


def build_imagefolder_dataloaders(
    root: str | Path,
    *,
    batch_size: int,
    transform: Any,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 17,
) -> tuple[DataLoader[Any], DataLoader[Any], int]:
    """Build deterministic train/val loaders from an ImageFolder-style layout.

    If `root` already contains `train/` and `val/` subdirectories (the
    standard ImageNet-1K layout: `root/train/<class>/*.JPEG`,
    `root/val/<class>/*.JPEG`), each is loaded as its own `ImageFolder` and
    the split is whatever that layout already encodes. Otherwise `root` is
    loaded as one `ImageFolder` and split by a seeded, deterministic random
    permutation held out at `val_fraction` (not stratified by class - a flat
    folder without a pre-made split is assumed to be a smoke/scale fixture,
    not a class-balance-sensitive benchmark).

    Returns `(train_loader, val_loader, classes)`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    try:
        from torchvision import datasets
    except ImportError as error:
        raise RuntimeError(
            "ImageFolder ingestion requires the vision extra: "
            "python -m pip install -e '.[vision]'"
        ) from error

    root = Path(root).expanduser()
    train_dir, val_dir = root / "train", root / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        classes = len(train_dataset.classes)
    else:
        full_dataset = datasets.ImageFolder(root, transform=transform)
        classes = len(full_dataset.classes)
        split_generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(
            len(full_dataset), generator=split_generator
        ).tolist()
        val_count = max(1, round(len(full_dataset) * val_fraction))
        if val_count >= len(full_dataset):
            raise ValueError("val_fraction leaves no training rows")
        train_dataset = Subset(full_dataset, permutation[val_count:])
        val_dataset = Subset(full_dataset, permutation[:val_count])

    loader_generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=loader_generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, classes
