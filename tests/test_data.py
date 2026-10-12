import pytest
from PIL import Image
from torchvision import transforms

from manifoldflow.data import build_imagefolder_dataloaders

_TRANSFORM = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])


def _write_flat_imagefolder(root, *, classes=("cat", "dog"), per_class=6):
    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(per_class):
            Image.new("RGB", (16, 16), color=(index * 10, 0, 0)).save(
                class_dir / f"{index}.png"
            )


def _write_presplit_imagefolder(root, *, classes=("cat", "dog"), per_split=4):
    for split in ("train", "val"):
        for class_name in classes:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for index in range(per_split):
                Image.new("RGB", (16, 16), color=(index * 10, 0, 0)).save(
                    class_dir / f"{index}.png"
                )


def test_flat_imagefolder_splits_deterministically_by_seed(tmp_path) -> None:
    _write_flat_imagefolder(tmp_path)

    train_loader, val_loader, classes = build_imagefolder_dataloaders(
        tmp_path, batch_size=4, transform=_TRANSFORM, val_fraction=0.25, seed=17
    )
    train_loader_again, val_loader_again, _ = build_imagefolder_dataloaders(
        tmp_path, batch_size=4, transform=_TRANSFORM, val_fraction=0.25, seed=17
    )

    assert classes == 2
    assert len(val_loader.dataset) == 3  # 25% of 12
    assert len(train_loader.dataset) == 9
    assert list(val_loader.dataset.indices) == list(val_loader_again.dataset.indices)


def test_flat_imagefolder_split_changes_with_seed(tmp_path) -> None:
    _write_flat_imagefolder(tmp_path)

    _, val_a, _ = build_imagefolder_dataloaders(
        tmp_path, batch_size=4, transform=_TRANSFORM, val_fraction=0.25, seed=1
    )
    _, val_b, _ = build_imagefolder_dataloaders(
        tmp_path, batch_size=4, transform=_TRANSFORM, val_fraction=0.25, seed=2
    )

    assert list(val_a.dataset.indices) != list(val_b.dataset.indices)


def test_presplit_imagefolder_uses_existing_train_val_directories(tmp_path) -> None:
    _write_presplit_imagefolder(tmp_path)

    train_loader, val_loader, classes = build_imagefolder_dataloaders(
        tmp_path, batch_size=4, transform=_TRANSFORM
    )

    assert classes == 2
    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 8


def test_build_imagefolder_dataloaders_rejects_invalid_arguments(tmp_path) -> None:
    _write_flat_imagefolder(tmp_path)

    with pytest.raises(ValueError):
        build_imagefolder_dataloaders(tmp_path, batch_size=0, transform=_TRANSFORM)
    with pytest.raises(ValueError):
        build_imagefolder_dataloaders(
            tmp_path, batch_size=4, transform=_TRANSFORM, val_fraction=0.0
        )
