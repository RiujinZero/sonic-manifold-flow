import json
from pathlib import Path

import pytest

from manifoldflow.experiments.world_mix_corpus import build_mixed_corpus

CATEGORIES = [
    {"id": 1, "name": "toy_car"},
    {"id": 2, "name": "occluder_box"},
    {"id": 3, "name": "cup"},
    {"id": 4, "name": "ball"},
]


def _write_corpus(
    root: Path,
    split: str,
    *,
    count: int,
    prefix: str,
    categories: list[dict] = CATEGORIES,
) -> None:
    images_dir = root / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for index in range(count):
        file_name = f"{prefix}-{index:03d}.jpg"
        (images_dir / file_name).write_bytes(b"fake-jpeg")
        images.append({"id": index + 1, "file_name": file_name, "width": 10, "height": 10})
        category_id = categories[index % len(categories)]["id"]
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "category_id": category_id,
                "bbox": [0, 0, 1, 1],
                "area": 1,
                "iscrowd": 0,
            }
        )
    (root / split).mkdir(parents=True, exist_ok=True)
    (root / split / "coco.json").write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}
        )
    )


@pytest.fixture
def corpora(tmp_path: Path) -> tuple[Path, Path]:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    _write_corpus(real_root, "train", count=5, prefix="real")
    _write_corpus(real_root, "validation", count=2, prefix="real-val")
    _write_corpus(synthetic_root, "train", count=20, prefix="synth")
    return real_root, synthetic_root


def test_merge_counts_and_interleaving(tmp_path: Path, corpora: tuple[Path, Path]) -> None:
    real_root, synthetic_root = corpora
    output_dir = tmp_path / "mixed"
    manifest = build_mixed_corpus(
        real_root, synthetic_root, output_dir, synthetic_ratio=1.0, seed=17
    )

    assert manifest["real_images"] == 5
    assert manifest["synthetic_images_sampled"] == 5
    assert manifest["merged_images"] == 10

    merged = json.loads((output_dir / "train" / "coco.json").read_text())
    assert len(merged["images"]) == 10
    assert merged["categories"] == CATEGORIES

    sources = [image["source"] for image in merged["images"]]
    assert sources.count("real") == 5
    assert sources.count("synthetic") == 5
    # Interleaved, not grouped: both sources appear before the final entry.
    assert sources[:-1] != sorted(sources[:-1])

    image_ids = {image["id"] for image in merged["images"]}
    assert image_ids == set(range(1, 11))
    for annotation in merged["annotations"]:
        assert annotation["image_id"] in image_ids

    for image in merged["images"]:
        linked = output_dir / "train" / "images" / image["file_name"]
        assert linked.exists()


def test_validation_is_linked_unmodified(
    tmp_path: Path, corpora: tuple[Path, Path]
) -> None:
    real_root, synthetic_root = corpora
    output_dir = tmp_path / "mixed"
    build_mixed_corpus(real_root, synthetic_root, output_dir, seed=17)

    merged_validation = json.loads(
        (output_dir / "validation" / "coco.json").read_text()
    )
    real_validation = json.loads(
        (real_root / "validation" / "coco.json").read_text()
    )
    assert merged_validation == real_validation


def test_deterministic_with_same_seed(tmp_path: Path, corpora: tuple[Path, Path]) -> None:
    real_root, synthetic_root = corpora
    first = build_mixed_corpus(
        real_root, synthetic_root, tmp_path / "mixed-a", seed=42
    )
    second = build_mixed_corpus(
        real_root, synthetic_root, tmp_path / "mixed-b", seed=42
    )
    first_images = json.loads((tmp_path / "mixed-a" / "train" / "coco.json").read_text())
    second_images = json.loads((tmp_path / "mixed-b" / "train" / "coco.json").read_text())
    assert [i["file_name"] for i in first_images["images"]] == [
        i["file_name"] for i in second_images["images"]
    ]
    assert first["class_counts"] == second["class_counts"]


def test_category_mismatch_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    other_categories = [{"id": 1, "name": "widget"}]
    _write_corpus(real_root, "train", count=2, prefix="real")
    _write_corpus(
        synthetic_root, "train", count=2, prefix="synth", categories=other_categories
    )

    with pytest.raises(ValueError, match="categories"):
        build_mixed_corpus(real_root, synthetic_root, tmp_path / "mixed")


def test_insufficient_synthetic_pool_is_rejected(
    tmp_path: Path, corpora: tuple[Path, Path]
) -> None:
    real_root, synthetic_root = corpora
    with pytest.raises(ValueError, match="synthetic train split has only"):
        build_mixed_corpus(
            real_root,
            synthetic_root,
            tmp_path / "mixed",
            synthetic_ratio=10.0,
        )


def test_filename_collision_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    _write_corpus(real_root, "train", count=2, prefix="shared")
    _write_corpus(synthetic_root, "train", count=2, prefix="shared")

    with pytest.raises(ValueError, match="collision"):
        build_mixed_corpus(real_root, synthetic_root, tmp_path / "mixed", seed=1)
