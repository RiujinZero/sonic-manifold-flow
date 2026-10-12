"""Build a source-balanced, interleaved train corpus from real and synthetic COCO splits.

The real Perception Test cup corpus has almost no toy_car/occluder_box
coverage, so fine-tuning on it alone erases those classes from the shared
backbone (see docs/World-Gemma-Dev-Journal.md, 2026-09-01 R2/R3). This tool
mixes it with a matched sample of the synthetic corpus, which is roughly
class-balanced, so real-domain adaptation stops costing synthetic-domain
recall.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def _load_split(root: Path, split: str) -> dict[str, Any]:
    path = root / split / "coco.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing COCO split: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _class_counts(
    categories: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> dict[str, int]:
    id_to_name = {category["id"]: category["name"] for category in categories}
    counts = {name: 0 for name in id_to_name.values()}
    for annotation in annotations:
        counts[id_to_name[annotation["category_id"]]] += 1
    return counts


def _link_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def _link_validation(real_root: Path, output_dir: Path) -> bool:
    source = real_root / "validation"
    if not (source / "coco.json").is_file():
        return False
    target = output_dir / "validation"
    if target.exists() or target.is_symlink():
        return True
    try:
        os.symlink(source.resolve(), target, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)
    return True


def build_mixed_corpus(
    real_root: Path,
    synthetic_root: Path,
    output_dir: Path,
    *,
    synthetic_ratio: float = 1.0,
    seed: int = 17,
) -> dict[str, Any]:
    real = _load_split(real_root, "train")
    synthetic = _load_split(synthetic_root, "train")
    if real["categories"] != synthetic["categories"]:
        raise ValueError(
            "real and synthetic train splits must share identical categories, "
            f"got {real['categories']} vs {synthetic['categories']}"
        )
    categories = real["categories"]

    target_synthetic = round(len(real["images"]) * synthetic_ratio)
    if target_synthetic > len(synthetic["images"]):
        raise ValueError(
            f"synthetic train split has only {len(synthetic['images'])} images, "
            f"need {target_synthetic} for synthetic_ratio={synthetic_ratio}"
        )

    rng = random.Random(seed)
    sampled_synthetic = rng.sample(synthetic["images"], target_synthetic)
    sampled_ids = {image["id"] for image in sampled_synthetic}
    synthetic_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in synthetic["annotations"]:
        if annotation["image_id"] in sampled_ids:
            synthetic_by_image.setdefault(annotation["image_id"], []).append(
                annotation
            )
    real_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in real["annotations"]:
        real_by_image.setdefault(annotation["image_id"], []).append(annotation)

    tagged = [("real", image) for image in real["images"]] + [
        ("synthetic", image) for image in sampled_synthetic
    ]
    rng.shuffle(tagged)  # interleave sources within the merged corpus itself

    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    seen_filenames: set[str] = set()
    images_dir = output_dir / "train" / "images"

    for source, image in tagged:
        file_name = image["file_name"]
        if file_name in seen_filenames:
            raise ValueError(
                f"filename collision between sources for {file_name!r}"
            )
        seen_filenames.add(file_name)

        source_root = real_root if source == "real" else synthetic_root
        _link_image(
            source_root / "train" / "images" / file_name, images_dir / file_name
        )

        old_id = image["id"]
        new_id = next_image_id
        next_image_id += 1
        merged_images.append({**image, "id": new_id, "source": source})

        by_image = real_by_image if source == "real" else synthetic_by_image
        for annotation in by_image.get(old_id, []):
            merged_annotations.append(
                {
                    **annotation,
                    "id": next_annotation_id,
                    "image_id": new_id,
                }
            )
            next_annotation_id += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "coco.json").write_text(
        json.dumps(
            {
                "images": merged_images,
                "annotations": merged_annotations,
                "categories": categories,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_linked = _link_validation(real_root, output_dir)

    manifest = {
        "real_root": str(real_root),
        "synthetic_root": str(synthetic_root),
        "synthetic_ratio": synthetic_ratio,
        "seed": seed,
        "real_images": len(real["images"]),
        "synthetic_images_available": len(synthetic["images"]),
        "synthetic_images_sampled": len(sampled_synthetic),
        "merged_images": len(merged_images),
        "merged_annotations": len(merged_annotations),
        "validation_linked": validation_linked,
        "class_counts": {
            "real": _class_counts(categories, real["annotations"]),
            "synthetic_sampled": _class_counts(
                categories,
                [a for ann_list in synthetic_by_image.values() for a in ann_list],
            ),
            "merged": _class_counts(categories, merged_annotations),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        type=Path,
        default=Path("artifacts/world-corpora/perception-test-train-cups"),
        help="real-domain COCO corpus root (train/ and validation/ subdirs)",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=Path("artifacts/world-sim"),
        help="synthetic COCO corpus root (train/ subdir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/world-corpora/mixed-perception-synthetic"),
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=1.0,
        help="synthetic:real image count ratio for the merged train split",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_mixed_corpus(
        args.real,
        args.synthetic,
        args.output,
        synthetic_ratio=args.synthetic_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
