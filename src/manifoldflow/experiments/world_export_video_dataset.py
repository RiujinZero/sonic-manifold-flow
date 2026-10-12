"""Export timestamped real-video annotations as a small COCO adaptation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from manifoldflow.world_dataset import CLASSES
from manifoldflow.world_video import load_annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir")
    parser.add_argument("annotation_dir")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--validation-clips",
        nargs="+",
        default=(),
        help="Video stems reserved entirely for validation",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def _coco_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]


def _dataset_split(data: dict[str, Any], stem: str, validation: set[str]) -> str:
    annotated = data.get("split")
    if annotated is not None and annotated not in {"train", "validation"}:
        raise ValueError(f"invalid annotation split for {stem}: {annotated}")
    cli_split = "validation" if stem in validation else "train"
    if annotated is not None and stem in validation and annotated != cli_split:
        raise ValueError(
            f"annotation split for {stem} conflicts with --validation-clips"
        )
    return str(annotated or cli_split)


def export(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Video dataset export requires opencv-python-headless"
        ) from error
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    video_dir = Path(args.video_dir)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    validation = set(args.validation_clips)
    datasets = {
        split: {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": index + 1, "name": name} for index, name in enumerate(CLASSES)
            ],
        }
        for split in ("train", "validation")
    }
    image_ids = {"train": 1, "validation": 1}
    annotation_ids = {"train": 1, "validation": 1}
    class_ids = {name: index + 1 for index, name in enumerate(CLASSES)}
    clip_counts: dict[str, dict[str, int]] = {}

    for annotation_path in sorted(annotation_dir.glob("*.annotations.json")):
        data = load_annotations(annotation_path)
        video_path = video_dir / str(data["video"])
        stem = video_path.stem
        split = _dataset_split(data, stem, validation)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        clip_counts[stem] = {"images": 0, "boxes": 0, "split": split}
        try:
            export_frames = [*data["checkpoints"], *data.get("training_frames", [])]
            for checkpoint in sorted(
                export_frames, key=lambda item: float(item["timestamp_s"])
            ):
                timestamp = float(checkpoint["timestamp_s"])
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"could not decode {video_path.name} at {timestamp:.3f}s"
                    )
                image_id = image_ids[split]
                image_ids[split] += 1
                filename = f"{stem}-{round(timestamp * 1000):06d}.jpg"
                target = output_dir / split / "images" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(target), image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                ):
                    raise RuntimeError(f"could not write frame: {target}")
                datasets[split]["images"].append(
                    {
                        "id": image_id,
                        "file_name": filename,
                        "width": width,
                        "height": height,
                        "source_video": video_path.name,
                        "timestamp_s": timestamp,
                    }
                )
                clip_counts[stem]["images"] += 1
                for entity in checkpoint.get("entities", []):
                    box = entity.get("box")
                    class_name = entity.get("class")
                    if (
                        box is None
                        or entity.get("visibility", "visible") != "visible"
                        or class_name not in class_ids
                    ):
                        continue
                    coco_box = _coco_box(box, width, height)
                    datasets[split]["annotations"].append(
                        {
                            "id": annotation_ids[split],
                            "image_id": image_id,
                            "category_id": class_ids[class_name],
                            "bbox": coco_box,
                            "area": coco_box[2] * coco_box[3],
                            "iscrowd": 0,
                            "segmentation": [],
                            "entity_id": entity["id"],
                        }
                    )
                    annotation_ids[split] += 1
                    clip_counts[stem]["boxes"] += 1
        finally:
            capture.release()

    for split, dataset in datasets.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "coco.json").write_text(
            json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "clips": clip_counts,
        "train_images": len(datasets["train"]["images"]),
        "train_boxes": len(datasets["train"]["annotations"]),
        "validation_images": len(datasets["validation"]["images"]),
        "validation_boxes": len(datasets["validation"]["annotations"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(export(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
