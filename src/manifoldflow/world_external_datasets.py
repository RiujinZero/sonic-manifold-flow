"""Download and normalize external MP4 corpora for World-Gemma.

The normalized files deliberately retain source track identities and labels.  They
do not pretend that sparse human annotations are dense detector ground truth.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetAsset:
    name: str
    dataset: str
    kind: str
    url: str
    filename: str
    license_name: str
    license_url: str
    expected_bytes: int | None = None


ASSETS = {
    "perception-test-sample-annotations": DatasetAsset(
        name="perception-test-sample-annotations",
        dataset="perception-test",
        kind="annotations",
        url=(
            "https://storage.googleapis.com/dm-perception-test/zip_data/"
            "sample_annotations.zip"
        ),
        filename="sample_annotations.zip",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
    "perception-test-sample-videos": DatasetAsset(
        name="perception-test-sample-videos",
        dataset="perception-test",
        kind="videos",
        url=(
            "https://storage.googleapis.com/dm-perception-test/zip_data/"
            "sample_videos.zip"
        ),
        filename="sample_videos.zip",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
    "perception-test-train-annotations": DatasetAsset(
        name="perception-test-train-annotations",
        dataset="perception-test",
        kind="annotations",
        url=(
            "https://storage.googleapis.com/dm-perception-test/zip_data/"
            "train_annotations.zip"
        ),
        filename="train_annotations.zip",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        expected_bytes=26_250_000,
    ),
    "perception-test-train-videos": DatasetAsset(
        name="perception-test-train-videos",
        dataset="perception-test",
        kind="videos",
        url=(
            "https://storage.googleapis.com/dm-perception-test/zip_data/"
            "train_videos.zip"
        ),
        filename="train_videos.zip",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        expected_bytes=26_500_000_000,
    ),
    "perception-test-train-cut-frames": DatasetAsset(
        name="perception-test-train-cut-frames",
        dataset="perception-test",
        kind="annotations",
        url=(
            "https://storage.googleapis.com/dm-perception-test/"
            "misc/cut_frame_mapping_train.json"
        ),
        filename="cut_frame_mapping_train.json",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
    "physion-train-videos": DatasetAsset(
        name="physion-train-videos",
        dataset="physion",
        kind="videos",
        url=(
            "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/"
            "PhysionTrainMP4s.tar.gz"
        ),
        filename="PhysionTrainMP4s.tar.gz",
        license_name="MIT",
        license_url=(
            "https://github.com/cogtoolslab/physics-benchmarking-neurips2021/"
            "blob/master/LICENSE"
        ),
        expected_bytes=770_000_000,
    ),
    # LaSOT (CVPR-19): real-world YouTube video with dense per-frame manual
    # bounding boxes, addressing the gap the visual world model's R2/R3 runs
    # hit and never closed -- the synthetic training corpus had no real
    # occluder_box examples and only 99 real toy_car observations, and
    # mixing synthetic re-exposure alone cannot teach real-domain
    # appearance. Only vehicle categories are registered here (a scripted
    # LaSOT-driven scenario, not general real-domain retraining) to keep
    # the demo's scope narrow, per the actual plan.
    #
    # No formal SPDX license is published for LaSOT itself -- the dataset
    # card and homepage state only a citation requirement ("please
    # consider citing our papers"); the paper describes the source clips
    # as collected from YouTube under their original Creative Commons
    # licenses, not a single dataset-wide license grant. Recorded
    # accurately rather than assumed; worth a closer look before any
    # commercial/redistribution use, though training a detector on it for
    # research/demo purposes is exactly the benchmark's intended use.
    **{
        f"lasot-{category}": DatasetAsset(
            name=f"lasot-{category}",
            dataset="lasot",
            kind="videos",
            url=f"https://huggingface.co/datasets/l-lt/LaSOT/resolve/main/{category}.zip",
            filename=f"{category}.zip",
            license_name="Source clips under original YouTube CC licenses; no dataset-wide SPDX license published (citation requested)",
            license_url="http://vision.cs.stonybrook.edu/~lasot/",
        )
        for category in ("car", "truck", "bus", "motorcycle", "bicycle")
    },
}


def asset_catalog() -> list[dict[str, Any]]:
    """Return the auditable, allow-listed external assets."""
    return [asdict(ASSETS[name]) for name in sorted(ASSETS)]


def download_asset(
    asset_name: str,
    raw_root: str | Path,
    *,
    extract: bool = True,
    chunk_size: int = 1024 * 1024,
) -> dict[str, Any]:
    """Download one allow-listed asset with resumable partial-file handling."""
    if asset_name not in ASSETS:
        raise ValueError(f"unknown dataset asset: {asset_name}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    asset = ASSETS[asset_name]
    root = Path(raw_root) / asset.dataset
    archive_dir = root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / asset.filename
    partial_path = archive_path.with_name(archive_path.name + ".part")
    downloaded = not archive_path.exists()
    resumed_from = 0
    if downloaded:
        resumed_from = partial_path.stat().st_size if partial_path.exists() else 0
        request = urllib.request.Request(asset.url)
        if resumed_from:
            request.add_header("Range", f"bytes={resumed_from}-")
        with urllib.request.urlopen(request) as response:  # noqa: S310 - allow-list
            append = resumed_from > 0 and response.status == 206
            if not append:
                resumed_from = 0
            with partial_path.open("ab" if append else "wb") as target:
                shutil.copyfileobj(response, target, length=chunk_size)
        partial_path.replace(archive_path)
    extracted_to = None
    is_archive = zipfile.is_zipfile(archive_path) or tarfile.is_tarfile(archive_path)
    if extract and is_archive:
        extracted_to = root / "extracted" / asset_name
        extract_archive(archive_path, extracted_to)
    receipt = {
        "asset": asdict(asset),
        "archive": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "downloaded": downloaded,
        "resumed_from": resumed_from,
        "extracted_to": None if extracted_to is None else str(extracted_to),
    }
    receipt_path = archive_path.with_suffix(archive_path.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def extract_archive(
    archive_path: str | Path, output_dir: str | Path, *, workers: int = 1
) -> None:
    """Extract supported archives while rejecting path traversal members."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    archive = Path(archive_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            names = [item.filename for item in bundle.infolist()]
            _validate_members(output, names)
            if workers == 1:
                bundle.extractall(output)
                return
        chunks = [names[index::workers] for index in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(
                executor.map(
                    lambda members: _extract_zip_members(archive, output, members),
                    chunks,
                )
            )
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, mode="r:*") as bundle:
            members = bundle.getmembers()
            _validate_members(output, (item.name for item in members))
            bundle.extractall(output, members=members, filter="data")
        return
    raise ValueError(f"unsupported archive: {archive}")


def prepare_perception_test(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    annotation_file: str | Path | None = None,
    video_root: str | Path | None = None,
    cup_games_only: bool = True,
    extract_frames: bool = False,
    annotation_stride: int = 1,
    validation_fraction: float = 0.15,
    split_seed: int = 17,
) -> dict[str, Any]:
    """Normalize Perception Test tracks, questions, and optional JPEG crops."""
    if annotation_stride <= 0:
        raise ValueError("annotation_stride must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    raw = Path(raw_root)
    annotation_path = (
        _find_annotation_json(raw)
        if annotation_file is None
        else _resolve_annotation_path(raw, annotation_file)
    )
    source: dict[str, dict[str, Any]] = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )
    video_source = raw if video_root is None else Path(video_root)
    videos = {path.stem: path for path in video_source.rglob("*.mp4")}
    cut_frames = _load_cut_frames(raw)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clip_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    split_by_video: dict[str, str] = {}
    split_counts: defaultdict[str, int] = defaultdict(int)
    classes: defaultdict[str, int] = defaultdict(int)
    masked_observations = 0

    for video_id, record in sorted(source.items()):
        metadata = record["metadata"]
        if cup_games_only and not bool(metadata.get("is_cup_game")):
            continue
        selected[video_id] = record
        corpus_split = _corpus_split(video_id, validation_fraction, split_seed)
        split_by_video[video_id] = corpus_split
        split_counts[corpus_split] += 1
        roles = _perception_track_roles(record)
        relevant_tracks = [
            track
            for track in record.get("object_tracking", [])
            if _perception_class(track, roles) is not None
        ]
        video_path = videos.get(video_id)
        clip_rows.append(
            {
                "clip_id": video_id,
                "split": corpus_split,
                "source_split": metadata["split"],
                "source_video": (
                    None
                    if video_path is None
                    else str(video_path.relative_to(video_source))
                ),
                "frame_rate": metadata["frame_rate"],
                "num_frames": metadata["num_frames"],
                "resolution": metadata["resolution"],
                "is_cup_game": bool(metadata.get("is_cup_game")),
                "is_camera_moving": bool(metadata.get("is_camera_moving")),
                "track_count": len(relevant_tracks),
            }
        )
        observations_by_frame: defaultdict[int, list[dict[str, Any]]] = defaultdict(
            list
        )
        for track in relevant_tracks:
            class_name = _perception_class(track, roles)
            assert class_name is not None
            entity_id = f"{video_id}:track-{track['id']}"
            values = zip(
                track["frame_ids"],
                track["timestamps"],
                track["bounding_boxes"],
                track["is_masked"],
                strict=True,
            )
            for row_index, (frame_id, timestamp_us, box, is_masked) in enumerate(
                values
            ):
                if row_index % annotation_stride:
                    continue
                masked = bool(is_masked)
                masked_observations += int(masked)
                classes[class_name] += 1
                observations_by_frame[int(frame_id)].append(
                    {
                        "entity_id": entity_id,
                        "source_track_id": int(track["id"]),
                        "class_name": class_name,
                        "source_label": track["label"],
                        "box_xyxy": [float(value) for value in box],
                        "visibility": "occluded" if masked else "visible",
                        "is_occluder": bool(track.get("is_occluder")),
                        "timestamp_us": int(timestamp_us),
                    }
                )
        for frame_id, observations in sorted(observations_by_frame.items()):
            frame_rows.append(
                {
                    "clip_id": video_id,
                    "split": corpus_split,
                    "frame_index": frame_id,
                    "timestamp_s": frame_id / float(metadata["frame_rate"]),
                    "observations": observations,
                }
            )
        for question in record.get("mc_question", []):
            tags = {str(tag).lower() for tag in question.get("tag", [])}
            if cup_games_only and not tags.intersection(
                {"object permanence", "occlusion", "motion", "solidity"}
            ):
                continue
            answer_id = int(question["answer_id"])
            question_rows.append(
                {
                    "clip_id": video_id,
                    "split": corpus_split,
                    "frame_index": int(
                        cut_frames.get(video_id, int(metadata["num_frames"]) - 1)
                    ),
                    "question": question["question"],
                    "answer": question["options"][answer_id],
                    "options": question["options"],
                    "answer_id": answer_id,
                    "area": question.get("area"),
                    "reasoning": question.get("reasoning"),
                    "tags": question.get("tag", []),
                    "evidence_entity_ids": sorted(
                        f"{video_id}:track-{track_id}"
                        for track_id in roles["all"]
                    ),
                }
            )

    _write_jsonl(output / "clips.jsonl", clip_rows)
    _write_jsonl(output / "track_frames.jsonl", frame_rows)
    _write_jsonl(output / "questions.jsonl", question_rows)
    coco_summary = _write_perception_coco(
        selected,
        split_by_video,
        videos,
        output,
        extract_frames=extract_frames,
        annotation_stride=annotation_stride,
    )
    manifest = {
        "format_version": 1,
        "dataset": "perception-test",
        "source_annotations": str(annotation_path),
        "video_root": str(video_source),
        "license": {
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "cup_games_only": cup_games_only,
        "annotation_stride": annotation_stride,
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "splits": dict(sorted(split_counts.items())),
        "clips": len(clip_rows),
        "clips_with_video": sum(row["source_video"] is not None for row in clip_rows),
        "track_frames": len(frame_rows),
        "track_observations": sum(classes.values()),
        "masked_observations": masked_observations,
        "questions": len(question_rows),
        "clips_with_cut_frame": sum(video_id in cut_frames for video_id in selected),
        "class_observations": dict(sorted(classes.items())),
        "coco": coco_summary,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_physion(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.15,
    split_seed: int = 17,
) -> dict[str, Any]:
    """Index Physion MP4 files without duplicating the media."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    raw = Path(raw_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    scenarios: defaultdict[str, int] = defaultdict(int)
    splits: defaultdict[str, int] = defaultdict(int)
    for path in sorted(raw.rglob("*.mp4")):
        scenario = _physion_scenario(path, raw)
        split = _corpus_split(path.stem, validation_fraction, split_seed)
        scenarios[scenario] += 1
        splits[split] += 1
        rows.append(
            {
                "clip_id": path.stem,
                "split": split,
                "scenario": scenario,
                "source_video": str(path.relative_to(raw)),
            }
        )
    _write_jsonl(output / "clips.jsonl", rows)
    manifest = {
        "format_version": 1,
        "dataset": "physion",
        "license": {
            "name": "MIT",
            "url": (
                "https://github.com/cogtoolslab/"
                "physics-benchmarking-neurips2021/blob/master/LICENSE"
            ),
        },
        "clips": len(rows),
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "splits": dict(sorted(splits.items())),
        "scenarios": dict(sorted(scenarios.items())),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _perception_track_roles(record: dict[str, Any]) -> dict[str, set[int]]:
    result = {"hidden": set(), "occluder": set(), "similar": set(), "all": set()}
    for question in record.get("grounded_question", []):
        text = str(question.get("question", "")).lower()
        answers = {int(value) for value in question.get("answers", [])}
        if "track the occluded object" in text:
            result["hidden"].update(answers)
        elif "track the object that occludes" in text:
            result["occluder"].update(answers)
        elif "track these similar" in text:
            result["similar"].update(answers)
    for track in record.get("object_tracking", []):
        if track.get("is_occluder"):
            result["occluder"].add(int(track["id"]))
    result["all"] = result["hidden"] | result["occluder"] | result["similar"]
    return result


def _perception_class(
    track: dict[str, Any], roles: dict[str, set[int]]
) -> str | None:
    track_id = int(track["id"])
    label = str(track.get("label", "")).lower().replace("_", "-")
    if track_id in roles["hidden"]:
        return "ball"
    if track_id in roles["occluder"] or track_id in roles["similar"]:
        return "cup"
    if "toy-car" in label or label == "car":
        return "toy_car"
    if any(value in label for value in ("ball", "marble")):
        return "ball"
    if any(value in label for value in ("cup", "glass", "mug")):
        return "cup"
    return None


def _write_perception_coco(
    selected: dict[str, dict[str, Any]],
    split_by_video: dict[str, str],
    videos: dict[str, Path],
    output: Path,
    *,
    extract_frames: bool,
    annotation_stride: int,
) -> dict[str, Any]:
    categories = [
        {"id": 1, "name": "toy_car"},
        {"id": 2, "name": "occluder_box"},
        {"id": 3, "name": "cup"},
        {"id": 4, "name": "ball"},
    ]
    datasets: dict[str, dict[str, Any]] = {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in ("train", "validation")
    }
    if not extract_frames:
        for split, dataset in datasets.items():
            split_dir = output / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "coco.json").write_text(
                json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
            )
        return {
            "images": 0,
            "boxes": 0,
            "splits": {
                split: {"images": 0, "boxes": 0} for split in datasets
            },
        }
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("frame extraction requires the world extra") from error
    class_ids = {item["name"]: item["id"] for item in categories}
    image_ids = {"train": 1, "validation": 1}
    annotation_ids = {"train": 1, "validation": 1}
    for video_id, record in selected.items():
        video_path = videos.get(video_id)
        if video_path is None:
            continue
        metadata = record["metadata"]
        split = split_by_video[video_id]
        dataset = datasets[split]
        image_dir = output / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        height, width = (int(value) for value in metadata["resolution"])
        roles = _perception_track_roles(record)
        by_frame: defaultdict[int, list[tuple[dict[str, Any], int]]] = defaultdict(list)
        for track in record.get("object_tracking", []):
            class_name = _perception_class(track, roles)
            if class_name is None:
                continue
            for row_index, (frame_id, _box, masked) in enumerate(
                zip(
                    track["frame_ids"],
                    track["bounding_boxes"],
                    track["is_masked"],
                    strict=True,
                )
            ):
                if row_index % annotation_stride or bool(masked):
                    continue
                by_frame[int(frame_id)].append((track, row_index))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")
        try:
            for frame_id, items in sorted(by_frame.items()):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"could not decode {video_id} frame {frame_id}")
                filename = f"{video_id}-{frame_id:06d}.jpg"
                if not cv2.imwrite(str(image_dir / filename), frame):
                    raise RuntimeError(f"could not write {filename}")
                dataset["images"].append(
                    {
                        "id": image_ids[split],
                        "file_name": filename,
                        "width": width,
                        "height": height,
                        "source_video": video_path.name,
                        "frame_index": frame_id,
                    }
                )
                for track, row_index in items:
                    class_name = _perception_class(track, roles)
                    x1, y1, x2, y2 = (
                        float(value) for value in track["bounding_boxes"][row_index]
                    )
                    box = [
                        x1 * width,
                        y1 * height,
                        (x2 - x1) * width,
                        (y2 - y1) * height,
                    ]
                    dataset["annotations"].append(
                        {
                            "id": annotation_ids[split],
                            "image_id": image_ids[split],
                            "category_id": class_ids[class_name],
                            "bbox": box,
                            "area": box[2] * box[3],
                            "iscrowd": 0,
                            "entity_id": f"{video_id}:track-{track['id']}",
                            "source_label": track["label"],
                        }
                    )
                    annotation_ids[split] += 1
                image_ids[split] += 1
        finally:
            capture.release()
    split_summary = {}
    for split, dataset in datasets.items():
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "coco.json").write_text(
            json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
        )
        split_summary[split] = {
            "images": len(dataset["images"]),
            "boxes": len(dataset["annotations"]),
        }
    return {
        "images": sum(item["images"] for item in split_summary.values()),
        "boxes": sum(item["boxes"] for item in split_summary.values()),
        "splits": split_summary,
    }


def _find_annotation_json(root: Path) -> Path:
    candidates = sorted(root.rglob("all_*.json")) + sorted(root.rglob("sample.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no Perception Test all_*.json or sample.json found under {root}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "multiple annotation files found; select one with --annotation-file: "
            f"{candidates}"
        )
    return candidates[0]


def _resolve_annotation_path(root: Path, annotation_file: str | Path) -> Path:
    candidate = Path(annotation_file)
    path = candidate if candidate.is_absolute() else root / candidate
    if not path.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {path}")
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
        raise ValueError("annotation file must be inside raw_root")
    return path


def _load_cut_frames(root: Path) -> dict[str, int]:
    candidates = [
        path
        for path in sorted(root.rglob("cut_frame_mapping*.json"))
        if ".receipt." not in path.name
    ]
    if not candidates:
        return {}
    mappings: dict[str, int] = {}
    for path in candidates:
        values = json.loads(path.read_text(encoding="utf-8"))
        mappings.update(
            {
                str(key): int(value)
                for key, value in values.items()
                if int(value) >= 0
            }
        )
    return mappings


def _physion_scenario(path: Path, root: Path) -> str:
    known = {
        "collide",
        "contain",
        "dominoes",
        "drape",
        "drop",
        "link",
        "roll",
        "support",
    }
    for part in reversed(path.relative_to(root).parts[:-1]):
        normalized = part.lower().replace("_", "-")
        for scenario in known:
            if scenario in normalized:
                return scenario
    return "unknown"


def _corpus_split(clip_id: str, validation_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{clip_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if unit < validation_fraction else "train"


def _validate_members(output: Path, members: Iterable[str]) -> None:
    root = output.resolve()
    for member in members:
        target = (output / member).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"archive member escapes output directory: {member}")


def _extract_zip_members(archive: Path, output: Path, members: list[str]) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for name in members:
            info = bundle.getinfo(name)
            target = output / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
