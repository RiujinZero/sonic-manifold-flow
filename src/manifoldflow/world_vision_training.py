"""Synthetic crop records and metrics for the SigLIP identity bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from manifoldflow.world_vision import Detection, crop_detection


@dataclass(frozen=True)
class VisionCropExample:
    episode_id: str
    frame_index: int
    entity_id: str
    description: str
    image_path: Path
    box_xyxy: tuple[float, float, float, float]
    target_key: Tensor

    @property
    def frame_id(self) -> str:
        return f"{self.episode_id}:{self.frame_index}"


def load_visible_crop_examples(
    dataset_dir: str | Path,
    split: str,
    *,
    frame_stride: int = 4,
    max_examples: int = 0,
) -> list[VisionCropExample]:
    """Load visible entities only; held-out episodes remain structurally isolated."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if frame_stride <= 0 or max_examples < 0:
        raise ValueError("frame_stride must be positive and max_examples non-negative")
    root = Path(dataset_dir)
    rows = (
        (root / split / "world_frames.jsonl").read_text(encoding="utf-8").splitlines()
    )
    result: list[VisionCropExample] = []
    for line in rows:
        frame = json.loads(line)
        frame_index = int(frame["frame_index"])
        if frame_index % frame_stride:
            continue
        for entity in frame["entities"]:
            if entity["visibility"] != "visible":
                continue
            box = tuple(float(value) for value in entity["box_xyxy"])
            key = torch.tensor(entity["appearance_key"], dtype=torch.float32)
            result.append(
                VisionCropExample(
                    episode_id=str(frame["episode_id"]),
                    frame_index=frame_index,
                    entity_id=str(entity["entity_id"]),
                    description=_description(entity),
                    image_path=root / split / "images" / str(frame["image"]),
                    box_xyxy=box,
                    target_key=key,
                )
            )
            if max_examples and len(result) >= max_examples:
                return result
    return result


def open_crop(example: VisionCropExample, padding: float = 0.05):
    """Open and crop one example without retaining the source frame."""
    from PIL import Image

    with Image.open(example.image_path) as image:
        return crop_detection(
            image.convert("RGB"),
            Detection("crop", 1.0, example.box_xyxy),
            padding=padding,
        )


def frame_retrieval_accuracy(
    projected_keys: Tensor, target_keys: Tensor, frame_ids: list[str]
) -> float:
    """Measure whether each visual key selects its entity within the same frame."""
    if projected_keys.shape != target_keys.shape or projected_keys.ndim != 2:
        raise ValueError("projected_keys and target_keys must share shape [items, key]")
    if len(frame_ids) != projected_keys.shape[0]:
        raise ValueError("frame_ids must have one item per key")
    projected = torch.nn.functional.normalize(projected_keys.float(), dim=-1)
    targets = torch.nn.functional.normalize(target_keys.float(), dim=-1)
    groups: dict[str, list[int]] = {}
    for index, frame_id in enumerate(frame_ids):
        groups.setdefault(frame_id, []).append(index)
    correct = 0
    total = 0
    for indices in groups.values():
        rows = torch.tensor(indices, dtype=torch.long, device=projected.device)
        scores = projected.index_select(0, rows) @ targets.index_select(0, rows).T
        correct += int(
            (scores.argmax(dim=1) == torch.arange(len(rows), device=rows.device)).sum()
        )
        total += len(indices)
    return correct / max(total, 1)


def mean_target_cosine(projected_keys: Tensor, target_keys: Tensor) -> float:
    if projected_keys.shape != target_keys.shape:
        raise ValueError("projected_keys and target_keys must have equal shapes")
    return float(
        torch.nn.functional.cosine_similarity(
            projected_keys.float(), target_keys.float(), dim=-1
        ).mean()
    )


def temporal_reidentification_accuracy(
    projected_keys: Tensor, episode_ids: list[str], entity_ids: list[str]
) -> float:
    """Re-identify later crops against each episode's earliest entity crops."""
    if projected_keys.ndim != 2:
        raise ValueError("projected_keys must have shape [items, key]")
    if len(episode_ids) != len(entity_ids) or len(entity_ids) != len(projected_keys):
        raise ValueError("episode_ids and entity_ids must align with projected_keys")
    projected = torch.nn.functional.normalize(projected_keys.float(), dim=-1)
    episodes: dict[str, list[int]] = {}
    for row, episode_id in enumerate(episode_ids):
        episodes.setdefault(episode_id, []).append(row)
    hits = 0
    total = 0
    for rows in episodes.values():
        first_by_entity: dict[str, int] = {}
        for row in rows:
            first_by_entity.setdefault(entity_ids[row], row)
        gallery_entities = list(first_by_entity)
        if len(gallery_entities) < 2:
            continue
        gallery_rows = torch.tensor(
            [first_by_entity[item] for item in gallery_entities],
            dtype=torch.long,
            device=projected.device,
        )
        gallery = projected.index_select(0, gallery_rows)
        entity_to_column = {
            entity_id: column for column, entity_id in enumerate(gallery_entities)
        }
        for row in rows:
            if row == first_by_entity[entity_ids[row]]:
                continue
            prediction = int((projected[row] @ gallery.T).argmax())
            hits += int(prediction == entity_to_column[entity_ids[row]])
            total += 1
    return hits / max(total, 1)


def _description(entity: dict[str, object]) -> str:
    class_name = str(entity["class_name"])
    if class_name == "toy_car":
        return "toy car"
    if class_name == "occluder_box":
        return "brown box"
    if class_name == "cup":
        label = str(entity.get("label", "")).strip()
        return f"{label} cup" if label else "cup"
    return class_name
