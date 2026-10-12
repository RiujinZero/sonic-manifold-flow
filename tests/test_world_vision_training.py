import json

import torch
from PIL import Image

from manifoldflow.world_vision_training import (
    frame_retrieval_accuracy,
    load_visible_crop_examples,
    open_crop,
    temporal_reidentification_accuracy,
)


def test_visible_crop_loader_filters_and_crops(tmp_path) -> None:
    split = tmp_path / "train"
    images = split / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (100, 50), "white").save(images / "frame.jpg")
    entities = [
        {
            "entity_id": "episode:car",
            "class_name": "toy_car",
            "box_xyxy": [0.1, 0.2, 0.5, 0.8],
            "visibility": "visible",
            "appearance_key": [1.0, 0.0],
            "label": "",
        },
        {
            "entity_id": "episode:ball",
            "class_name": "ball",
            "box_xyxy": [0.6, 0.2, 0.8, 0.4],
            "visibility": "contained",
            "appearance_key": [0.0, 1.0],
            "label": "",
        },
    ]
    row = {
        "episode_id": "episode",
        "frame_index": 0,
        "image": "frame.jpg",
        "entities": entities,
    }
    (split / "world_frames.jsonl").write_text(json.dumps(row) + "\n")

    examples = load_visible_crop_examples(tmp_path, "train")

    assert len(examples) == 1
    assert examples[0].description == "toy car"
    assert open_crop(examples[0], padding=0).size == (40, 30)


def test_frame_retrieval_accuracy_uses_only_frame_candidates() -> None:
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    projected = targets.clone()

    accuracy = frame_retrieval_accuracy(
        projected, targets, ["frame-a", "frame-a", "frame-b"]
    )

    assert accuracy == 1.0


def test_temporal_reidentification_uses_earliest_entity_gallery() -> None:
    projected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])

    accuracy = temporal_reidentification_accuracy(
        projected,
        ["episode", "episode", "episode", "episode"],
        ["car", "box", "car", "box"],
    )

    assert accuracy == 1.0
