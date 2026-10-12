"""Load synthetic episodes as padded oracle-memory training examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.world_dataset import CLASSES
from manifoldflow.world_types import ZONES

_VISIBILITY = {
    "visible": 0.0,
    "occluded": 0.25,
    "contained": 0.5,
    "missing": 0.75,
    "unknown": 1.0,
}
_RELATIONS = {"occluded_by": 0, "contained_by": 1}


@dataclass(frozen=True)
class OracleWorldExample:
    episode_id: str
    frame_index: int
    question: str
    answer: str
    entity_ids: tuple[str, ...]
    appearance_keys: Tensor
    state_values: Tensor
    evidence_mask: Tensor
    category: str


@dataclass(frozen=True)
class OracleWorldBatch:
    questions: tuple[str, ...]
    answers: tuple[str, ...]
    entity_ids: tuple[tuple[str, ...], ...]
    appearance_keys: Tensor
    state_values: Tensor
    memory_mask: Tensor
    evidence_mask: Tensor


def load_oracle_examples(
    dataset_dir: str | Path, split: str
) -> list[OracleWorldExample]:
    """Join questions to perfect simulator state for latent-port stage one."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    root = Path(dataset_dir)
    # Validate that this is a generated corpus. Geometry in world_frames.jsonl is
    # already normalized and must not be divided by renderer dimensions again.
    json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frame_rows = _read_jsonl(root / split / "world_frames.jsonl")
    frames = {
        (str(row["episode_id"]), int(row["frame_index"])): row for row in frame_rows
    }
    examples: list[OracleWorldExample] = []
    for question in _read_jsonl(root / split / "questions.jsonl"):
        key = (str(question["episode_id"]), int(question["frame_index"]))
        frame = frames[key]
        entities = list(frame["entities"])
        entity_ids = tuple(str(entity["entity_id"]) for entity in entities)
        evidence_ids = {str(value) for value in question["evidence_entity_ids"]}
        examples.append(
            OracleWorldExample(
                episode_id=key[0],
                frame_index=key[1],
                question=str(question["question"]),
                answer=str(question["answer"]),
                entity_ids=entity_ids,
                appearance_keys=torch.tensor(
                    [entity["appearance_key"] for entity in entities],
                    dtype=torch.float32,
                ),
                state_values=torch.stack(
                    [_state_vector(entity, frame) for entity in entities]
                ),
                evidence_mask=torch.tensor(
                    [entity_id in evidence_ids for entity_id in entity_ids],
                    dtype=torch.bool,
                ),
                category=str(question["category"]),
            )
        )
    return examples


def collate_oracle_examples(
    examples: list[OracleWorldExample],
) -> OracleWorldBatch:
    if not examples:
        raise ValueError("examples cannot be empty")
    batch_size = len(examples)
    max_entities = max(example.appearance_keys.shape[0] for example in examples)
    keys = torch.zeros(batch_size, max_entities, 128)
    states = torch.zeros(batch_size, max_entities, 256)
    memory_mask = torch.zeros(batch_size, max_entities, dtype=torch.bool)
    evidence_mask = torch.zeros_like(memory_mask)
    for index, example in enumerate(examples):
        count = example.appearance_keys.shape[0]
        keys[index, :count] = example.appearance_keys
        states[index, :count] = example.state_values
        memory_mask[index, :count] = True
        evidence_mask[index, :count] = example.evidence_mask
    return OracleWorldBatch(
        questions=tuple(example.question for example in examples),
        answers=tuple(example.answer for example in examples),
        entity_ids=tuple(example.entity_ids for example in examples),
        appearance_keys=keys,
        state_values=states,
        memory_mask=memory_mask,
        evidence_mask=evidence_mask,
    )


def _state_vector(
    entity: dict[str, Any],
    frame: dict[str, Any],
) -> Tensor:
    vector = torch.zeros(256, dtype=torch.float32)
    x1, y1, x2, y2 = (float(value) for value in entity["box_xyxy"])
    vx, vy = (float(value) for value in entity["velocity"])
    vector[0:2] = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
    vector[2:4] = torch.tensor([vx, vy])
    vector[4:8] = torch.tensor([x1, y1, x2, y2])
    vector[8] = 1.0
    vector[9] = _VISIBILITY[str(entity["visibility"])]
    vector[10] = min(float(frame["frame_index"]) / 64.0, 1.0)
    vector[11] = vector[10]
    entity_id = str(entity["entity_id"])
    relation_count = 0
    for relation in frame["relations"]:
        relation_type = str(relation["relation"])
        relation_index = _RELATIONS.get(relation_type)
        if relation_index is None:
            continue
        if str(relation["subject_id"]) == entity_id:
            vector[16 + relation_index] = float(relation["confidence"])
            relation_count += 1
        if str(relation["target_id"]) == entity_id:
            vector[20 + relation_index] = float(relation["confidence"])
            relation_count += 1
    vector[12] = min(relation_count / 4.0, 1.0)
    zone_name = str(entity["zone"])
    if zone_name in ZONES:
        vector[24 + ZONES.index(zone_name)] = 1.0
    vector[33 + CLASSES.index(str(entity["class_name"]))] = 1.0
    return vector


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
