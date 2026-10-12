"""Clip-separated synthetic dataset export for Gemma and YOLOX training."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from manifoldflow.world_renderer import RenderConfig, TabletopRenderer
from manifoldflow.world_simulator import (
    ScenarioKind,
    SimEntity,
    SimEpisode,
    SimFrame,
    SimulatorConfig,
    TabletopSimulator,
)

CLASSES = ("toy_car", "occluder_box", "cup", "ball")


@dataclass(frozen=True)
class DatasetBuildResult:
    output_dir: Path
    episodes: int
    frames: int
    questions: int
    split_episodes: dict[str, int]


def split_episode_ids(episode_ids: list[str], seed: int) -> dict[str, tuple[str, ...]]:
    if len(episode_ids) < 3:
        raise ValueError("at least three episodes are required for clip splits")
    shuffled = list(episode_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = max(1, round(len(shuffled) * 0.70))
    validation_end = max(train_end + 1, round(len(shuffled) * 0.85))
    validation_end = min(validation_end, len(shuffled) - 1)
    return {
        "train": tuple(shuffled[:train_end]),
        "validation": tuple(shuffled[train_end:validation_end]),
        "test": tuple(shuffled[validation_end:]),
    }


def generate_synthetic_dataset(
    output_dir: str | Path,
    *,
    episode_count: int = 100,
    seed: int = 17,
    simulator_config: SimulatorConfig | None = None,
    render_config: RenderConfig | None = None,
) -> DatasetBuildResult:
    if episode_count < 3:
        raise ValueError("episode_count must be at least three")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    simulator = TabletopSimulator(simulator_config)
    renderer = TabletopRenderer(render_config)
    episodes = [
        simulator.generate(
            ScenarioKind.CAR_OCCLUSION if index % 2 == 0 else ScenarioKind.CUP_SHUFFLE,
            seed=seed + index,
            episode_id=f"episode-{index:06d}",
        )
        for index in range(episode_count)
    ]
    by_id = {item.episode_id: item for item in episodes}
    splits = split_episode_ids(list(by_id), seed)
    total_frames = 0
    total_questions = 0
    for split, episode_ids in splits.items():
        split_dir = output / split
        image_dir = split_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        frame_rows: list[dict[str, object]] = []
        question_rows: list[dict[str, object]] = []
        coco_images: list[dict[str, object]] = []
        coco_annotations: list[dict[str, object]] = []
        annotation_id = 1
        image_id = 1
        for episode_id in episode_ids:
            episode = by_id[episode_id]
            for frame in episode.frames:
                rendered = renderer.render(episode, frame)
                filename = f"{episode_id}-{frame.frame_index:04d}.jpg"
                rendered.image.save(
                    image_dir / filename,
                    format="JPEG",
                    quality=renderer.config.jpeg_quality,
                    optimize=True,
                )
                frame_rows.append(_frame_row(episode, frame, filename))
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": filename,
                        "width": renderer.config.width,
                        "height": renderer.config.height,
                        "episode_id": episode_id,
                        "frame_index": frame.frame_index,
                    }
                )
                for annotation in rendered.annotations:
                    class_name = str(annotation["class_name"])
                    bbox = annotation["bbox_xywh"]
                    coco_annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": CLASSES.index(class_name) + 1,
                            "bbox": bbox,
                            "area": float(bbox[2] * bbox[3]),  # type: ignore[index,operator]
                            "iscrowd": 0,
                            "entity_id": annotation["entity_id"],
                            "tracker_id": annotation["tracker_id"],
                        }
                    )
                    annotation_id += 1
                image_id += 1
            for question in episode.questions:
                question_rows.append(
                    {
                        "episode_id": episode_id,
                        "scenario": episode.scenario.value,
                        "frame_index": question.frame_index,
                        "question": _split_question(
                            question.question, split=split, seed=episode.seed
                        ),
                        "answer": question.answer,
                        "evidence_entity_ids": question.evidence_entity_ids,
                        "category": question.category,
                    }
                )
        _write_jsonl(split_dir / "world_frames.jsonl", frame_rows)
        _write_jsonl(split_dir / "questions.jsonl", question_rows)
        (split_dir / "coco.json").write_text(
            json.dumps(
                {
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": [
                        {"id": index + 1, "name": name}
                        for index, name in enumerate(CLASSES)
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        total_frames += len(frame_rows)
        total_questions += len(question_rows)
    manifest = {
        "format_version": 1,
        "latent_source": "oracle_semantic_identity_seed",
        "latent_dimension": 128,
        "question_templates": "disjoint_by_split",
        "seed": seed,
        "episode_count": episode_count,
        "classes": CLASSES,
        "simulator": asdict(simulator.config),
        "renderer": asdict(renderer.config),
        "splits": {name: list(values) for name, values in splits.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return DatasetBuildResult(
        output,
        episode_count,
        total_frames,
        total_questions,
        {name: len(values) for name, values in splits.items()},
    )


def _frame_row(
    episode: SimEpisode, frame: SimFrame, filename: str
) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "scenario": episode.scenario.value,
        "seed": episode.seed,
        "frame_index": frame.frame_index,
        "timestamp": frame.timestamp,
        "image": filename,
        "entities": [_entity_row(item) for item in frame.entities],
        "relations": [asdict(item) for item in frame.relations],
        "events": frame.events,
        "metadata": episode.metadata,
    }


def _entity_row(entity: SimEntity) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "tracker_id": entity.tracker_id,
        "class_name": entity.class_name,
        "box_xyxy": entity.box_xyxy,
        "color": entity.color,
        "visibility": entity.visibility.value,
        "zone": entity.zone,
        "label": entity.label,
        "velocity": entity.velocity,
        "appearance_key": (
            None if entity.appearance_key is None else entity.appearance_key.tolist()
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _split_question(question: str, *, split: str, seed: int) -> str:
    """Use held-out language templates without changing task semantics."""
    templates = {
        "Where is the car?": {
            "train": ("Where is the car?", "Which zone is the toy car in?"),
            "validation": ("Can you locate the toy car?",),
            "test": ("What is the toy car's current position?",),
        },
        "Is the car visible?": {
            "train": ("Is the car visible?", "Can you see the toy car?"),
            "validation": ("Is the toy car currently in view?",),
            "test": ("Does the camera currently show the car?",),
        },
        "What is hiding the car?": {
            "train": ("What is hiding the car?", "What blocks the car from view?"),
            "validation": ("Which object occludes the toy car?",),
            "test": ("What object is the car concealed behind?",),
        },
        "Which cup contains the ball?": {
            "train": (
                "Which cup contains the ball?",
                "What color cup is holding the ball?",
            ),
            "validation": ("Identify the cup that has the ball.",),
            "test": ("Under which colored cup is the ball hidden?",),
        },
        "Where is the ball?": {
            "train": ("Where is the ball?", "Which zone is the ball in?"),
            "validation": ("Can you locate the ball?",),
            "test": ("What is the ball's current position?",),
        },
    }
    if question.startswith("Where is the ") and question.endswith(" cup?"):
        color = question.removeprefix("Where is the ").removesuffix(" cup?")
        color_templates = {
            "train": (
                f"Where is the {color} cup?",
                f"Which zone holds the {color} cup?",
            ),
            "validation": (f"Can you locate the {color} cup?",),
            "test": (f"What is the {color} cup's current position?",),
        }
        choices = color_templates[split]
    else:
        choices = templates.get(question, {}).get(split, (question,))
    return choices[seed % len(choices)]
