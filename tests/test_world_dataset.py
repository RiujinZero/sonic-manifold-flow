import json

import torch

from manifoldflow.world_dataset import generate_synthetic_dataset
from manifoldflow.world_renderer import RenderConfig
from manifoldflow.world_simulator import SimulatorConfig
from manifoldflow.world_synthetic_training import (
    collate_oracle_examples,
    load_oracle_examples,
)


def test_dataset_export_is_clip_separated_and_coco_compatible(tmp_path) -> None:
    result = generate_synthetic_dataset(
        tmp_path,
        episode_count=6,
        seed=22,
        simulator_config=SimulatorConfig(
            frames_per_episode=16, ambiguous_probability=0.2
        ),
        render_config=RenderConfig(width=160, height=90, domain_randomization=False),
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    split_sets = [set(values) for values in manifest["splits"].values()]

    assert result.frames == 96
    assert all(
        split_sets[index].isdisjoint(split_sets[other])
        for index in range(3)
        for other in range(index + 1, 3)
    )
    assert set.union(*split_sets) == {f"episode-{index:06d}" for index in range(6)}
    for split in ("train", "validation", "test"):
        coco = json.loads((tmp_path / split / "coco.json").read_text())
        assert [item["name"] for item in coco["categories"]] == [
            "toy_car",
            "occluder_box",
            "cup",
            "ball",
        ]
        assert coco["images"]
        assert (tmp_path / split / "questions.jsonl").stat().st_size > 0
        assert all(
            (tmp_path / split / "images" / item["file_name"]).exists()
            for item in coco["images"]
        )


def test_oracle_world_rows_include_latents_relations_and_events(tmp_path) -> None:
    generate_synthetic_dataset(
        tmp_path,
        episode_count=4,
        simulator_config=SimulatorConfig(
            frames_per_episode=16, ambiguous_probability=0.0
        ),
        render_config=RenderConfig(width=160, height=90),
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "train" / "world_frames.jsonl").read_text().splitlines()
    ]

    assert all(
        len(entity["appearance_key"]) == 128
        for row in rows
        for entity in row["entities"]
    )
    assert any(row["relations"] for row in rows)
    assert any(row["events"] for row in rows)


def test_language_templates_are_held_out_by_split(tmp_path) -> None:
    generate_synthetic_dataset(
        tmp_path,
        episode_count=12,
        simulator_config=SimulatorConfig(frames_per_episode=16),
        render_config=RenderConfig(width=160, height=90),
    )
    questions = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (tmp_path / split / "questions.jsonl").read_text().splitlines()
        ]
        questions[split] = {row["question"] for row in rows}

    assert questions["train"].isdisjoint(questions["validation"])
    assert questions["train"].isdisjoint(questions["test"])
    assert questions["validation"].isdisjoint(questions["test"])


def test_oracle_loader_builds_padded_latent_port_batches(tmp_path) -> None:
    generate_synthetic_dataset(
        tmp_path,
        episode_count=8,
        simulator_config=SimulatorConfig(
            frames_per_episode=16, ambiguous_probability=0.0
        ),
        render_config=RenderConfig(width=160, height=90),
    )
    examples = load_oracle_examples(tmp_path, "train")
    car_example = next(example for example in examples if len(example.entity_ids) == 2)
    cup_example = next(example for example in examples if len(example.entity_ids) == 4)
    batch = collate_oracle_examples([car_example, cup_example])

    assert batch.appearance_keys.shape == (2, 4, 128)
    assert batch.state_values.shape == (2, 4, 256)
    assert batch.memory_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert not batch.appearance_keys[0, 2:].any()
    assert batch.evidence_mask.any()
    assert torch.all(
        batch.state_values[:, :, 24:33].sum(dim=-1)[batch.memory_mask] == 1
    )
    populated = batch.state_values[batch.memory_mask]
    assert torch.all((populated[:, 4:8] >= 0) & (populated[:, 4:8] <= 1))
    assert populated[:, 4:8].max() > 0.1
    assert torch.allclose(populated[:, 0], (populated[:, 4] + populated[:, 6]) / 2)
    assert torch.allclose(populated[:, 1], (populated[:, 5] + populated[:, 7]) / 2)
