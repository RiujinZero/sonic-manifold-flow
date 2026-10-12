import json
import zipfile

import pytest

from manifoldflow.world_external_datasets import (
    asset_catalog,
    extract_archive,
    prepare_perception_test,
    prepare_physion,
)


def _perception_record():
    frames = [0, 30, 60]
    timestamps = [0, 1_000_000, 2_000_000]
    boxes = [[0.1, 0.2, 0.2, 0.4]] * 3

    def track(track_id, label, *, occluder=False, masked=None):
        return {
            "id": track_id,
            "label": label,
            "is_occluder": int(occluder),
            "bounding_boxes": boxes,
            "initial_tracking_box": [1, 0, 0],
            "frame_ids": frames,
            "timestamps": timestamps,
            "is_masked": masked or [0, 0, 0],
        }

    return {
        "metadata": {
            "split": "train",
            "video_id": "video_1",
            "frame_rate": 30.0,
            "num_frames": 61,
            "resolution": [100, 200],
            "is_cup_game": 1,
            "is_camera_moving": 0,
        },
        "object_tracking": [
            track(1, "glass", occluder=True),
            track(2, "glass"),
            track(3, "purple-toy", masked=[0, 1, 0]),
            track(4, "person"),
        ],
        "grounded_question": [
            {"question": "Track the occluded object.", "answers": [3]},
            {
                "question": "Track the object that occludes the smaller object.",
                "answers": [1],
            },
            {"question": "Track these similar objects.", "answers": [1, 2]},
        ],
        "mc_question": [
            {
                "question": "Where is the hidden object?",
                "options": ["left", "middle", "right"],
                "answer_id": 1,
                "area": "physics",
                "reasoning": "predictive",
                "tag": ["object permanence", "occlusion"],
            },
            {
                "question": "What room is this?",
                "options": ["office", "kitchen", "garage"],
                "answer_id": 0,
                "area": "semantics",
                "reasoning": "descriptive",
                "tag": ["place recognition"],
            },
        ],
    }


def test_prepare_perception_test_maps_grounded_roles(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "all_train.json").write_text(
        json.dumps({"video_1": _perception_record()}), encoding="utf-8"
    )
    (raw / "cut_frame_mapping_train.json").write_text(
        json.dumps({"video_1": 50}), encoding="utf-8"
    )
    output = tmp_path / "prepared"

    result = prepare_perception_test(raw, output)
    frames = [
        json.loads(line)
        for line in (output / "track_frames.jsonl").read_text().splitlines()
    ]
    questions = [
        json.loads(line)
        for line in (output / "questions.jsonl").read_text().splitlines()
    ]

    assert result["clips"] == 1
    assert result["clips_with_video"] == 0
    assert result["class_observations"] == {"ball": 3, "cup": 6}
    assert result["masked_observations"] == 1
    assert len(questions) == 1
    middle = next(row for row in frames if row["frame_index"] == 30)
    hidden = next(
        item for item in middle["observations"] if item["class_name"] == "ball"
    )
    assert hidden["visibility"] == "occluded"
    assert questions[0]["answer"] == "middle"
    assert questions[0]["frame_index"] == 50
    assert result["clips_with_cut_frame"] == 1
    assert sum(result["splits"].values()) == 1
    assert (output / "train" / "coco.json").exists()
    assert (output / "validation" / "coco.json").exists()


def test_prepare_perception_test_accepts_separate_video_root(tmp_path) -> None:
    raw = tmp_path / "annotations"
    videos = tmp_path / "native-videos"
    raw.mkdir()
    videos.mkdir()
    (raw / "all_train.json").write_text(
        json.dumps({"video_1": _perception_record()}), encoding="utf-8"
    )
    (videos / "video_1.mp4").touch()

    result = prepare_perception_test(raw, tmp_path / "prepared", video_root=videos)

    assert result["clips_with_video"] == 1
    assert result["video_root"] == str(videos)


def test_prepare_physion_indexes_mp4s_by_scenario(tmp_path) -> None:
    raw = tmp_path / "raw"
    (raw / "train" / "contain").mkdir(parents=True)
    (raw / "train" / "roll").mkdir(parents=True)
    (raw / "train" / "contain" / "one.mp4").touch()
    (raw / "train" / "roll" / "two.mp4").touch()

    result = prepare_physion(raw, tmp_path / "prepared")

    assert result["clips"] == 2
    assert result["scenarios"] == {"contain": 1, "roll": 1}
    assert sum(result["splits"].values()) == 2


def test_archive_extraction_rejects_parent_traversal(tmp_path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="escapes output"):
        extract_archive(archive, tmp_path / "output")


def test_parallel_zip_extraction_skips_complete_files(tmp_path) -> None:
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("one/a.txt", "alpha")
        bundle.writestr("two/b.txt", "beta")
    output = tmp_path / "output"

    extract_archive(archive, output, workers=2)
    extract_archive(archive, output, workers=2)

    assert (output / "one" / "a.txt").read_text() == "alpha"
    assert (output / "two" / "b.txt").read_text() == "beta"


def test_catalog_records_license_for_every_asset() -> None:
    catalog = asset_catalog()
    assert catalog
    assert all(item["license_name"] and item["license_url"] for item in catalog)
