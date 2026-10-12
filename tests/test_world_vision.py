import pytest
import torch
from PIL import Image

from manifoldflow.world_vision import Detection, MultiClassByteTracker, crop_detection


def test_tracker_fallback_preserves_ids_across_latest_frames() -> None:
    tracker = MultiClassByteTracker(("cup", "ball"), match_threshold=0.8)

    first = tracker.update(
        [
            Detection("cup", 0.9, (0.1, 0.1, 0.3, 0.4)),
            Detection("ball", 0.9, (0.7, 0.6, 0.8, 0.7)),
        ]
    )
    second = tracker.update(
        [
            Detection("cup", 0.9, (0.12, 0.1, 0.32, 0.4)),
            Detection("ball", 0.9, (0.68, 0.6, 0.78, 0.7)),
        ]
    )

    assert [item.tracker_id for item in first] == [1, 2]
    assert [item.tracker_id for item in second] == [1, 2]


def test_tracker_supports_class_specific_confidence_gates() -> None:
    tracker = MultiClassByteTracker(
        ("cup", "ball"),
        track_threshold=0.5,
        class_track_thresholds={"ball": 0.1},
    )

    tracked = tracker.update(
        [
            Detection("cup", 0.2, (0.1, 0.1, 0.3, 0.4)),
            Detection("ball", 0.2, (0.7, 0.6, 0.8, 0.7)),
        ]
    )

    assert [item.class_name for item in tracked] == ["ball"]


def test_appearance_association_preserves_ids_when_objects_cross() -> None:
    tracker = MultiClassByteTracker(("cup",), match_threshold=0.8)
    first, _, _ = tracker.update_with_features(
        [
            Detection("cup", 0.9, (0.1, 0.1, 0.3, 0.5)),
            Detection("cup", 0.9, (0.7, 0.1, 0.9, 0.5)),
        ],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    second, _, _ = tracker.update_with_features(
        [
            Detection("cup", 0.9, (0.7, 0.1, 0.9, 0.5)),
            Detection("cup", 0.9, (0.1, 0.1, 0.3, 0.5)),
        ],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert [item.tracker_id for item in first] == [1, 2]
    assert [item.tracker_id for item in second] == [1, 2]


def test_strong_overlap_survives_temporarily_noisy_appearance() -> None:
    tracker = MultiClassByteTracker(("cup",), match_threshold=0.8)
    first, _, _ = tracker.update_with_features(
        [Detection("cup", 0.9, (0.1, 0.1, 0.3, 0.5))],
        torch.tensor([[1.0, 0.0]]),
    )
    second, _, _ = tracker.update_with_features(
        [Detection("cup", 0.9, (0.11, 0.1, 0.31, 0.5))],
        torch.tensor([[0.0, 1.0]]),
    )

    assert second[0].tracker_id == first[0].tracker_id


def test_crop_detection_clips_partially_out_of_bounds_box() -> None:
    image = Image.new("RGB", (100, 50))

    crop = crop_detection(
        image, Detection("cup", 0.9, (-0.2, 0.2, 0.4, 1.2)), padding=0
    )

    assert crop.size == (40, 40)


def test_crop_detection_rejects_box_fully_outside_image() -> None:
    image = Image.new("RGB", (100, 50))

    with pytest.raises(ValueError, match="outside image bounds"):
        crop_detection(image, Detection("cup", 0.9, (1.2, 0.2, 1.4, 0.4)))
