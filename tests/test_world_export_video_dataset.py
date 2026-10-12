import pytest

from manifoldflow.experiments.world_export_video_dataset import (
    _coco_box,
    _dataset_split,
)


def test_coco_box_scales_normalized_coordinates() -> None:
    assert _coco_box([0.1, 0.2, 0.4, 0.7], 100, 200) == pytest.approx(
        [10.0, 40.0, 30.0, 100.0]
    )


def test_annotation_split_is_authoritative() -> None:
    assert _dataset_split({"split": "validation"}, "clip", set()) == "validation"
    assert _dataset_split({}, "clip", {"clip"}) == "validation"


def test_conflicting_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        _dataset_split({"split": "train"}, "clip", {"clip"})
