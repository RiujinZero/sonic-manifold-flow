import argparse

import pytest

from manifoldflow.experiments.world_replay_video import _class_threshold


def test_class_threshold_parses_known_class() -> None:
    assert _class_threshold("ball=0.05") == ("ball", 0.05)


@pytest.mark.parametrize("value", ["person=0.2", "ball=2", "ball=bad", "ball"])
def test_class_threshold_rejects_invalid_value(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _class_threshold(value)
