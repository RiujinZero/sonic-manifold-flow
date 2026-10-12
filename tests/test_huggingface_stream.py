import time

import pytest
import torch
from PIL import Image
from torchvision import transforms

from manifoldflow.huggingface_stream import batches_from_rows, with_timeout

_TRANSFORM = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])


def _rows(count: int):
    for index in range(count):
        yield {
            "image": Image.new("RGB", (16, 16), color=(index % 256, 0, 0)),
            "label": index % 3,
        }


def test_batches_from_rows_yields_full_batches() -> None:
    batches = list(batches_from_rows(_rows(10), transform=_TRANSFORM, batch_size=4))

    assert [image.shape[0] for image, _ in batches] == [4, 4, 2]
    assert batches[0][0].shape == (4, 3, 8, 8)
    assert batches[-1][1].tolist() == [8 % 3, 9 % 3]


def test_batches_from_rows_applies_the_transform() -> None:
    batches = batches_from_rows(_rows(1), transform=_TRANSFORM, batch_size=1)
    (image, _label), *_ = batches

    assert image.dtype == torch.float32
    assert image.shape == (1, 3, 8, 8)


def test_batches_from_rows_rejects_non_positive_batch_size() -> None:
    try:
        list(batches_from_rows(_rows(1), transform=_TRANSFORM, batch_size=0))
    except ValueError:
        pass
    else:
        raise AssertionError("batch_size <= 0 should raise")


def test_batches_from_rows_handles_empty_input() -> None:
    assert list(batches_from_rows(_rows(0), transform=_TRANSFORM, batch_size=4)) == []


def test_with_timeout_passes_through_items_from_a_healthy_iterable() -> None:
    assert list(with_timeout(range(5), seconds=2)) == [0, 1, 2, 3, 4]


def test_with_timeout_raises_when_the_iterable_stalls() -> None:
    def _stalls():
        yield 1
        time.sleep(2)
        yield 2  # pragma: no cover - never reached before the timeout fires

    with pytest.raises(TimeoutError):
        list(with_timeout(_stalls(), seconds=0.2))


def test_with_timeout_propagates_exceptions_from_the_wrapped_iterable() -> None:
    def _fails():
        yield 1
        raise RuntimeError("upstream failure")

    collected = []
    with pytest.raises(RuntimeError, match="upstream failure"):
        for item in with_timeout(_fails(), seconds=2):
            collected.append(item)
    assert collected == [1]


def test_with_timeout_rejects_non_positive_seconds() -> None:
    with pytest.raises(ValueError):
        list(with_timeout(range(3), seconds=0))
