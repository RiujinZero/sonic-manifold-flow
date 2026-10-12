from pathlib import Path

import torch

from manifoldflow.bank import BasinState
from manifoldflow.experiments.flow_common import load_valued_bank


def test_load_valued_bank_normalizes_class_counts(tmp_path: Path) -> None:
    state = BasinState.from_centers(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    payload = {
        "mu": state.mu,
        "raw_rate": state.raw_rate,
        "bias": state.bias,
        "count": state.count,
        "m2": state.m2,
        "usage": state.usage,
        "version": state.version,
        "class_count": torch.tensor([[3, 1], [0, 4]]),
    }
    path = tmp_path / "bank.pt"
    torch.save(payload, path)

    bank, probability = load_valued_bank(
        path,
        expected_dimension=2,
        expected_classes=2,
    )

    assert bank.mu.shape == (2, 2)
    assert probability is not None
    torch.testing.assert_close(
        probability,
        torch.tensor([[0.75, 0.25], [0.0, 1.0]]),
    )
