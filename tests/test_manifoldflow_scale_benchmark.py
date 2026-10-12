from argparse import Namespace

import pytest
import torch

from manifoldflow.experiments.manifoldflow_scale_benchmark import (
    _percentile,
    _RoutingBank,
    _validate,
)


def test_routing_bank_gathers_only_inference_fields() -> None:
    bank = _RoutingBank(
        mu=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        raw_rate=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        bias=torch.tensor([0.5, 0.6]),
    )

    gathered = bank.gather(torch.tensor([[1]]))

    assert gathered["mu"].tolist() == [[[3.0, 4.0]]]
    assert bank.bytes == 40


def test_scale_benchmark_validation_rejects_invalid_candidate_order() -> None:
    args = Namespace(
        basins=100,
        dimension=8,
        queries=10,
        latency_samples=5,
        candidates=16,
        top_k=32,
        ef_construction=40,
        graph_degree=8,
        ef_search=32,
        batch_sizes=[1, 8],
    )

    with pytest.raises(ValueError, match="top_k <= candidates <= basins"):
        _validate(args)


def test_percentile_uses_nearest_rank() -> None:
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
