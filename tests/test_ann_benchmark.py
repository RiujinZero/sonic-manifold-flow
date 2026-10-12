import torch

from manifoldflow import (
    BasinState,
    energy_reranked_top1,
    exact_energy_top1,
    exact_euclidean_top1,
    latency_distribution_ms,
)


def test_exact_ann_references_and_energy_rerank() -> None:
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    bank = BasinState.from_statistics(
        centers, torch.ones(3, dtype=torch.long), torch.zeros_like(centers)
    )
    queries = torch.tensor([[1.8, 0.0], [4.8, 0.0]])

    assert exact_euclidean_top1(
        queries, centers, query_chunk_size=1, center_chunk_size=2
    ).tolist() == [1, 2]
    assert exact_energy_top1(
        queries, bank, query_chunk_size=1, center_chunk_size=2
    ).tolist() == [1, 2]
    candidates = torch.tensor([[0, 1], [1, 2]])
    assert energy_reranked_top1(queries, candidates, bank).tolist() == [1, 2]


def test_latency_distribution_has_ordered_percentiles() -> None:
    result = latency_distribution_ms(lambda value: value.square(), torch.ones(4, 2))

    assert 0 <= result["p50"] <= result["p95"] <= result["maximum"]
