import pytest
import torch

from manifoldflow import (
    benchmark_hnsw_routing,
    exact_euclidean_top1,
    synthetic_gaussian_mixture_bank,
    synthetic_query_latents,
)


def test_synthetic_bank_has_requested_shape_and_valid_statistics() -> None:
    bank = synthetic_gaussian_mixture_bank(200, 8, clusters=5, seed=3)

    assert bank.mu.shape == (200, 8)
    assert torch.all(bank.count > 0)
    assert torch.all(bank.m2 >= 0)


def test_synthetic_bank_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        synthetic_gaussian_mixture_bank(0, 8)
    with pytest.raises(ValueError):
        synthetic_gaussian_mixture_bank(10, 0)


def test_synthetic_queries_match_bank_dimension_and_count() -> None:
    bank = synthetic_gaussian_mixture_bank(50, 4, clusters=3, seed=1)

    queries = synthetic_query_latents(bank, 20, near_fraction=0.5, seed=2)

    assert queries.shape == (20, 4)


def test_synthetic_queries_all_near_when_near_fraction_is_one() -> None:
    bank = synthetic_gaussian_mixture_bank(50, 4, clusters=3, seed=1)

    queries = synthetic_query_latents(
        bank, 30, near_fraction=1.0, near_std=1e-6, seed=2
    )

    distance = torch.cdist(queries, bank.mu).square().min(dim=1).values
    assert torch.all(distance < 1e-4)


def test_benchmark_hnsw_routing_reports_recall_and_bank_bytes() -> None:
    bank = synthetic_gaussian_mixture_bank(300, 8, clusters=10, seed=5)
    queries = synthetic_query_latents(bank, 40, seed=6)

    metrics = benchmark_hnsw_routing(
        bank,
        queries,
        target="euclidean",
        candidate_counts=[16, 32],
        ef_construction=100,
        graph_degree=8,
        ef_search=64,
    )

    assert metrics["basins"] == 300
    assert metrics["dimension"] == 8
    assert set(metrics["results"]) == {"16", "32"}
    for entry in metrics["results"].values():
        assert 0.0 <= entry["candidate_recall"] <= 1.0
        assert 0.0 <= entry["routed_top1_recall"] <= 1.0
    assert metrics["bank_bytes"]["mu_raw_rate_m2"] > 0


def test_benchmark_hnsw_routing_accepts_precomputed_reference() -> None:
    bank = synthetic_gaussian_mixture_bank(300, 8, clusters=10, seed=5)
    queries = synthetic_query_latents(bank, 40, seed=6)
    reference = exact_euclidean_top1(queries, bank.mu)

    metrics = benchmark_hnsw_routing(
        bank,
        queries,
        target="euclidean",
        candidate_counts=[16, 32],
        reference=reference,
    )

    assert metrics["exact_reference_seconds"] is None
    for entry in metrics["results"].values():
        assert 0.0 <= entry["routed_top1_recall"] <= 1.0


def test_benchmark_hnsw_routing_rejects_mismatched_reference_shape() -> None:
    bank = synthetic_gaussian_mixture_bank(50, 4, clusters=3, seed=1)
    queries = synthetic_query_latents(bank, 20, seed=2)

    with pytest.raises(ValueError):
        benchmark_hnsw_routing(
            bank,
            queries,
            target="euclidean",
            candidate_counts=[8],
            reference=torch.zeros(5, dtype=torch.long),
        )


def test_benchmark_hnsw_routing_rejects_candidates_beyond_bank_size() -> None:
    bank = synthetic_gaussian_mixture_bank(20, 4, clusters=2, seed=1)
    queries = synthetic_query_latents(bank, 5, seed=2)

    with pytest.raises(ValueError):
        benchmark_hnsw_routing(
            bank, queries, target="euclidean", candidate_counts=[50]
        )
