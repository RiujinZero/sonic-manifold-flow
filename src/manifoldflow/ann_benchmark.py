"""Exact-reference helpers for ANN candidate recall and latency gates."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from manifoldflow.bank import BasinState
from manifoldflow.flow import basin_energy
from manifoldflow.index import HnswIndex
from manifoldflow.routing import rerank_topk


@torch.no_grad()
def exact_euclidean_top1(
    queries: Tensor,
    centers: Tensor,
    *,
    query_chunk_size: int = 128,
    center_chunk_size: int = 4096,
) -> Tensor:
    """Find exact nearest centers without materializing the full distance matrix."""
    if queries.ndim != 2 or centers.ndim != 2 or queries.shape[1] != centers.shape[1]:
        raise ValueError("queries and centers must be non-empty matched matrices")
    if queries.shape[0] == 0 or centers.shape[0] == 0:
        raise ValueError("queries and centers cannot be empty")
    if min(query_chunk_size, center_chunk_size) <= 0:
        raise ValueError("chunk sizes must be positive")
    output = []
    for start in range(0, queries.shape[0], query_chunk_size):
        query = queries[start : start + query_chunk_size]
        best_distance = torch.full(
            (query.shape[0],), torch.inf, device=query.device, dtype=query.dtype
        )
        best_id = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
        for center_start in range(0, centers.shape[0], center_chunk_size):
            center = centers[center_start : center_start + center_chunk_size].to(
                device=query.device, dtype=query.dtype
            )
            distance, local_id = torch.cdist(query, center).square().min(dim=1)
            improve = distance < best_distance
            best_distance = torch.where(improve, distance, best_distance)
            best_id = torch.where(improve, local_id + center_start, best_id)
        output.append(best_id)
    return torch.cat(output)


@torch.no_grad()
def exact_energy_top1(
    queries: Tensor,
    bank: BasinState,
    *,
    query_chunk_size: int = 64,
    center_chunk_size: int = 4096,
) -> Tensor:
    """Find the exact lowest-energy basin in bounded chunks."""
    if (
        queries.ndim != 2
        or queries.shape[0] == 0
        or queries.shape[1] != bank.mu.shape[1]
    ):
        raise ValueError("queries must have shape [samples, bank.dimension]")
    if min(query_chunk_size, center_chunk_size) <= 0:
        raise ValueError("chunk sizes must be positive")
    output = []
    for start in range(0, queries.shape[0], query_chunk_size):
        query = queries[start : start + query_chunk_size]
        best_energy = torch.full(
            (query.shape[0],), torch.inf, device=query.device, dtype=query.dtype
        )
        best_id = torch.zeros(query.shape[0], device=query.device, dtype=torch.long)
        for center_start in range(0, bank.mu.shape[0], center_chunk_size):
            stop = min(center_start + center_chunk_size, bank.mu.shape[0])
            ids = torch.arange(center_start, stop, device=bank.mu.device)
            records = bank.gather(ids, device=query.device)
            energy = basin_energy(
                query,
                records["mu"][None, :, :].expand(query.shape[0], -1, -1),
                records["raw_rate"][None, :, :].expand(query.shape[0], -1, -1),
                records["bias"][None, :].expand(query.shape[0], -1),
            )
            value, local_id = energy.min(dim=1)
            improve = value < best_energy
            best_energy = torch.where(improve, value, best_energy)
            best_id = torch.where(improve, local_id + center_start, best_id)
        output.append(best_id)
    return torch.cat(output)


@torch.no_grad()
def energy_reranked_top1(
    queries: Tensor, candidate_ids: Tensor, bank: BasinState
) -> Tensor:
    ids, _, _, _, _ = rerank_topk(queries, candidate_ids, bank, top_k=1)
    return ids.squeeze(1)


def latency_distribution_ms(
    operation: Callable[[Tensor], object],
    queries: Tensor,
    *,
    warmup: int = 20,
) -> dict[str, float]:
    """Measure batch-one wall latency and report p50/p95/mean."""
    if queries.ndim != 2 or queries.shape[0] == 0 or warmup < 0:
        raise ValueError("queries must be non-empty and warmup non-negative")
    for index in range(min(warmup, queries.shape[0])):
        operation(queries[index : index + 1])
    samples = []
    for query in queries:
        started = time.perf_counter_ns()
        operation(query[None, :])
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "mean": sum(samples) / len(samples),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


def _tensor_bytes(*values: Tensor) -> int:
    return sum(value.numel() * value.element_size() for value in values)


@torch.no_grad()
def benchmark_hnsw_routing(
    bank: BasinState,
    queries: Tensor,
    *,
    target: str,
    candidate_counts: Sequence[int],
    ef_construction: int = 200,
    graph_degree: int = 16,
    ef_search: int = 128,
    seed: int = 17,
    threads: int = 0,
    reference: Tensor | None = None,
) -> dict[str, Any]:
    """Measure HNSW candidate recall, routed recall, latency, and index bytes.

    Shared by the real-artifact and synthetic-scale ANN experiment scripts so
    both report identical fields regardless of where the bank came from. The
    exact reference does not depend on any HNSW setting, so callers sweeping
    settings against the same bank/queries should compute it once and pass it
    in rather than paying the exact-reference cost (minutes, at 1M energy
    targets) on every combination.
    """
    if target not in ("euclidean", "energy"):
        raise ValueError("target must be 'euclidean' or 'energy'")
    if not candidate_counts or min(candidate_counts) <= 0:
        raise ValueError("candidate_counts must be a non-empty positive sequence")
    if max(candidate_counts) > bank.mu.shape[0]:
        raise ValueError("candidate count cannot exceed bank size")
    if queries.ndim != 2 or queries.shape[1] != bank.mu.shape[1]:
        raise ValueError("queries must have shape [samples, bank.dimension]")
    if reference is not None and reference.shape != (queries.shape[0],):
        raise ValueError("reference must have shape [queries.shape[0]]")

    if reference is None:
        reference_started = time.perf_counter()
        reference = (
            exact_energy_top1(queries, bank)
            if target == "energy"
            else exact_euclidean_top1(queries, bank.mu)
        )
        reference_seconds = time.perf_counter() - reference_started
    else:
        reference_seconds = None

    build_started = time.perf_counter()
    index = HnswIndex(
        bank.mu,
        ef_construction=ef_construction,
        graph_degree=graph_degree,
        ef_search=ef_search,
        seed=seed,
        threads=threads,
    )
    build_seconds = time.perf_counter() - build_started

    results = {}
    for candidate_count in candidate_counts:
        candidate_ids = index.query(queries, candidate_count)
        candidate_recall = (
            (candidate_ids == reference[:, None]).any(dim=1).float().mean()
        )
        routed = (
            energy_reranked_top1(queries, candidate_ids, bank)
            if target == "energy"
            else candidate_ids[:, 0]
        )

        def operation(query: Tensor, count: int = candidate_count) -> None:
            ids = index.query(query, count)
            if target == "energy":
                energy_reranked_top1(query, ids, bank)

        results[str(candidate_count)] = {
            "candidate_recall": float(candidate_recall.item()),
            "routed_top1_recall": float((routed == reference).float().mean().item()),
            "lookup_rerank_latency_ms_batch1": latency_distribution_ms(
                operation, queries
            ),
        }

    return {
        "target": target,
        "basins": bank.mu.shape[0],
        "dimension": bank.mu.shape[1],
        "queries": queries.shape[0],
        "candidate_counts": list(candidate_counts),
        "hnsw": {
            "ef_construction": ef_construction,
            "graph_degree": graph_degree,
            "ef_search": ef_search,
            "threads": threads,
            "build_seconds": build_seconds,
        },
        "exact_reference_seconds": reference_seconds,
        "results": results,
        "bank_bytes": {
            "mu_raw_rate_m2": _tensor_bytes(bank.mu, bank.raw_rate, bank.m2),
            "metadata": _tensor_bytes(bank.bias, bank.count, bank.usage),
        },
    }
