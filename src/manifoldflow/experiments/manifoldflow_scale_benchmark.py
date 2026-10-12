"""Benchmark CPU and CPU/GPU ManifoldFlow routing from 100k to 1M basins.

The immutable HNSW candidate index and complete basin bank remain in CPU memory
for both modes. Hybrid mode transfers only the query and selected candidate rows
to CUDA for anisotropic energy reranking and analytic top-k flow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.flow import analytic_topk_flow
from manifoldflow.index import HnswIndex
from manifoldflow.routing import rerank_topk
from manifoldflow.synthetic_scale import (
    synthetic_gaussian_mixture_bank,
    synthetic_query_latents,
)


@dataclass(frozen=True)
class _RoutingBank:
    """Inference-only basin fields, optionally cached on the GPU."""

    mu: Tensor
    raw_rate: Tensor
    bias: Tensor

    def gather(
        self, ids: Tensor, *, device: torch.device | str | None = None
    ) -> dict[str, Tensor]:
        target = self.mu.device if device is None else torch.device(device)
        local_ids = ids.to(self.mu.device)
        return {
            "mu": self.mu[local_ids].to(target),
            "raw_rate": self.raw_rate[local_ids].to(target),
            "bias": self.bias[local_ids].to(target),
            "ids": ids.to(target),
        }

    @property
    def bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.mu, self.raw_rate, self.bias)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basins", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--graph-degree", type=int, default=16)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu-only", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    positive = (
        args.basins,
        args.dimension,
        args.queries,
        args.latency_samples,
        args.candidates,
        args.top_k,
        args.ef_construction,
        args.graph_degree,
        args.ef_search,
    )
    if min(positive) <= 0 or not args.batch_sizes or min(args.batch_sizes) <= 0:
        raise ValueError("benchmark sizes and settings must be positive")
    if args.top_k > args.candidates or args.candidates > args.basins:
        raise ValueError("top_k <= candidates <= basins is required")
    if args.ef_search < args.candidates:
        raise ValueError("ef_search must be at least candidates")


def _process_memory_bytes() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            name, _, raw = line.partition(":")
            if name in {"VmRSS", "VmHWM"}:
                values[name] = int(raw.strip().split()[0]) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return values


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _latency_distribution(
    operation,
    queries: Tensor,
    *,
    samples: int,
    warmup: int = 20,
) -> dict[str, float]:
    selected = queries[: min(samples, queries.shape[0])]
    for index in range(min(warmup, selected.shape[0])):
        operation(selected[index : index + 1])
    timings = []
    for query in selected:
        started = time.perf_counter_ns()
        operation(query[None, :])
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "samples": len(timings),
        "mean_ms": sum(timings) / len(timings),
        "p50_ms": _percentile(timings, 0.50),
        "p95_ms": _percentile(timings, 0.95),
        "max_ms": max(timings),
    }


def _throughput(operation, queries: Tensor, batch_size: int) -> dict[str, float]:
    warmup = queries[: min(batch_size, queries.shape[0])]
    operation(warmup)
    started = time.perf_counter()
    batches = 0
    for offset in range(0, queries.shape[0], batch_size):
        operation(queries[offset : offset + batch_size])
        batches += 1
    seconds = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "queries": queries.shape[0],
        "batches": batches,
        "seconds": seconds,
        "queries_per_second": queries.shape[0] / seconds,
        "mean_batch_ms": seconds * 1000.0 / batches,
    }


def _bank_bytes(bank) -> int:
    tensors = (bank.mu, bank.raw_rate, bank.bias, bank.count, bank.m2, bank.usage)
    return sum(value.numel() * value.element_size() for value in tensors)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate(args)
    generation_started = time.perf_counter()
    bank = synthetic_gaussian_mixture_bank(
        args.basins, args.dimension, seed=args.seed
    )
    queries = synthetic_query_latents(bank, args.queries, seed=args.seed + 1)
    generation_seconds = time.perf_counter() - generation_started

    build_started = time.perf_counter()
    index = HnswIndex(
        bank.mu,
        ef_construction=args.ef_construction,
        graph_degree=args.graph_degree,
        ef_search=args.ef_search,
        seed=args.seed,
        threads=args.threads,
    )
    index_build_seconds = time.perf_counter() - build_started

    def lookup(query: Tensor) -> None:
        index.query(query, args.candidates)

    def route_flow(query: Tensor, device: torch.device, routing_bank=bank) -> None:
        candidate_ids = index.query(query, args.candidates).to(device)
        value = query.to(device)
        _, mu, raw_rate, bias, _ = rerank_topk(
            value, candidate_ids, routing_bank, top_k=args.top_k
        )
        analytic_topk_flow(value, mu, raw_rate, bias, t=1.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    modes: dict[str, Any] = {}
    lookup_latency = _latency_distribution(
        lookup, queries, samples=args.latency_samples
    )
    cpu_device = torch.device("cpu")

    def cpu_operation(value: Tensor) -> None:
        route_flow(value, cpu_device)

    modes["cpu"] = {
        "device": str(cpu_device),
        "route_flow_latency_batch1": _latency_distribution(
            cpu_operation, queries, samples=args.latency_samples
        ),
        "throughput": {
            str(size): _throughput(cpu_operation, queries, size)
            for size in args.batch_sizes
        },
    }

    cuda_available = torch.cuda.is_available()
    if not args.cpu_only and cuda_available:
        cuda_device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(cuda_device)

        def gpu_operation(value: Tensor) -> None:
            route_flow(value, cuda_device)

        modes["cpu_gpu_transfer"] = {
            "device": torch.cuda.get_device_name(cuda_device),
            "route_flow_latency_batch1": _latency_distribution(
                gpu_operation, queries, samples=args.latency_samples
            ),
            "throughput": {
                str(size): _throughput(gpu_operation, queries, size)
                for size in args.batch_sizes
            },
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_device),
        }

        cache_started = time.perf_counter()
        gpu_bank = _RoutingBank(
            bank.mu.to(cuda_device),
            bank.raw_rate.to(cuda_device),
            bank.bias.to(cuda_device),
        )
        torch.cuda.synchronize(cuda_device)
        cache_seconds = time.perf_counter() - cache_started

        def resident_operation(value: Tensor) -> None:
            route_flow(value, cuda_device, gpu_bank)

        modes["cpu_gpu_resident"] = {
            "device": torch.cuda.get_device_name(cuda_device),
            "routing_cache_bytes": gpu_bank.bytes,
            "routing_cache_load_seconds": cache_seconds,
            "route_flow_latency_batch1": _latency_distribution(
                resident_operation, queries, samples=args.latency_samples
            ),
            "throughput": {
                str(size): _throughput(resident_operation, queries, size)
                for size in args.batch_sizes
            },
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_device),
        }

    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "benchmark": "manifoldflow_cpu_vs_cpu_gpu_scale",
        "basins": args.basins,
        "dimension": args.dimension,
        "queries": args.queries,
        "candidates": args.candidates,
        "top_k": args.top_k,
        "seed": args.seed,
        "synthetic_systems_scale": True,
        "generation_seconds": generation_seconds,
        "hnsw": {
            "build_seconds": index_build_seconds,
            "ef_construction": args.ef_construction,
            "graph_degree": args.graph_degree,
            "ef_search": args.ef_search,
            "threads": args.threads,
            "index_bytes": index.index_bytes,
            "lookup_latency_batch1": lookup_latency,
        },
        "bank_bytes": _bank_bytes(bank),
        "process_memory": _process_memory_bytes(),
        "modes": modes,
        "hardware": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "gpu": gpu_name,
        },
    }


def main() -> None:
    args = _parser().parse_args()
    metrics = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
