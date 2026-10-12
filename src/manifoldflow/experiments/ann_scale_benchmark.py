"""Benchmark HNSW recall/latency/memory on a synthetic bank at systems scale.

Real Fashion-MNIST-derived banks cannot reach the 100k-1M basin range the
design's roadmap requires for scale validation (60,000 training images caps
real correctness banks well below that). This script exercises the same
`HnswIndex` and reranking path against a synthetic Gaussian-mixture bank so
recall, build time, query/rerank latency, and bank/index bytes can be
measured at 10k, 100k, and 1M basins. It is a systems-scale measurement, not
a substitute for the real-data correctness and learning-quality gates.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch

from manifoldflow.ann_benchmark import benchmark_hnsw_routing
from manifoldflow.synthetic_scale import (
    synthetic_gaussian_mixture_bank,
    synthetic_query_latents,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basins", required=True, type=int)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--candidates", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--target", choices=("euclidean", "energy"), default="energy")
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--graph-degree", type=int, default=16)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.basins <= 0 or args.dimension <= 0:
        raise ValueError("basins and dimension must be positive")
    if args.queries <= 0 or not args.candidates or min(args.candidates) <= 0:
        raise ValueError("queries and candidate counts must be positive")
    if max(args.candidates) > args.basins:
        raise ValueError("candidate count cannot exceed bank size")

    bank = synthetic_gaussian_mixture_bank(
        args.basins,
        args.dimension,
        clusters=args.clusters,
        seed=args.seed,
    )
    queries = synthetic_query_latents(bank, args.queries, seed=args.seed + 1)

    metrics = benchmark_hnsw_routing(
        bank,
        queries,
        target=args.target,
        candidate_counts=args.candidates,
        ef_construction=args.ef_construction,
        graph_degree=args.graph_degree,
        ef_search=args.ef_search,
        seed=args.seed,
        threads=args.threads,
    )
    metrics["synthetic"] = True
    metrics["seed"] = args.seed
    metrics["clusters"] = args.clusters if args.clusters is not None else round(
        args.basins**0.5
    )
    metrics["hardware"] = {
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    return metrics


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
