"""Sweep HNSW settings on one synthetic bank against one cached exact reference.

`ann_scale_benchmark.py` recomputes the exact reference on every invocation.
That is fine for a single measurement, but the exact reference does not
depend on any HNSW build or search setting, so recomputing it per combination
wastes almost all of a parameter sweep's wall time (minutes, for the
anisotropic energy target at 1M basins). This script generates one bank and
query set, computes the reference once, and reports recall/latency/build time
for every (graph_degree, ef_construction, ef_search) combination given on the
command line, so the cheapest setting that clears a recall target can be
found without re-paying the reference cost each time.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from manifoldflow.ann_benchmark import (
    benchmark_hnsw_routing,
    exact_energy_top1,
    exact_euclidean_top1,
)
from manifoldflow.synthetic_scale import (
    synthetic_gaussian_mixture_bank,
    synthetic_query_latents,
)


def _parse_config(spec: str) -> tuple[int, int, int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            "--config must be GRAPH_DEGREE:EF_CONSTRUCTION:EF_SEARCH, "
            f"got {spec!r}"
        )
    try:
        graph_degree, ef_construction, ef_search = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"--config values must be integers, got {spec!r}") from error
    if min(graph_degree, ef_construction, ef_search) <= 0:
        raise ValueError("--config values must be positive")
    return graph_degree, ef_construction, ef_search


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basins", required=True, type=int)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--target", choices=("euclidean", "energy"), default="energy")
    parser.add_argument("--candidates", type=int, nargs="+", default=[128])
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        dest="configs",
        metavar="GRAPH_DEGREE:EF_CONSTRUCTION:EF_SEARCH",
        help="Repeatable. One HNSW settings combination, e.g. 16:200:1024",
    )
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.basins <= 0 or args.dimension <= 0:
        raise ValueError("basins and dimension must be positive")
    if args.queries <= 0 or not args.candidates or min(args.candidates) <= 0:
        raise ValueError("queries and candidate counts must be positive")
    if not 0.0 < args.recall_target <= 1.0:
        raise ValueError("recall_target must be in (0, 1]")

    bank = synthetic_gaussian_mixture_bank(
        args.basins, args.dimension, clusters=args.clusters, seed=args.seed
    )
    queries = synthetic_query_latents(bank, args.queries, seed=args.seed + 1)

    reference_started = time.perf_counter()
    reference = (
        exact_energy_top1(queries, bank)
        if args.target == "energy"
        else exact_euclidean_top1(queries, bank.mu)
    )
    reference_seconds = time.perf_counter() - reference_started

    configs = [_parse_config(spec) for spec in args.configs]
    max_candidates = max(args.candidates)
    runs = []
    for graph_degree, ef_construction, ef_search in configs:
        if ef_search < max_candidates:
            raise ValueError(
                f"--config {graph_degree}:{ef_construction}:{ef_search} has "
                f"ef_search below the largest candidate count {max_candidates}"
            )
        metrics = benchmark_hnsw_routing(
            bank,
            queries,
            target=args.target,
            candidate_counts=args.candidates,
            ef_construction=ef_construction,
            graph_degree=graph_degree,
            ef_search=ef_search,
            seed=args.seed,
            threads=args.threads,
            reference=reference,
        )
        runs.append(metrics)

    passing = [
        run
        for run in runs
        if max(entry["routed_top1_recall"] for entry in run["results"].values())
        >= args.recall_target
    ]
    cheapest_passing = (
        min(passing, key=lambda run: run["hnsw"]["build_seconds"])
        if passing
        else None
    )

    return {
        "basins": args.basins,
        "dimension": args.dimension,
        "target": args.target,
        "queries": args.queries,
        "seed": args.seed,
        "recall_target": args.recall_target,
        "exact_reference_seconds": reference_seconds,
        "runs": runs,
        "cheapest_passing_hnsw": (
            cheapest_passing["hnsw"] if cheapest_passing is not None else None
        ),
        "hardware": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
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
