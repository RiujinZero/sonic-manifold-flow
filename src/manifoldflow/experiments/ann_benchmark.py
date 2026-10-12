"""Benchmark HNSW recall against exact Euclidean and anisotropic basin targets."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch

from manifoldflow.ann_benchmark import benchmark_hnsw_routing
from manifoldflow.bank import BasinState


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--candidates", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--target", choices=("euclidean", "energy"), default="energy")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--graph-degree", type=int, default=16)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.queries <= 0 or not args.candidates or min(args.candidates) <= 0:
        raise ValueError("queries and candidate counts must be positive")
    bank_payload = torch.load(args.bank, map_location="cpu", weights_only=True)
    latent_payload = torch.load(args.latents, map_location="cpu", weights_only=True)
    if not isinstance(bank_payload, dict) or not isinstance(latent_payload, dict):
        raise ValueError("bank and latent artifacts must contain mappings")
    bank = BasinState.from_serialized(bank_payload)
    test_latent = latent_payload.get("test_latent")
    if not isinstance(test_latent, torch.Tensor):
        raise ValueError("latent artifact must contain test_latent")
    queries = test_latent[: args.queries].float().contiguous()
    if queries.shape[0] < args.queries or queries.shape[1] != bank.mu.shape[1]:
        raise ValueError("requested queries or dimensions exceed the artifacts")
    if max(args.candidates) > bank.mu.shape[0]:
        raise ValueError("candidate count cannot exceed bank size")

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
    metrics["seed"] = args.seed
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
