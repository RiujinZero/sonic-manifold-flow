"""Aggregate compatible continual-stream metric files across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manifoldflow.continual_aggregation import aggregate_continual_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    result = aggregate_continual_metrics(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
