"""Select feasible robustness operating points from validation aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manifoldflow.selection import select_operating_points


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-clean-mse-increase", type=float, default=0.05)
    parser.add_argument("--max-clean-accuracy-drop", type=float, default=0.005)
    return parser


def main() -> None:
    args = _parser().parse_args()
    aggregate = json.loads(args.summary.read_text(encoding="utf-8"))
    result = select_operating_points(
        aggregate,
        max_clean_mse_relative_increase=args.max_clean_mse_increase,
        max_clean_accuracy_drop=args.max_clean_accuracy_drop,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
