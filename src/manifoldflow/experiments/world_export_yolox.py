"""Export a model-only YOLOX checkpoint for safe runtime loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_checkpoint(source: str | Path, destination: str | Path) -> Path:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("source must be a trusted YOLOX training checkpoint")
    state = payload["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("YOLOX checkpoint has no model state")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(export_checkpoint(args.source, args.destination))


if __name__ == "__main__":
    main()
