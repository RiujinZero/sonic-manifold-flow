"""Process one image through the complete local visual world-state pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from manifoldflow.world_dataset import CLASSES
from manifoldflow.world_events import TabletopEventEngine
from manifoldflow.world_live import VisionFrameProcessor
from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_vision import (
    CrossModalKeyProjector,
    MultiClassByteTracker,
    SiglipVisionLanguageEncoder,
    YoloXDetector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detector_checkpoint")
    parser.add_argument("projector_checkpoint")
    parser.add_argument("image")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    encoder = SiglipVisionLanguageEncoder(device=args.device, dtype=dtype)
    projector = CrossModalKeyProjector(encoder.feature_dimension, 128)
    projector.load_state_dict(
        torch.load(args.projector_checkpoint, map_location="cpu", weights_only=True)
    )
    projector.eval().to(args.device)
    memory = WorldMemory()
    processor = VisionFrameProcessor(
        YoloXDetector(args.detector_checkpoint, CLASSES, device=args.device),
        encoder,
        projector,
        memory,
        TabletopEventEngine(memory),
        lambda: MultiClassByteTracker(CLASSES),
    )
    result = processor.process("smoke", Path(args.image).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
