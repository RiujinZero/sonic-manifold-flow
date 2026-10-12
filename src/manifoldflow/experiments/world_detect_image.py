"""Run the local four-class YOLOX detector on one image."""

from __future__ import annotations

import argparse
import json

from PIL import Image

from manifoldflow.world_dataset import CLASSES
from manifoldflow.world_vision import YoloXDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("image")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.3)
    args = parser.parse_args()
    detector = YoloXDetector(
        args.checkpoint,
        CLASSES,
        device=args.device,
        confidence=args.confidence,
    )
    with Image.open(args.image) as source:
        detections = detector.detect(source.convert("RGB"))
    print(
        json.dumps(
            [
                {
                    "class": item.class_name,
                    "confidence": item.confidence,
                    "box": item.box_xyxy,
                }
                for item in detections
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
