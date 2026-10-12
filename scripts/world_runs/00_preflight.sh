#!/usr/bin/env bash
source "$(dirname -- "$0")/_common.sh"

require_dir "$YOLOX_SOURCE"
require_file "$YOLOX_SOURCE/tools/train.py"
require_file "$YOLOX_SOURCE/tools/eval.py"
require_file "$YOLOX_EXP"
require_file "$YOLOX_INIT_CKPT"
require_file "$WORLD_PROJECTOR_CKPT"
require_file "$WORLD_GEMMA_CKPT"
require_file "$WORLD_PERCEPTION_DATASET/manifest.json"
require_file "$WORLD_PERCEPTION_DATASET/train/coco.json"
require_file "$WORLD_PERCEPTION_DATASET/validation/coco.json"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["WORLD_PERCEPTION_DATASET"])
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["splits"] == {"train": 97, "validation": 19}
clips = [json.loads(line) for line in (root / "clips.jsonl").read_text().splitlines()]
train = {row["clip_id"] for row in clips if row["split"] == "train"}
validation = {row["clip_id"] for row in clips if row["split"] == "validation"}
assert train and validation and train.isdisjoint(validation)
for split in ("train", "validation"):
    coco = json.loads((root / split / "coco.json").read_text())
    assert coco["images"] and coco["annotations"]
    assert all((root / split / "images" / row["file_name"]).is_file() for row in coco["images"])
print(f"corpus_ok train_clips={len(train)} validation_clips={len(validation)}")
PY

python - <<'PY'
import torch
import transformers

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the scheduled training runs")
print(torch.cuda.get_device_name(0))
PY

python -m pytest \
  tests/test_world_external_datasets.py \
  tests/test_world_video.py \
  tests/test_world_replay_video.py \
  tests/test_world_identity.py \
  tests/test_world_events_tracker_merge.py

echo "PRECHECK PASS"
