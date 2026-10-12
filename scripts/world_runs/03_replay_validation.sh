#!/usr/bin/env bash
source "$(dirname -- "$0")/_common.sh"

RUN_NAME="${RUN_NAME:-world_yolox_perception_full}"
FULL_CHECKPOINT="${1:-$YOLOX_OUTPUT_DIR/$RUN_NAME/best_ckpt.pth}"
RUNTIME_CHECKPOINT="${2:-$YOLOX_OUTPUT_DIR/$RUN_NAME/runtime-model.pth}"

require_file "$FULL_CHECKPOINT"
require_file "$WORLD_PROJECTOR_CKPT"
require_file "$WORLD_GEMMA_CKPT"
mkdir -p "$(dirname -- "$RUNTIME_CHECKPOINT")" "$WORLD_VIDEO_OUTPUT"

manifoldflow-world-export-yolox "$FULL_CHECKPOINT" "$RUNTIME_CHECKPOINT"

for stem in Video_1 Video_3; do
  require_file "$WORLD_VIDEO_DIR/$stem.mov"
  require_file "$WORLD_VIDEO_ANNOTATIONS/$stem.annotations.json"
  manifoldflow-world-replay-video \
    "$RUNTIME_CHECKPOINT" \
    "$WORLD_PROJECTOR_CKPT" \
    "$WORLD_VIDEO_DIR/$stem.mov" \
    --annotations "$WORLD_VIDEO_ANNOTATIONS/$stem.annotations.json" \
    --gemma-checkpoint "$WORLD_GEMMA_CKPT" \
    --gemma-model "$WORLD_GEMMA_MODEL" \
    --output-dir "$WORLD_VIDEO_OUTPUT/$stem" \
    --mode deterministic \
    --detector-confidence "$WORLD_DETECTOR_CONFIDENCE" \
    --track-confidence "$WORLD_TRACK_CONFIDENCE" \
    --device "$WORLD_DEVICE" \
    --session-id "$RUN_NAME-$stem"
done

echo "Replay artifacts: $WORLD_VIDEO_OUTPUT"
