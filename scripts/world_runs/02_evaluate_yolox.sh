#!/usr/bin/env bash
source "$(dirname -- "$0")/_common.sh"

RUN_NAME="${RUN_NAME:-world_yolox_perception_smoke}"
CHECKPOINT="${1:-$YOLOX_OUTPUT_DIR/$RUN_NAME/best_ckpt.pth}"
EVAL_DATASET="${EVAL_DATASET:-$WORLD_PERCEPTION_DATASET}"
EVAL_LABEL="${EVAL_LABEL:-perception-validation}"
RESULT_PATH="${RESULT_PATH:-$WORLD_RESULTS_DIR/${RUN_NAME}-${EVAL_LABEL}.log}"

require_file "$CHECKPOINT"
require_file "$EVAL_DATASET/validation/coco.json"
require_file "$YOLOX_SOURCE/tools/eval.py"
mkdir -p "$(dirname -- "$RESULT_PATH")"

export MANIFOLDFLOW_WORLD_DATASET="$EVAL_DATASET"
python "$YOLOX_SOURCE/tools/eval.py" \
  -f "$YOLOX_EXP" \
  -c "$CHECKPOINT" \
  -d "$WORLD_GPU_DEVICES" \
  -b "$YOLOX_BATCH_SIZE" \
  --fp16 2>&1 | tee "$RESULT_PATH"

echo "Evaluation log: $RESULT_PATH"
