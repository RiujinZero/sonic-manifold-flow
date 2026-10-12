#!/usr/bin/env bash
source "$(dirname -- "$0")/_common.sh"

RUN_NAME="${RUN_NAME:-world_yolox_perception_smoke}"
RUN_EPOCHS="${RUN_EPOCHS:-2}"
RUN_EVAL_INTERVAL="${RUN_EVAL_INTERVAL:-1}"

require_file "$YOLOX_SOURCE/tools/train.py"
require_file "$YOLOX_EXP"
require_file "$YOLOX_INIT_CKPT"
require_file "$WORLD_PERCEPTION_DATASET/train/coco.json"

export MANIFOLDFLOW_WORLD_DATASET="$WORLD_PERCEPTION_DATASET"
python "$YOLOX_SOURCE/tools/train.py" \
  -f "$YOLOX_EXP" \
  -d "$WORLD_GPU_DEVICES" \
  -b "$YOLOX_BATCH_SIZE" \
  --fp16 \
  -c "$YOLOX_INIT_CKPT" \
  -expn "$RUN_NAME" \
  output_dir "$YOLOX_OUTPUT_DIR" \
  max_epoch "$RUN_EPOCHS" \
  eval_interval "$RUN_EVAL_INTERVAL" \
  data_num_workers "$YOLOX_WORKERS"

echo "Training complete: $YOLOX_OUTPUT_DIR/$RUN_NAME"
