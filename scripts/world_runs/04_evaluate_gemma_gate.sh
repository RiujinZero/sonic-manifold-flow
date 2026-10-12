#!/usr/bin/env bash
source "$(dirname -- "$0")/_common.sh"

OUTPUT="${1:-$WORLD_RESULTS_DIR/oracle-schema-final-validation.json}"
require_file "$WORLD_GEMMA_CKPT"
require_dir "$WORLD_SYNTHETIC_DATASET"
mkdir -p "$(dirname -- "$OUTPUT")"

manifoldflow-world-evaluate-oracle \
  --dataset "$WORLD_SYNTHETIC_DATASET" \
  --checkpoint "$WORLD_GEMMA_CKPT" \
  --output "$OUTPUT" \
  --split validation \
  --max-examples 512 \
  --rank-examples 512 \
  --generate-examples 128 \
  --device "$WORLD_DEVICE"

echo "Gemma gate metrics: $OUTPUT"
