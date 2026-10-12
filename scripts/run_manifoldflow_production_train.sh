#!/bin/bash
# Launches the production ManifoldFlow training loop
# (manifoldflow_production_train.py) in the background via WSL's GPU venv -
# matches every other scripts/run_*.sh launch convention this project uses
# (see project_sonic_gpu_env.md).
#
# Prerequisite: pregenerated session shards under --shard-dir. Generate
# them first with:
#   python scripts/pregenerate_manifoldflow_sessions.py \
#     --output-dir /path/to/manifoldflow_sessions --sessions <N> --workers <N>
# This script does not generate them - it assumes they already exist.
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow" || exit 1
mkdir -p artifacts/manifoldflow_production_train

SHARD_DIR="${SHARD_DIR:-/path/to/manifoldflow_sessions}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/gdrive/Sonic/prod/latest_eval_30B.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-artifacts/manifoldflow_production_train/checkpoint.pt}"

setsid /home/user/gtr_venv/bin/python -m manifoldflow.experiments.manifoldflow_production_train \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --shard-dir "$SHARD_DIR" \
  --output-checkpoint "$OUTPUT_CHECKPOINT" \
  --history-output artifacts/manifoldflow_production_train/history.json \
  --steps 100000 --num-lanes 16 --eval-lanes 8 --eval-every 200 --eval-windows 8 \
  --checkpoint-every-updates 200 --log-every 20 --log-every-seconds 60 \
  --window-tokens 8192 \
  --learning-rate 1e-3 --sonic-learning-rate 1e-5 \
  --unfreeze-accuracy-threshold 0.95 --unfreeze-patience 3 \
  --plain-lm-loss-weight 0.6 --distractor-loss-weight 0.4 \
  --mixed-precision bf16 --device cuda \
  < /dev/null > /tmp/manifoldflow_production_train.log 2>&1 &
disown
sleep 2
echo "started pid $!"
