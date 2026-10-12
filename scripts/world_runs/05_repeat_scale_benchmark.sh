#!/usr/bin/env bash
source "$(dirname -- "$0")/_common.sh"

mkdir -p "$SCALE_RESULTS_DIR"

for seed in 17 18 19; do
  for basins in 100000 1000000; do
    manifoldflow-scale-benchmark \
      --basins "$basins" \
      --dimension 128 \
      --queries 4096 \
      --latency-samples 200 \
      --batch-sizes 1 32 256 \
      --candidates 128 \
      --top-k 4 \
      --ef-construction 200 \
      --graph-degree 16 \
      --ef-search 256 \
      --seed "$seed" \
      --output "$SCALE_RESULTS_DIR/${basins}-seed-${seed}.json"
  done
done

echo "Scale repeats: $SCALE_RESULTS_DIR"
