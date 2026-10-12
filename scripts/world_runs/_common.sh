#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_REPO="${WORLD_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORLD_RUN_CONFIG="${WORLD_RUN_CONFIG:-$WORLD_REPO/configs/world-runs/next-runs.env}"

if [[ ! -f "$WORLD_RUN_CONFIG" ]]; then
  echo "Missing run configuration: $WORLD_RUN_CONFIG" >&2
  exit 2
fi

cd "$WORLD_REPO"
set -a
# shellcheck source=/dev/null
source "$WORLD_RUN_CONFIG"
set +a

if [[ ! -f "$WORLD_VENV/bin/activate" ]]; then
  echo "Missing WSL environment: $WORLD_VENV" >&2
  echo "Create it and install: pip install -e '.[world,dev,ann]'" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$WORLD_VENV/bin/activate"
export PYTHONPATH="$WORLD_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 2
  fi
}

mkdir -p "$WORLD_RESULTS_DIR"
