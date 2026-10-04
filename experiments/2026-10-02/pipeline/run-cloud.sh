#!/usr/bin/env bash
set -euo pipefail
# Run inside a CUDA PyTorch environment. No upload or Hub publication occurs.
DATASET_DIR=${1:?Pass the absolute dataset directory}
OUTPUT_DIR=${2:?Pass a new output directory}
python -m pip install -r "$(dirname "$0")/requirements-cloud.txt"
python "$(dirname "$0")/train_act.py" --dataset "$DATASET_DIR" --output "$OUTPUT_DIR" --device cuda --steps "${STEPS:-2000}" --batch-size "${BATCH_SIZE:-8}"
