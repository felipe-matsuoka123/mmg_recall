#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-${VARIANT:-grayscale}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/outputs}"
TORCH_HOME="${TORCH_HOME:-${OUTPUT_ROOT}/torch-cache}"
export TORCH_HOME

case "${VARIANT}" in
  grayscale)
    CONFIG="config/cloud/smoke_all_grayscale_convnext_tiny.yaml"
    RUN_DIR="${OUTPUT_ROOT}/cloud_smoke_all_grayscale_convnext_tiny"
    ;;
  rgb|rgb_multiwindow)
    CONFIG="config/cloud/smoke_all_rgb_multiwindow_convnext_tiny.yaml"
    RUN_DIR="${OUTPUT_ROOT}/cloud_smoke_all_rgb_multiwindow_convnext_tiny"
    ;;
  *)
    echo "Usage: $0 grayscale|rgb_multiwindow" >&2
    exit 2
    ;;
esac

python scripts/preflight.py \
  --config "${CONFIG}" \
  --check-wandb \
  --run-dir "${RUN_DIR}"

python scripts/train_classifier.py \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}"
