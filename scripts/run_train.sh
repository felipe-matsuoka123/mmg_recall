#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-${VARIANT:-grayscale}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/outputs}"
TORCH_HOME="${TORCH_HOME:-${OUTPUT_ROOT}/torch-cache}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
export TORCH_HOME

case "${VARIANT}" in
  grayscale)
    CONFIG="config/cloud/all_grayscale_convnext_tiny_1024.yaml"
    RUN_DIR="${OUTPUT_ROOT}/all_grayscale_convnext_tiny_1024"
    ;;
  rgb|rgb_multiwindow)
    CONFIG="config/cloud/all_rgb_multiwindow_convnext_tiny_1024.yaml"
    RUN_DIR="${OUTPUT_ROOT}/all_rgb_multiwindow_convnext_tiny_1024"
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

if [ "${UPLOAD_RESULTS}" = "1" ]; then
  scripts/upload_results.sh "${VARIANT}" "${RUN_DIR}"
else
  echo "Skipping result upload because UPLOAD_RESULTS=${UPLOAD_RESULTS}"
fi
