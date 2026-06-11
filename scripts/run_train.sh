#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-${VARIANT:-grayscale}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/outputs}"
TORCH_HOME="${TORCH_HOME:-${OUTPUT_ROOT}/torch-cache}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
export TORCH_HOME

case "${VARIANT}" in
  grayscale)
    DEFAULT_CONFIG="config/cloud_exp/all_grayscale_convnext_tiny_1024.yaml"
    DEFAULT_RUN_NAME="all_grayscale_convnext_tiny_1024"
    ;;
  rgb|rgb_multiwindow)
    DEFAULT_CONFIG="config/cloud_exp/all_rgb_multiwindow_convnext_tiny_1024.yaml"
    DEFAULT_RUN_NAME="all_rgb_multiwindow_convnext_tiny_1024"
    ;;
  *)
    echo "Usage: $0 grayscale|rgb_multiwindow" >&2
    exit 2
    ;;
esac

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

python scripts/preflight.py \
  --config "${CONFIG}" \
  --check-wandb \
  --run-dir "${RUN_DIR}"

python scripts/train_classifier.py \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}"

if [ "${UPLOAD_RESULTS}" = "1" ]; then
  RUN_NAME="${RUN_NAME}" scripts/upload_results.sh "${VARIANT}" "${RUN_DIR}"
else
  echo "Skipping result upload because UPLOAD_RESULTS=${UPLOAD_RESULTS}"
fi
