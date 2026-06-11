#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-${VARIANT:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/outputs}"
REMOTE="${REMOTE:-s2:mammo-recall-data}"

if [ -z "${VARIANT}" ]; then
  echo "Usage: $0 grayscale|rgb_multiwindow [run_dir]" >&2
  echo "Optional env: REMOTE=${REMOTE} OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
  exit 2
fi

case "${VARIANT}" in
  grayscale)
    DEFAULT_RUN_NAME="all_grayscale_convnext_tiny_1024"
    ;;
  rgb|rgb_multiwindow)
    DEFAULT_RUN_NAME="all_rgb_multiwindow_convnext_tiny_1024"
    ;;
  *)
    echo "Usage: $0 grayscale|rgb_multiwindow [run_dir]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
RUN_DIR="${2:-${OUTPUT_ROOT}/${RUN_NAME}}"
DESTINATION="${REMOTE}/runs/${RUN_NAME}"

if [ ! -d "${RUN_DIR}" ]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is not on PATH" >&2
  exit 1
fi

echo "Uploading ${RUN_DIR} to ${DESTINATION}"
rclone copy "${RUN_DIR}" "${DESTINATION}" --progress
