#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-${VARIANT:-}}"
REMOTE="${REMOTE:-s2:mammo-recall-data}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/processed_datasets}"

if [ -z "${VARIANT}" ]; then
  echo "Usage: $0 grayscale|rgb_multiwindow|both" >&2
  echo "Optional env: REMOTE=${REMOTE} DATA_ROOT=${DATA_ROOT}" >&2
  exit 2
fi

python scripts/download_data.py \
  --remote "${REMOTE}" \
  --data-root "${DATA_ROOT}" \
  --variant "${VARIANT}" \
  --execute
