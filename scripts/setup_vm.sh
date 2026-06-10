#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/data/processed_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/outputs}"
TORCH_HOME="${TORCH_HOME:-${OUTPUT_ROOT}/torch-cache}"

mkdir -p "${DATA_ROOT}/rsna" "${DATA_ROOT}/spr" "${DATA_ROOT}/vindr"
mkdir -p "${OUTPUT_ROOT}" "${TORCH_HOME}"

if [ ! -e processed_datasets ]; then
  ln -s "${DATA_ROOT}" processed_datasets
elif [ -L processed_datasets ]; then
  current_target="$(readlink -f processed_datasets)"
  expected_target="$(readlink -f "${DATA_ROOT}")"
  if [ "${current_target}" != "${expected_target}" ]; then
    echo "processed_datasets points to ${current_target}, expected ${expected_target}" >&2
    exit 1
  fi
else
  echo "processed_datasets exists and is not a symlink; leaving it unchanged." >&2
fi

echo "export TORCH_HOME=${TORCH_HOME}"
echo
echo "Add this to your shell before training:"
echo "  export TORCH_HOME=${TORCH_HOME}"
echo
echo "Setup complete:"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  TORCH_HOME=${TORCH_HOME}"
