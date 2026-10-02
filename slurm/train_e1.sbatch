#!/bin/bash
# Supply partition, GPU request, time, CPU/RAM and logs through sbatch flags.
set -euo pipefail
: "${THEME26_REPO:?Set the absolute repository path}"
: "${THEME26_PYTHON:?Set the server environment python executable}"
: "${THEME26_CONFIG:?Set the experiment config path}"
: "${THEME26_GRAYSCALE_RAW:?Set the Berea grayscale RAW path}"
cd "$THEME26_REPO"
args=(scripts/train.py --config "$THEME26_CONFIG" --device cuda)
if [[ -n "${THEME26_RESUME:-}" ]]; then
  args+=(--resume "$THEME26_RESUME")
fi
"$THEME26_PYTHON" "${args[@]}"
