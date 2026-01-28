#!/usr/bin/env bash

set -e  # exit immediately if a command fails

# ---- argument parsing ----
if [[ "$1" != "--scene" || -z "$2" ]]; then
    echo "Usage: $0 --scene <scene_x>"
    exit 1
fi

SCENE="$2"

# ---- commands ----
PYTHON="python"

echo "Running pipeline for scene: ${SCENE}"

echo "Step 1: move_PLY.py"
$PYTHON move_PLY.py --scene "${SCENE}" --output "tmp"

echo "Step 2: mask_combination.py"
$PYTHON ./utils_mask/mask_combination.py --scene "${SCENE}"

echo "Step 3: sbatch"
    ./combo.sh \
    "./data/${SCENE}/masks/combined" \
    "./output/${SCENE}/comb" 

echo "Pipeline submitted successfully for scene: ${SCENE}"
