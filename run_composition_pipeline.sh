################################################################################
# Split&Splat - Copyright (c) 2026, MEDIALab, University of Padova.
#
# Author(s):
#  Leonardo Monchieri (leonardo.monchieri@unipd.it)
#  Elena Camuffo (elenacamuffo97@gmail.com)
#  Francesco Barbato (francesco.barbato@dei.unipd.it)
#  Pietro Zanuttigh (zanuttigh@dei.unipd.it)
#  Simone Milani (simone.milani@dei.unipd.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################

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
