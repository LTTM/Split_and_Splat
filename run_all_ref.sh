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


#!/bin/bash

# Usage: ./run_all.sh /path/to/f

PARENT_FOLDER="$1"

if [ -z "$PARENT_FOLDER" ]; then
  echo "Usage: $0 <parent_folder>"
  exit 1
fi

for SUB in "./data/$PARENT_FOLDER/masks"/*/; do
    FOLDER_NAME=$(basename "$SUB")

    # Skip non-instance folders (e.g. combined/, images/)
    [ ! -f "$SUB/transforms_train.json" ] && continue

    echo "Running for folder: $FOLDER_NAME"

    python -u train.py \
        -s "./data/$PARENT_FOLDER/masks/$FOLDER_NAME" \
        -m "./output/$PARENT_FOLDER/ref/$FOLDER_NAME" \
        --iterations 1000 \
	 --is_instance \
        --test_iterations 500 1000  \
        --save_iteration 500 1000
done
