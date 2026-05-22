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

# folder passed as first argument
f="$1"

# check folder exists
if [ ! -d "./output/$f/raw" ]; then
    echo "Folder '$f' not found!"
    exit 1
fi

for sub in "./output/$f/raw"/*/; do
    subname=$(basename "$sub")

    # Skip entries that were not successfully trained (no point_cloud/)
    [ ! -d "$sub/point_cloud" ] && continue

    echo "Processing: $subname"

    python ./utils_mask/mask_optimizer_scannet.py \
        -m "./output/$f/raw/$subname" \
        --instance_test "$subname" \
	--scene  "$f"
done
