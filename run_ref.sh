#!/bin/bash

# folder passed as first argument
f="$1"

# check folder exists
if [ ! -d "./output/$f/raw" ]; then
    echo "Folder '$f' not found!"
    exit 1
fi

for sub in "./output/$f/raw"/*/; do
    # remove trailing slash to get clean name
    subname=$(basename "$sub")

    echo "Processing: $subname"

    python ./utils_mask/mask_optimizer_scannet.py \
        -m "./output/$f/raw/$subname" \
        --instance_test "$subname" \
	--scene  "$f"
done
