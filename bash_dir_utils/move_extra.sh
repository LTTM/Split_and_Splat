#!/bin/bash

BASE_DIR="$1"

if [ -z "$BASE_DIR" ]; then
    echo "Usage: $0 <base_folder>"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Directory does not exist!"
    exit 1
fi

for folder in "$BASE_DIR"/*; do
    if [ -d "$folder/mask_extra" ] && [ -d "$folder/mask" ]; then
        echo "Moving contents of $folder/mask_extra to $folder/mask"

        shopt -s dotglob
        mv "$folder/mask_extra/"* "$folder/mask/"

        rmdir "$folder/mask_extra"
    fi
done

echo "Done!"
