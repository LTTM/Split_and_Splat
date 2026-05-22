#!/bin/bash


PARENT_FOLDER="$1"

# Path to the parent folder containing the subfolders
PARENT_DIR="./data/$PARENT_FOLDER/masks"

# Path to the source folder and file you want to copy
SOURCE_IMAGES="./data/$PARENT_FOLDER/images"
SOURCE_JSON="./data/$PARENT_FOLDER/transforms_train.json"

SOURCE_SPARSE="./data/$PARENT_FOLDER/sparse"

# Discard folder
DISCARD_DIR="./data/$PARENT_FOLDER/discard"
mkdir -p "$DISCARD_DIR"

# Loop through each subfolder in the parent directory
for SUB in "$PARENT_DIR"/*/; do
    echo "Copying into: $SUB"

    # Copy the images folder
    cp -r "$SOURCE_IMAGES" "$SUB"

   set -e

    {
        echo "Trying to copy sparse folder..."
        cp -r "$SOURCE_SPARSE" "$SUB"
    } || {
        echo "Sparse copy failed, trying JSON..."
        cp "$SOURCE_JSON" "$SUB"
    }
    

    # Move masks into the folder and rename the ply
    echo "Processing $SUB"

    # 1. Create masks/ folder if not exists
    mkdir -p "$SUB/masks"

    # 2. Move all PNG images into masks/
    mv "$SUB"/*.png "$SUB/masks/" 2>/dev/null

    # 3. Rename the PLY file to points3d.ply (only if exists)
    for FILE in "$SUB"/*.ply; do
        if [ -f "$FILE" ]; then
            mv "$FILE" "$SUB/points3d.ply"
        fi
    done


    # Count number of images in masks/
    NUM_MASKS=$(find "$SUB/masks" -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)

    # If fewer than 2 images, move folder to discard
    if [ "$NUM_MASKS" -lt 2 ]; then
        echo "→ Less than 2 mask images ($NUM_MASKS). Moving to discard."
        mv "$SUB" "$DISCARD_DIR"
    fi

done

echo "Done!"
