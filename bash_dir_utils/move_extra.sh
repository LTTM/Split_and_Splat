#!/bin/bash

# Ask user for base directory
read -p "Enter the path to your base folder: " BASE_DIR

# Check if the directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Directory does not exist!"
    exit 1
fi

# Loop through each folder
for folder in "$BASE_DIR"/*; do
    if [ -d "$folder/mask_extra" ] && [ -d "$folder/mask" ]; then
        echo "Moving contents of $folder/mask_extra to $folder/mask"

        # Move all files and folders, including hidden ones
        shopt -s dotglob  # include hidden files
        mv "$folder/mask_extra/"* "$folder/mask/"

        # Remove the empty mask_extra folder
        rmdir "$folder/mask_extra"
    fi
done

echo "Done!"
