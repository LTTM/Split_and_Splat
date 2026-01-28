#!/bin/bash

# Usage: ./run_all.sh /path/to/f

PARENT_FOLDER="$1"

if [ -z "$PARENT_FOLDER" ]; then
  echo "Usage: $0 <parent_folder>"
  exit 1
fi

for SUB in "./data/$PARENT_FOLDER/masks"/*/; do
    # Remove trailing slash for cleaner names
    FOLDER_NAME=$(basename "$SUB")

    echo "Running for folder: $FOLDER_NAME"

    python -u train.py \
        -s "./data/$PARENT_FOLDER/masks/$FOLDER_NAME" \
        -m "./output/$PARENT_FOLDER/ref/$FOLDER_NAME" \
        --iterations 1000 \
	 --is_instance \
        --test_iterations 500 1000  \
        --save_iteration 500 1000
done
