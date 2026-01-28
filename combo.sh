#!/bin/bash

PARENT_FOLDER=$1
OUT_FOLDER=$2
mkdir -p $OUT_FOLDER


# Define and actually run the python command
for SUB in "$PARENT_FOLDER"/*/; do
    # Remove trailing slash for cleaner names
    FOLDER_NAME=$(basename "$SUB")

    echo "Running for folder: $FOLDER_NAME"

    python -u train.py \
        -s "$PARENT_FOLDER/$FOLDER_NAME" \
        -m "$OUT_FOLDER/$FOLDER_NAME" \
        --iterations 1000 \
        --test_iterations 500 1000 \
        --save_iteration 500 1000 \
        --densify_from_iter 999999
done
