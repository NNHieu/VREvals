#!/bin/bash

# List of GPUs you want to use
GPUS=(0 1 6 7)
# GPUS=(0)

JOB_DIR=runs/default/aime25.qwen2.5-math-7b

# Search all sampler_config_dir in job dir and start sample

# Find all directories at depth 2 from $JOB_DIR that match "sample_*"
SAMPLE_DIRS=($(find "$JOB_DIR" -mindepth 2 -maxdepth 2 -type d -name "sample_*" | sort))

for SAMPLE_DIR in "${SAMPLE_DIRS[@]}"; do
    for GPU in "${GPUS[@]}"; do
        echo "Starting sampling for $SAMPLE_DIR on GPU $GPU"
        # Get the sampler_config_dir as the path relative to $JOB_DIR
        SAMPLER_CONFIG_DIR=$(realpath --relative-to="$JOB_DIR" "$SAMPLE_DIR")
        CUDA_VISIBLE_DEVICES=$GPU python run_sampling.py \
            --job_dir $JOB_DIR \
            --sampler_config_dir "$SAMPLER_CONFIG_DIR" \
            --gpu_id $GPU \
            --n 256 &
    done
    echo "Launched all processes!"
    wait
done