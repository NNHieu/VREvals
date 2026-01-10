#!/bin/bash

# List of GPUs you want to use
GPUS=(0 1 2 3 4 5 6 7)
# GPUS=(0 3 4 5 6 7)
# GPUS=(1)

# JOB_DIR=runs/gsm8k.gemma-2b
# JOB_DIR=runs/gsm8k.qwen2.5-0.5b-mutual_metaninstruct1
# JOB_DIR=runs/legostar.qwen2.5-0.5b
# SPLIT=imo-bench-combinatorics-005
JOB_DIR=runs/gsm8k.evolm-1b-exclusive_styles
SPLIT=test

# Search all sampler_config_dir in job dir and start sample

# Find all directories at depth 2 from $JOB_DIR that match "sample_*"
SAMPLE_DIRS=($(find "$JOB_DIR" -mindepth 2 -maxdepth 2 -type d -wholename "*/sample*" | sort))

for SAMPLE_DIR in "${SAMPLE_DIRS[@]}"; do
    for GPU in "${GPUS[@]}"; do
        echo "Starting sampling for $SAMPLE_DIR on GPU $GPU"
        # Get the sampler_config_dir as the path relative to $JOB_DIR
        SAMPLER_CONFIG_DIR=$(realpath --relative-to="$JOB_DIR" "$SAMPLE_DIR")
        CUDA_VISIBLE_DEVICES=$GPU python run_sampling.py \
            --job_dir $JOB_DIR \
            --split $SPLIT \
            --sampler_config_dir "$SAMPLER_CONFIG_DIR" \
            --gpu_id $GPU \
            --n 8 &
    done
    wait
    echo "Launched all processes!"
done