#!/bin/bash

# List of GPUs you want to use
GPUS=(0 1 2 3 4 5 7)
# GPUS=(0)

# Loop through GPUs and launch a process on each
for GPU in "${GPUS[@]}"; do
    echo "Starting process on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python run_sampling.py \
        --job_dir runs/default/math500.qwen-1.5b-inst \
        --sampler_config_dir distilled-150.prefix/ThinkTo \
        --gpu_id $GPU \
        --n 5 &
done

echo "Launched all processes!"
wait