#!/bin/bash

# List of GPUs you want to use
GPUS=(1 2 3 4 5 7)

# Loop through GPUs and launch a process on each
for GPU in "${GPUS[@]}"; do
    echo "Starting process on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python run_sampling.py \
        --job_dir runs/default/gsm8k.qwen-1.5b-inst \
        --sampler_config_dir distilled-100.direct/sample_1 \
        --n 2 &
done

echo "Launched all processes!"
wait