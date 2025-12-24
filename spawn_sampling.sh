#!/bin/bash

# List of GPUs you want to use
GPUS=(4)
# GPUS=(4 5)

JOB_DIR=runs/v1/inspect
SAMPLE_DIR=qwen2.5-math-1.5b/instruct.direct/sample_1
# SAMPLE_DIR=qwen2.5-math-7b/base.direct/sample_1
# SAMPLE_DIR=qwen2.5-math-7b/base.prefix/To_solve_this_problem
# SAMPLE_DIR=qwen2.5-math-7b/rlzero.prefix/To_determine_the
SPLIT=imo-bench-combinatorics-005

for GPU in "${GPUS[@]}"; do
    echo "Starting sampling for $SAMPLE_DIR on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python run_sampling.py \
        --job_dir $JOB_DIR \
        --split $SPLIT \
        --sampler_config_dir "$SAMPLE_DIR" \
        --gpu_id $GPU \
        --n 1024 &
done
wait
echo "Launched all processes!"
