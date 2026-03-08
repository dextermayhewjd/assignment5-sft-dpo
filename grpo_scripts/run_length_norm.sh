#!/bin/bash
# grpo_length_normalization：对比 masked_mean vs masked_normalize
# 固定 reinforce_with_baseline，其他超参保持默认，只改 length_norm
# 对应作业 Problem (grpo_length_normalization)

set -e

BASE_ARGS="
    --n_grpo_steps 200
    --learning_rate 1e-5
    --rollout_batch_size 256
    --group_size 8
    --train_batch_size 256
    --gradient_accumulation_steps 256
    --epochs_per_rollout_batch 1
    --loss_type reinforce_with_baseline
    --use_std_normalization
    --sampling_temperature 1.0
    --sampling_max_tokens 1024
    --eval_every 5
    --save_every 200
    --gpu_memory_utilization 0.75
    --wandb_project gsm8k-grpo-length-norm
"

echo "===== Run 1: masked_mean ====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --length_norm masked_mean \
    --output_dir checkpoints/length_norm_mean

echo "===== Run 2: masked_normalize ====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --length_norm masked_normalize \
    --output_dir checkpoints/length_norm_normalize
