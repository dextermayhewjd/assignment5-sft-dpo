#!/bin/bash
# grpo_group_standard_deviation：对比 use_std_normalization True vs False
# 固定上一步选出的最佳 length_norm（默认 masked_mean），只改 std normalization
# 对应作业 Problem (grpo_group_standard_deviation)

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
    --length_norm masked_mean
    --sampling_temperature 1.0
    --sampling_max_tokens 1024
    --eval_every 5
    --save_every 200
    --gpu_memory_utilization 0.75
    --wandb_project gsm8k-grpo-std-norm
"

echo "===== Run 1: use_std_normalization=True（默认 GRPO）====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --use_std_normalization \
    --output_dir checkpoints/std_norm_true

echo "===== Run 2: use_std_normalization=False（Dr. GRPO）====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --output_dir checkpoints/std_norm_false
# 注：不传 --use_std_normalization 则为 False（action=store_true 的默认值）
