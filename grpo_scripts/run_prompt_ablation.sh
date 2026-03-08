#!/bin/bash
# grpo_prompt_ablation：对比 r1_zero prompt vs question_only prompt
# prompt_type 同时控制数据加载方式和 reward function
# 对应作业 Problem (grpo_prompt_ablation)
#
# 注意：
#   r1_zero    → sft_train.jsonl（已格式化）+ r1_zero_reward_fn
#   question_only → 原始 train.jsonl（只有 question）+ question_only_reward_fn
#
# question_only 的 val_data 也需要是原始格式（test.jsonl）
# r1_zero 的 val_data 用 sft_val.jsonl

set -e

# 固定使用前面实验选出的最优超参（on-policy baseline）
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
    --length_norm masked_mean
    --sampling_temperature 1.0
    --sampling_max_tokens 1024
    --eval_every 5
    --save_every 200
    --gpu_memory_utilization 0.75
    --wandb_project gsm8k-grpo-prompt-ablation
"

echo "===== Run 1: r1_zero prompt + r1_zero_reward_fn ====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --prompt_type r1_zero \
    --train_data data/gsm8k/sft_train.jsonl \
    --val_data data/gsm8k/sft_val.jsonl \
    --output_dir checkpoints/prompt_ablation_r1zero

echo "===== Run 2: question_only prompt + question_only_reward_fn ====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --prompt_type question_only \
    --train_data data/gsm8k/train.jsonl \
    --val_data data/gsm8k/test.jsonl \
    --output_dir checkpoints/prompt_ablation_question_only
