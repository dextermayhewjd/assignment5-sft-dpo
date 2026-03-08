#!/bin/bash
# grpo_off_policy_clip_ablation：消融 clip，对比 grpo_clip vs grpo_no_clip
# 使用上一步 off-policy 实验中最优的超参
# 对应作业 Problem (grpo_off_policy_clip_ablation)

set -e

# 根据 run_off_policy.sh 第一阶段结果填入最优超参
BEST_EPOCHS=4
BEST_TRAIN_BS=256
BEST_ACCUM=256

BASE_ARGS="
    --n_grpo_steps 200
    --learning_rate 1e-5
    --rollout_batch_size 256
    --group_size 8
    --train_batch_size ${BEST_TRAIN_BS}
    --gradient_accumulation_steps ${BEST_ACCUM}
    --epochs_per_rollout_batch ${BEST_EPOCHS}
    --use_std_normalization
    --length_norm masked_mean
    --sampling_temperature 1.0
    --sampling_max_tokens 1024
    --eval_every 5
    --save_every 200
    --gpu_memory_utilization 0.75
    --wandb_project gsm8k-grpo-clip-ablation
"

echo "===== Run 1: grpo_clip（有截断，off-policy 标准做法）====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --loss_type grpo_clip \
    --cliprange 0.2 \
    --output_dir checkpoints/clip_ablation_clip

echo "===== Run 2: grpo_no_clip（无截断，消融实验）====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --loss_type grpo_no_clip \
    --output_dir checkpoints/clip_ablation_noclip
