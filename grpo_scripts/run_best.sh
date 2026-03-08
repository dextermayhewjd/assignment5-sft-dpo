#!/bin/bash
# grpo_best：最优组合实验
# masked_normalize + no std norm (Dr. GRPO) + off-policy (epochs=4) + grpo_clip + r1_zero prompt
# 对比基线（之前的 reinforce_with_baseline on-policy run，val/accuracy=43.3%）
#  ┌──────────────────────────┬─────────────────────────┬──────────────────┐                
# │           参数           │      基线 (43.3%)       │       本次       │                
#  ├──────────────────────────┼─────────────────────────┼──────────────────┤                
# │ loss_type                │ reinforce_with_baseline │ grpo_clip        │                
# ├──────────────────────────┼─────────────────────────┼──────────────────┤                
# │ length_norm              │ masked_mean             │ masked_normalize │                
# ├──────────────────────────┼─────────────────────────┼──────────────────┤                
# │ use_std_normalization    │ True                    │ False            │                
# ├──────────────────────────┼─────────────────────────┼──────────────────┤                
# │ epochs_per_rollout_batch │ 1 (on-policy)           │ 4 (off-policy)   │                
# ├──────────────────────────┼─────────────────────────┼──────────────────┤                
# │ prompt_type              │ r1_zero                 │ r1_zero          │                
# └──────────────────────────┴─────────────────────────┴──────────────────┘     
set -e

CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    --n_grpo_steps 200 \
    --learning_rate 1e-5 \
    --rollout_batch_size 256 \
    --group_size 8 \
    --train_batch_size 256 \
    --gradient_accumulation_steps 256 \
    --epochs_per_rollout_batch 4 \
    --loss_type grpo_clip \
    --cliprange 0.2 \
    --length_norm masked_normalize \
    --sampling_temperature 1.0 \
    --sampling_max_tokens 1024 \
    --eval_every 5 \
    --save_every 200 \
    --gpu_memory_utilization 0.75 \
    --prompt_type r1_zero \
    --train_data data/gsm8k/sft_train.jsonl \
    --val_data data/gsm8k/sft_val.jsonl \
    --output_dir checkpoints/grpo_best \
    --wandb_project gsm8k-grpo-best
