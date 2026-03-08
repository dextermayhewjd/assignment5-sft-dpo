#!/bin/bash
# grpo_off_policy：off-policy GRPO，epochs_per_rollout_batch > 1，使用 grpo_clip
# 固定 rollout_batch_size=256，对比 on-policy baseline 与 off-policy 变体
# 对应作业 Problem (grpo_off_policy) + (grpo_off_policy_sweep)
#
# 第一阶段：粗粒度扫描（<50 steps），探索参数空间
# 第二阶段：最优超参跑满 200 steps，对比 on-policy baseline

set -e

# ===== 固定参数 =====
ROLLOUT_BATCH_SIZE=256
GROUP_SIZE=8
# micro_batch_size 保持 1（显存限制），通过 gradient_accumulation_steps 控制等效 batch
MICRO_BATCH=1

BASE_ARGS="
    --learning_rate 1e-5
    --rollout_batch_size ${ROLLOUT_BATCH_SIZE}
    --group_size ${GROUP_SIZE}
    --loss_type grpo_clip
    --cliprange 0.2
    --use_std_normalization
    --length_norm masked_mean
    --sampling_temperature 1.0
    --sampling_max_tokens 1024
    --gpu_memory_utilization 0.75
    --wandb_project gsm8k-grpo-off-policy
"

# ===== 第一阶段：粗粒度扫描（最多 50 steps）=====
echo "===== 粗粒度扫描：epochs × train_batch_size ====="

# epochs=2, train_batch_size=256（micro=1, accum=256）
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --n_grpo_steps 50 --eval_every 5 --save_every 50 \
    --epochs_per_rollout_batch 2 \
    --train_batch_size 256 --gradient_accumulation_steps 256 \
    --output_dir checkpoints/offpolicy_sweep_ep2_bs256

# epochs=4, train_batch_size=256（micro=1, accum=256）
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --n_grpo_steps 50 --eval_every 5 --save_every 50 \
    --epochs_per_rollout_batch 4 \
    --train_batch_size 256 --gradient_accumulation_steps 256 \
    --output_dir checkpoints/offpolicy_sweep_ep4_bs256

# epochs=4, train_batch_size=128（同 rollout_batch 内多个 optimizer step，micro=1, accum=128）
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --n_grpo_steps 50 --eval_every 5 --save_every 50 \
    --epochs_per_rollout_batch 4 \
    --train_batch_size 128 --gradient_accumulation_steps 128 \
    --output_dir checkpoints/offpolicy_sweep_ep4_bs128

# ===== 第二阶段：最优超参跑满 200 steps（根据第一阶段结果填入）=====
# TODO: 根据粗扫结果，将最优 epochs_per_rollout_batch 和 train_batch_size 填入下面
BEST_EPOCHS=4
BEST_TRAIN_BS=256
BEST_ACCUM=256

echo "===== 精细实验：最优超参跑满 200 steps ====="
CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
    $BASE_ARGS \
    --n_grpo_steps 200 --eval_every 5 --save_every 50 \
    --epochs_per_rollout_batch ${BEST_EPOCHS} \
    --train_batch_size ${BEST_TRAIN_BS} \
    --gradient_accumulation_steps ${BEST_ACCUM} \
    --output_dir checkpoints/offpolicy_best
