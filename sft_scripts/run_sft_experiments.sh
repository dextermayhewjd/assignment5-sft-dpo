#!/bin/bash
# 运行 SFT 实验：不同数据量对比
# batch_size=2 (适配 24GB 3090), gradient_accumulation_steps=8, 等效 batch=16
# 用法: CUDA_VISIBLE_DEVICES=0,1 bash sft_scripts/run_sft_experiments.sh

set -e

LR=1e-5
BS=2
GRAD_ACCUM=8
EPOCHS=3
EVAL_EVERY=50
LOG_EVERY=10

for N in 128 256 512 1024 0; do
    echo "=========================================="
    echo "Running SFT with num_examples=$N"
    echo "=========================================="
    uv run python sft_scripts/train_sft.py \
        --num_examples $N \
        --lr $LR \
        --batch_size $BS \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $EPOCHS \
        --eval_every $EVAL_EVERY \
        --log_every $LOG_EVERY \
        --num_log_examples 10 \
        --wandb_project gsm8k-sft
done

echo "All experiments done."
