"""
GSM8K SFT 训练脚本 (Algorithm 1)

用法 (2 GPU, 各 24GB):
    CUDA_VISIBLE_DEVICES=0,1 uv run python sft_scripts/train_sft.py \
        --num_examples 0 \
        --lr 1e-5 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --num_epochs 3

    num_examples=0 表示使用全部数据。
    batch_size 保持小（2），通过增大 gradient_accumulation_steps 来扩大等效 batch。

显存分配（24GB x 2）:
    GPU 0: policy 模型 (~3GB bf16) + 训练激活值 + 优化器状态
    GPU 1: vLLM 推理引擎 (enforce_eager, gpu_memory_utilization=0.70)
"""

import argparse
import json
import os
import sys
import random
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from sft_scripts.tokenize_prompt_and_output import tokenize_prompt_and_output
from sft_scripts.get_response_log_probs import get_response_log_probs
from sft_scripts.sft_microbatch_train_step import sft_microbatch_train_step
from sft_scripts.log_generations import log_generations


# ======================== vLLM 工具函数 ========================

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.70):
    """在指定 GPU 上启动 vLLM 推理引擎。"""
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,  # 跳过 CUDA graph capture，节省显存
        )


def load_policy_into_vllm_instance(policy, llm):
    """把训练中的 policy 权重同步到 vLLM 实例里。"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_with_vllm(llm, prompts, ground_truths, reward_fn, sampling_params):
    """用 vLLM 批量生成并评估。"""
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]

    total_reward = 0.0
    total_format = 0.0
    total_answer = 0.0
    for gen_text, gt in zip(generated_texts, ground_truths):
        info = reward_fn(gen_text, gt)
        total_reward += info["reward"]
        total_format += info["format_reward"]
        total_answer += info["answer_reward"]

    n = len(prompts)
    return {
        "avg_reward": total_reward / n,
        "avg_format_reward": total_format / n,
        "avg_answer_reward": total_answer / n,
    }


# ======================== 数据加载 ========================

def load_sft_data(path: str, num_examples: int = 0, seed: int = 42):
    """加载 SFT jsonl 数据，可选截取前 num_examples 条。"""
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    if num_examples > 0:
        random.seed(seed)
        random.shuffle(data)
        data = data[:num_examples]

    prompts = [d["prompt"] for d in data]
    responses = [d["response"] for d in data]
    return prompts, responses


# ======================== 主训练循环 ========================

def train(args):
    # --- 设备分配 ---
    policy_device = "cuda:0"
    vllm_device = "cuda:1"

    # 确保 policy 模型只在 GPU 0 上分配显存
    torch.cuda.set_device(policy_device)

    # --- 加载 tokenizer 和模型 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=policy_device,
    )

    # --- 加载训练和验证数据 ---
    train_prompts, train_responses = load_sft_data(args.train_data, args.num_examples, seed=args.seed)
    print(f"Training examples: {len(train_prompts)}")

    # 加载验证集 prompt 和 ground truth
    with open(args.val_data, "r") as f:
        val_data = [json.loads(line) for line in f]
    val_prompts = [d["prompt"] for d in val_data]
    val_ground_truths = []
    for d in val_data:
        resp = d["response"]
        gt = resp.split("<answer>")[-1].replace("</answer>", "").strip()
        val_ground_truths.append(gt)

    # --- 初始化 vLLM ---
    print("Initializing vLLM on", vllm_device)
    llm = init_vllm(args.model_id, vllm_device, seed=args.seed)
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    # --- 初始化 wandb ---
    wandb.init(
        project=args.wandb_project,
        name=f"sft_n{args.num_examples}_lr{args.lr}_bs{args.batch_size}x{args.gradient_accumulation_steps}",
        config=vars(args),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # --- 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # --- 训练循环 ---
    # effective_batch_size = batch_size * gradient_accumulation_steps
    # 例如 batch_size=2, grad_accum=8 → 等效 batch=16
    # 显存只需装下 2 条数据的激活值，但梯度效果等价于 16 条
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_step = 0
    eval_step = 0

    for epoch in range(args.num_epochs):
        indices = list(range(len(train_prompts)))
        random.seed(args.seed + epoch)
        random.shuffle(indices)

        for batch_start in range(0, len(indices), effective_batch_size):
            batch_indices = indices[batch_start:batch_start + effective_batch_size]
            if len(batch_indices) == 0:
                break

            # 计算当前 batch 实际的 accumulation 步数（最后一个 batch 可能不满）
            actual_accum_steps = (len(batch_indices) + args.batch_size - 1) // args.batch_size

            optimizer.zero_grad()
            total_loss = 0.0
            micro_count = 0

            # --- 梯度累积：分 microbatch 处理 ---
            for micro_start in range(0, len(batch_indices), args.batch_size):
                micro_indices = batch_indices[micro_start:micro_start + args.batch_size]
                micro_prompts = [train_prompts[i] for i in micro_indices]
                micro_responses = [train_responses[i] for i in micro_indices]

                # tokenize
                batch = tokenize_prompt_and_output(micro_prompts, micro_responses, tokenizer)
                input_ids = batch["input_ids"].to(policy_device)
                labels = batch["labels"].to(policy_device)
                response_mask = batch["response_mask"].to(policy_device)

                # 前向：获取 log probs
                result = get_response_log_probs(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                log_probs = result["log_probs"]

                # 反向：计算 loss 并累积梯度
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=actual_accum_steps,
                )
                total_loss += loss.item()
                micro_count += 1

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            avg_loss = total_loss / micro_count

            # --- 训练 logging ---
            if global_step % args.log_every == 0:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/epoch": epoch + batch_start / len(indices),
                    "train_step": global_step,
                })
                print(f"[Step {global_step}] loss={avg_loss:.4f}")

            # --- 定期评估 + log_generations ---
            if global_step % args.eval_every == 0:
                print(f"[Step {global_step}] Evaluating...")
                model.eval()
                load_policy_into_vllm_instance(model, llm)

                # 1. 全量验证集评估（只算 accuracy）
                eval_metrics = evaluate_with_vllm(
                    llm, val_prompts, val_ground_truths,
                    r1_zero_reward_fn, eval_sampling_params,
                )

                # 2. log_generations：抽样生成 + entropy + reward 详细 logging
                gen_result = log_generations(
                    policy_model=model,
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=val_prompts,
                    ground_truths=val_ground_truths,
                    reward_fn=r1_zero_reward_fn,
                    sampling_params=eval_sampling_params,
                    num_examples=args.num_log_examples,
                )

                eval_step += 1
                wandb.log({
                    # 全量评估指标
                    "eval/accuracy": eval_metrics["avg_answer_reward"],
                    "eval/format_reward": eval_metrics["avg_format_reward"],
                    "eval/reward": eval_metrics["avg_reward"],
                    # log_generations 的汇总统计
                    "eval/avg_entropy": gen_result["summary"]["avg_entropy"],
                    "eval/avg_response_length": gen_result["summary"]["avg_response_length"],
                    "eval/avg_correct_length": gen_result["summary"]["avg_correct_response_length"],
                    "eval/avg_incorrect_length": gen_result["summary"]["avg_incorrect_response_length"],
                    # 生成样例表格
                    "eval/generations": gen_result["table"],
                    "eval_step": eval_step,
                })
                print(f"  accuracy={eval_metrics['avg_answer_reward']:.4f}, "
                      f"format={eval_metrics['avg_format_reward']:.4f}, "
                      f"entropy={gen_result['summary']['avg_entropy']:.4f}, "
                      f"avg_len={gen_result['summary']['avg_response_length']:.0f}")

                model.train()

    # --- 最终评估 ---
    print("Final evaluation...")
    model.eval()
    load_policy_into_vllm_instance(model, llm)

    eval_metrics = evaluate_with_vllm(
        llm, val_prompts, val_ground_truths,
        r1_zero_reward_fn, eval_sampling_params,
    )
    gen_result = log_generations(
        policy_model=model,
        llm=llm,
        tokenizer=tokenizer,
        prompts=val_prompts,
        ground_truths=val_ground_truths,
        reward_fn=r1_zero_reward_fn,
        sampling_params=eval_sampling_params,
        num_examples=args.num_log_examples,
    )

    eval_step += 1
    wandb.log({
        "eval/accuracy": eval_metrics["avg_answer_reward"],
        "eval/format_reward": eval_metrics["avg_format_reward"],
        "eval/reward": eval_metrics["avg_reward"],
        "eval/avg_entropy": gen_result["summary"]["avg_entropy"],
        "eval/avg_response_length": gen_result["summary"]["avg_response_length"],
        "eval/avg_correct_length": gen_result["summary"]["avg_correct_response_length"],
        "eval/avg_incorrect_length": gen_result["summary"]["avg_incorrect_response_length"],
        "eval/generations": gen_result["table"],
        "eval_step": eval_step,
    })
    print(f"Final: accuracy={eval_metrics['avg_answer_reward']:.4f}, "
          f"format={eval_metrics['avg_format_reward']:.4f}, "
          f"entropy={gen_result['summary']['avg_entropy']:.4f}")

    # --- 保存模型 ---
    save_dir = os.path.join(args.output_dir, f"sft_n{args.num_examples}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/home/fredkeira/Data/models/Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data", type=str, default="data/gsm8k/sft_train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/gsm8k/sft_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_examples", type=int, default=0, help="0 = use all data")
    parser.add_argument("--lr", type=float, default=1e-5)
    # batch_size 保持小（2）以适配 24GB 显存，通过 gradient_accumulation_steps 扩大等效 batch
    parser.add_argument("--batch_size", type=int, default=2, help="Microbatch size (keep small for 24GB GPU)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="等效 batch = batch_size * 此值")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--num_log_examples", type=int, default=10, help="log_generations 每次抽样多少条")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="gsm8k-sft")
    args = parser.parse_args()
    train(args)
