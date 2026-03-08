"""
GRPO (Group Relative Policy Optimization) 训练脚本

核心思路（对比 SFT 和 EI）：
    SFT：用固定的人工标注数据训练，数据从头到尾不变。
    EI ：让模型做题 → 只留答对的 → 在答对的上做 SFT → 循环。
    GRPO：让模型做题 → 用 reward 打分 → 组内归一化得到 advantage → 用策略梯度更新。
         不过滤数据，而是用 advantage 加权——答得好的响应获得正 advantage，答得差的获得负 advantage。

算法（Algorithm 3）：
    for step = 1..n_grpo_steps:
        1. 从题库中采样一批 question
        2. 设置旧策略 π_old ← π_θ
        3. 用 π_old 对每个 question 采样 G 个 response（rollout）
        4. 用 reward_fn 对每个 response 打分
        5. 组内归一化得到 advantage
        6. for epoch = 1..epochs_per_rollout_batch:
              用策略梯度（naive / reinforce_with_baseline / grpo_clip）更新 π_θ

用法:
    CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \
        --loss_type reinforce_with_baseline --n_grpo_steps 200
"""

import json
import os
import sys
import random
import re
from typing import Literal
from unittest.mock import patch

# 把项目根目录加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from sft_scripts.tokenize_prompt_and_output import tokenize_prompt_and_output
from sft_scripts.get_response_log_probs import get_response_log_probs
from sft_scripts.log_generations import log_generations
from grpo_scripts.compute_group_normalized_rewards import compute_group_normalized_rewards
from grpo_scripts.grpo_microbatch_train_step import grpo_microbatch_train_step


# ======================== vLLM 工具函数 ========================

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
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
            enforce_eager=True,
        )


def load_policy_into_vllm_instance(policy, llm):
    """把训练中的 policy 权重同步到 vLLM。"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_with_vllm(llm, prompts, ground_truths, reward_fn, sampling_params):
    """在验证集上评估，返回平均 reward。"""
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    total_reward = total_format = total_answer = 0.0
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

def load_sft_data(path: str):
    """从 sft_train.jsonl / sft_val.jsonl 加载数据。

    每行格式: {"prompt": "...(r1_zero模板)...", "response": "...<answer>72</answer>"}
    prompt 已经套好 r1_zero 模板，ground_truth 从 response 的 <answer> 标签提取。
    """
    prompts = []        # 已格式化的 r1_zero prompt
    ground_truths = []  # 从 <answer> 标签提取的答案
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            prompts.append(d["prompt"])
            # 从 response 中提取 <answer>...</answer> 里的内容作为 ground truth
            resp = d["response"]
            if "<answer>" in resp:
                gt = resp.split("<answer>")[-1].replace("</answer>", "").strip()
            else:
                gt = resp.strip()
            ground_truths.append(gt)
    return prompts, ground_truths


def load_raw_questions(path: str, prompt_template: str):
    """从原始 jsonl（只有 question/answer）加载数据，用于 question_only prompt。"""
    prompts = []
    ground_truths = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            question = d.get("question") or d.get("problem") or ""
            raw_answer = d.get("answer") or d.get("solution") or ""
            if "####" in raw_answer:
                gt = re.sub(r"<<.*?>>", "", raw_answer).split("####")[-1].strip()
            elif "<answer>" in raw_answer:
                gt = raw_answer.split("<answer>")[-1].replace("</answer>", "").strip()
            else:
                gt = raw_answer.strip()
            prompt = prompt_template.format(question=question)
            prompts.append(prompt)
            ground_truths.append(gt)
    return prompts, ground_truths


# ======================== Rollout ========================

def do_rollout(llm, prompts, ground_truths, group_size, sampling_params):
    """对一批 prompt 做 rollout，每个 prompt 生成 group_size 个 response。

    【对比 EI】EI 的 rollout 会过滤只留答对的；GRPO 不过滤，所有响应都参与训练，
    通过 advantage 的正负来区分好坏响应。

    返回:
        rollout_prompts: 重复后的 prompt 列表，长度 = len(prompts) * group_size
        rollout_responses: 所有生成的 response，长度同上
        repeated_ground_truths: 重复后的 ground truth，长度同上
    """
    # vLLM 的 n 参数让每个 prompt 生成 group_size 个不同响应
    outputs = llm.generate(prompts, sampling_params)

    rollout_prompts = []       # 重复 group_size 次的 prompt
    rollout_responses = []     # 所有生成的 response
    repeated_ground_truths = []  # 重复 group_size 次的 ground truth

    for prompt, gt, output in zip(prompts, ground_truths, outputs):
        for candidate in output.outputs:  # output.outputs 长度为 group_size
            rollout_prompts.append(prompt)
            rollout_responses.append(candidate.text)
            repeated_ground_truths.append(gt)

    return rollout_prompts, rollout_responses, repeated_ground_truths



# ======================== GRPO 训练主循环 ========================

def train(args):
    # --- 设备分配 ---
    policy_device = "cuda:0"   # GPU 0：训练 policy 模型
    vllm_device = "cuda:1"     # GPU 1：vLLM 推理（rollout + 评估）
    torch.cuda.set_device(policy_device)

    # --- 根据 prompt_type 同时确定数据加载方式和 reward function ---
    # r1_zero：使用预格式化 sft jsonl + 严格要求 </think> <answer>...</answer> 格式
    # question_only：从原始 jsonl 加载 + 只检查 \boxed{} 格式
    if args.prompt_type == "question_only":
        reward_fn = question_only_reward_fn
    else:
        reward_fn = r1_zero_reward_fn

    # --- 加载数据 ---
    if args.prompt_type == "question_only":
        # question_only prompt：从原始 jsonl 加载，只有 question + answer
        prompt_template_path = os.path.join(
            os.path.dirname(__file__), "..", "cs336_alignment", "prompts", "question_only.prompt"
        )
        with open(prompt_template_path) as f:
            prompt_template = f.read()
        train_prompts, train_gts = load_raw_questions(args.train_data, prompt_template)
        val_prompts, val_gts = load_raw_questions(args.val_data, prompt_template)
    else:
        # r1_zero prompt：使用预格式化的 sft jsonl，prompt 已套好 r1_zero 模板
        train_prompts, train_gts = load_sft_data(args.train_data)
        val_prompts, val_gts = load_sft_data(args.val_data)
    print(f"训练集: {len(train_prompts)} 条, 验证集: {len(val_prompts)} 条")

    # --- 加载模型和 tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=policy_device,
    )

    # --- 初始化 vLLM ---
    print(f"初始化 vLLM (设备: {vllm_device})...")
    llm = init_vllm(args.model_id, vllm_device, seed=args.seed,
                     gpu_memory_utilization=args.gpu_memory_utilization)

    # --- 优化器 ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,       # GRPO 论文中不用 weight decay
        betas=(0.9, 0.95),
    )

    # --- 采样参数 ---
    # rollout 用：每个 prompt 生成 group_size 个响应
    rollout_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,  # 防止空字符串响应
        n=args.group_size,                     # 每个 prompt 采样 group_size 个
        stop=["</answer>"],                    # 遇到 </answer> 停止
        include_stop_str_in_output=True,       # 保留 </answer> 在输出中
        seed=args.seed,
    )
    # 评估用：每个 prompt 只生成 1 个响应
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        min_tokens=4, stop=["</answer>"], include_stop_str_in_output=True,
    )

    # --- 常量和 sanity check ---
    rollout_batch_size = args.rollout_batch_size           # 总 rollout 响应数
    group_size = args.group_size                            # 每个 prompt 的响应数
    train_batch_size = args.train_batch_size                # 训练 batch 大小
    gradient_accumulation_steps = args.gradient_accumulation_steps

    assert train_batch_size % gradient_accumulation_steps == 0, \
        "train_batch_size 必须能被 gradient_accumulation_steps 整除"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps  # 每个 microbatch 的大小

    assert rollout_batch_size % group_size == 0, \
        "rollout_batch_size 必须能被 group_size 整除"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size  # 每个 rollout batch 的 prompt 数

    assert train_batch_size >= group_size, \
        "train_batch_size 必须 >= group_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size  # 每个 rollout batch 的 microbatch 数

    # --- 初始化 wandb ---
    wandb.init(
        project=args.wandb_project,
        name=f"grpo_{args.loss_type}_lr{args.learning_rate}_gs{group_size}",
        config=vars(args),
    )
    wandb.define_metric("grpo_step")
    wandb.define_metric("train/*", step_metric="grpo_step")
    wandb.define_metric("val/*", step_metric="grpo_step")

    # ======================== GRPO 主循环（Algorithm 3）========================
    for step in range(1, args.n_grpo_steps + 1):
        print(f"\n{'='*60}")
        print(f"GRPO Step {step}/{args.n_grpo_steps}")
        print(f"{'='*60}")

        # ===== 第1步：采样一批 question =====
        # 从训练集中随机选 n_prompts_per_rollout_batch 个 prompt
        random.seed(args.seed + step)
        db_indices = random.sample(
            range(len(train_prompts)),
            min(n_prompts_per_rollout_batch, len(train_prompts)),
        )
        batch_prompts = [train_prompts[i] for i in db_indices]  # 本步用的 prompt
        batch_gts = [train_gts[i] for i in db_indices]          # 对应的 ground truth

        # ===== 第2步：设置旧策略 π_old ← π_θ，同步权重到 vLLM =====
        model.eval()
        load_policy_into_vllm_instance(model, llm)

        # ===== 第3步：用 π_old 对每个 prompt 采样 G 个 response =====
        print(f"  Rollout: {len(batch_prompts)} prompts × {group_size} responses...")
        rollout_prompts, rollout_responses, repeated_gts = do_rollout(
            llm, batch_prompts, batch_gts, group_size, rollout_sampling_params,
        )

        # ===== 第4+5步：计算 reward 并组内归一化得到 advantage =====
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )
        print(f"  Reward: mean={reward_metadata['reward/mean']:.4f}, "
              f"std={reward_metadata['reward/std']:.4f}, "
              f"answer_reward={reward_metadata['answer_reward/mean']:.4f}")

        # ===== 预计算旧策略的 log_probs（GRPO-Clip 需要）=====
        # 在 off-policy 设置中只计算一次，多个 epoch 复用
        # 关键：不对 old_log_probs 求梯度
        # 做法：按 microbatch 大小分组 tokenize，每组的 padding 长度一致，
        #       计算 old_log_probs 后按原始 index 存储
        old_log_probs_cache = {}  # key=样本index, value=(log_probs, response_mask) 都在 CPU 上
        if args.loss_type == "grpo_clip":
            print("  计算旧策略 log_probs...")
            model.eval()
            with torch.no_grad():
                for i in range(0, rollout_batch_size, micro_train_batch_size):
                    end = min(i + micro_train_batch_size, rollout_batch_size)
                    mb_prompts = rollout_prompts[i:end]
                    mb_responses = rollout_responses[i:end]
                    batch = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                    input_ids = batch["input_ids"].to(policy_device)
                    labels = batch["labels"].to(policy_device)
                    result = get_response_log_probs(
                        model=model, input_ids=input_ids, labels=labels,
                        return_token_entropy=False,
                    )
                    # 逐样本存储到 cache（因为不同 microbatch padding 不同）
                    for j in range(end - i):
                        old_log_probs_cache[i + j] = result["log_probs"][j].detach().cpu()
            model.train()

        # ===== 第6步：策略梯度更新 =====
        model.train()

        for epoch in range(args.epochs_per_rollout_batch):
            # 打乱 rollout 的顺序
            indices = list(range(rollout_batch_size))
            random.seed(args.seed + step * 1000 + epoch)
            random.shuffle(indices)

            epoch_loss = 0.0
            epoch_entropy = 0.0    # 累积 token entropy
            epoch_clip_frac = 0.0  # 累积 clip fraction（仅 grpo_clip 用）
            n_updates = 0
            total_micro = 0        # 总 microbatch 计数（用于平均 entropy 和 clip_frac）

            # 按 train_batch_size 分批，每批做一次 optimizer.step()
            for batch_start in range(0, rollout_batch_size, train_batch_size):
                batch_indices = indices[batch_start:batch_start + train_batch_size]
                if len(batch_indices) == 0:
                    break

                # 实际的 gradient_accumulation_steps（最后一个 batch 可能不满）
                actual_accum = max(1, (len(batch_indices) + micro_train_batch_size - 1) // micro_train_batch_size)

                optimizer.zero_grad()  # 清空梯度
                batch_loss = 0.0
                micro_count = 0

                # 遍历 microbatch
                for micro_start in range(0, len(batch_indices), micro_train_batch_size):
                    micro_idx = batch_indices[micro_start:micro_start + micro_train_batch_size]
                    if len(micro_idx) == 0:
                        break

                    # 取出当前 microbatch 的 prompt、response
                    micro_prompts = [rollout_prompts[i] for i in micro_idx]
                    micro_responses = [rollout_responses[i] for i in micro_idx]

                    # tokenize（同一批样本一起 tokenize，padding 长度一致）
                    batch = tokenize_prompt_and_output(micro_prompts, micro_responses, tokenizer)
                    input_ids = batch["input_ids"].to(policy_device)
                    labels = batch["labels"].to(policy_device)
                    response_mask = batch["response_mask"].to(policy_device)

                    # 前向传播：获取当前策略的逐 token log 概率和 entropy
                    result = get_response_log_probs(
                        model=model, input_ids=input_ids, labels=labels,
                        return_token_entropy=True,
                    )
                    policy_log_probs = result["log_probs"]  # (micro_bs, seq_len)

                    # 计算 response 部分的平均 token entropy
                    if result.get("token_entropy") is not None:
                        masked_ent = (result["token_entropy"] * response_mask).sum() / response_mask.sum().clamp(min=1)
                        epoch_entropy += masked_ent.item()

                    # 取出当前 microbatch 的 advantage 和 raw_reward
                    micro_advantages = advantages[micro_idx].to(policy_device).unsqueeze(1)  # (micro_bs, 1)
                    micro_raw_rewards = raw_rewards[micro_idx].to(policy_device).unsqueeze(1)  # (micro_bs, 1)

                    # 准备旧策略 log_probs（grpo_clip 模式需要）
                    micro_old_log_probs = None
                    if args.loss_type == "grpo_clip":
                        # 预计算的 old_log_probs 是按原始顺序单独 tokenize 的，
                        # 但当前 microbatch 的 padding 可能不同。
                        # 最安全的做法：用同一批 input_ids 重新前向传播旧模型。
                        # 但旧模型权重已经被更新了（epoch>0 时）。
                        # 所以我们用当前 microbatch 的 tokenize 结果重新计算，
                        # 但这需要保存一份旧权重——这太复杂了。
                        #
                        # 实际做法（和大多数 GRPO 实现一致）：
                        # 预计算 old_log_probs 时和训练时用相同的 microbatch 分组。
                        # 在 on-policy（epochs=1）时，old_log_probs 就是训练前的模型输出。
                        # 在 off-policy（epochs>1）时，old_log_probs 在第一个 epoch 前计算，
                        # 后续 epoch 的 ratio 相对于同一个 old policy。
                        #
                        # 由于 shuffle 后 microbatch 组合变了（padding 不同），
                        # 我们需要用当前 input_ids 重新计算 old_log_probs。
                        # 但此时模型已经更新过了，所以这里有一个近似。
                        #
                        # 更好的方案：不 shuffle，保持和预计算时相同的分组。
                        # 这里我们采用简化方案：cache 中存的是原始长度的 log_probs，
                        # 重新 pad 到当前 seq_len。
                        seq_len = policy_log_probs.shape[1]
                        old_lps = []
                        for idx in micro_idx:
                            cached = old_log_probs_cache[idx]  # (cached_seq_len,)
                            # 截断或 pad 到当前 seq_len
                            if cached.shape[0] >= seq_len:
                                old_lps.append(cached[:seq_len])
                            else:
                                padded = torch.zeros(seq_len)
                                padded[:cached.shape[0]] = cached
                                old_lps.append(padded)
                        micro_old_log_probs = torch.stack(old_lps).to(policy_device)  # (micro_bs, seq_len)
                        # detach 确保不对 old_log_probs 求梯度
                        micro_old_log_probs = micro_old_log_probs.detach()

                    # 调用 microbatch 训练步骤：计算 loss + backward
                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=actual_accum,
                        loss_type=args.loss_type,
                        raw_rewards=micro_raw_rewards,
                        advantages=micro_advantages,
                        old_log_probs=micro_old_log_probs,
                        cliprange=args.cliprange,
                        length_norm=args.length_norm,  # masked_mean 或 masked_normalize
                    )

                    batch_loss += loss.item()
                    micro_count += 1
                    total_micro += 1

                    # 累积 clip fraction（grpo_clip 模式下 meta 中有 "clipped" mask）
                    if "clipped" in meta and response_mask is not None:
                        clip_frac = (meta["clipped"] * response_mask).sum() / response_mask.sum().clamp(min=1)
                        epoch_clip_frac += clip_frac.item()

                # 梯度裁剪：防止梯度爆炸，clip 值为 1.0
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 参数更新
                optimizer.step()
                n_updates += 1
                epoch_loss += batch_loss / micro_count if micro_count > 0 else 0.0

            avg_epoch_loss = epoch_loss / n_updates if n_updates > 0 else 0.0
            avg_epoch_entropy = epoch_entropy / total_micro if total_micro > 0 else 0.0
            avg_epoch_clip_frac = epoch_clip_frac / total_micro if total_micro > 0 else 0.0
            print(f"  Epoch {epoch+1}/{args.epochs_per_rollout_batch}: "
                  f"loss={avg_epoch_loss:.6f}, grad_norm={grad_norm:.4f}, "
                  f"entropy={avg_epoch_entropy:.4f}"
                  + (f", clip_frac={avg_epoch_clip_frac:.4f}" if args.loss_type == "grpo_clip" else ""))

        # ===== 日志记录 =====
        log_dict = {
            "train/loss": avg_epoch_loss,                      # 损失
            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,  # 梯度范数
            "train/token_entropy": avg_epoch_entropy,          # response token 的平均 entropy
            "train/reward_mean": reward_metadata["reward/mean"],        # 训练 reward 均值
            "train/reward_std": reward_metadata["reward/std"],          # 训练 reward 标准差
            "train/reward_max": reward_metadata["reward/max"],          # 训练 reward 最大值
            "train/reward_min": reward_metadata["reward/min"],          # 训练 reward 最小值
            "train/format_reward_mean": reward_metadata["format_reward/mean"],  # 格式 reward
            "train/answer_reward_mean": reward_metadata["answer_reward/mean"],  # 答案 reward
            "train/advantage_mean": reward_metadata["advantages/mean"],  # 优势值均值
            "train/advantage_std": reward_metadata["advantages/std"],    # 优势值标准差
            "grpo_step": step,
        }
        # grpo_clip 模式下额外记录 clip fraction
        if args.loss_type == "grpo_clip":
            log_dict["train/clip_fraction"] = avg_epoch_clip_frac
        wandb.log(log_dict)

        # ===== 定期验证 =====
        if step % args.eval_every == 0 or step == 1 or step == args.n_grpo_steps:
            print("  验证中...")
            model.eval()
            load_policy_into_vllm_instance(model, llm)

            # 取前 args.n_eval_examples 条验证数据评估
            eval_prompts = val_prompts[:args.n_eval_examples]
            eval_gts = val_gts[:args.n_eval_examples]

            eval_metrics = evaluate_with_vllm(
                llm, eval_prompts, eval_gts, reward_fn, eval_sampling_params,
            )
            print(f"  验证: accuracy={eval_metrics['avg_answer_reward']:.4f}, "
                  f"format={eval_metrics['avg_format_reward']:.4f}")

            # 记录生成样例和 entropy
            gen_result = log_generations(
                policy_model=model, llm=llm, tokenizer=tokenizer,
                prompts=val_prompts, ground_truths=val_gts,
                reward_fn=reward_fn,
                sampling_params=eval_sampling_params,
                num_examples=args.num_log_examples,
            )

            wandb.log({
                "val/accuracy": eval_metrics["avg_answer_reward"],
                "val/format_reward": eval_metrics["avg_format_reward"],
                "val/avg_reward": eval_metrics["avg_reward"],
                "val/avg_entropy": gen_result["summary"]["avg_entropy"],
                "val/avg_response_length": gen_result["summary"]["avg_response_length"],
                "val/generations": gen_result["table"],
                "grpo_step": step,
            })

            model.train()

        # ===== 定期保存 checkpoint =====
        if step % args.save_every == 0 or step == args.n_grpo_steps:
            save_dir = os.path.join(args.output_dir, f"grpo_{args.loss_type}_step{step}")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  模型已保存到 {save_dir}")

    # --- 训练结束 ---
    wandb.finish()
    print("\nGRPO 训练完成！")


# ======================== 参数解析 ========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO 训练脚本")

    # 模型和数据
    parser.add_argument("--model_id", type=str,
                        default="/home/fredkeira/Data/models/Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data", type=str,
                        default="data/gsm8k/sft_train.jsonl",
                        help="训练数据，使用 sft_train.jsonl（prompt 已套好 r1_zero 模板）")
    parser.add_argument("--val_data", type=str,
                        default="data/gsm8k/sft_val.jsonl",
                        help="验证数据，使用 sft_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    # GRPO 核心超参
    parser.add_argument("--n_grpo_steps", type=int, default=200,
                        help="GRPO 外层循环次数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--advantage_eps", type=float, default=1e-6,
                        help="组内归一化时防止除零的 epsilon")
    parser.add_argument("--rollout_batch_size", type=int, default=256,
                        help="每步 rollout 的总响应数 = n_prompts × group_size")
    parser.add_argument("--group_size", type=int, default=8,
                        help="每个 prompt 的响应数（组大小）")
    parser.add_argument("--sampling_temperature", type=float, default=1.0,
                        help="rollout 采样温度")
    parser.add_argument("--sampling_min_tokens", type=int, default=4,
                        help="最少生成 token 数，防止空响应")
    parser.add_argument("--sampling_max_tokens", type=int, default=1024,
                        help="最多生成 token 数")
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1,
                        help="每个 rollout batch 训练几轮（1=on-policy）")
    parser.add_argument("--train_batch_size", type=int, default=256,
                        help="训练 batch 大小（on-policy 时 = rollout_batch_size）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=256,
                        help="梯度累积步数（microbatch = train_batch_size / 此值）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75,
                        help="vLLM GPU 显存占用比例")

    # 损失类型
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
                        help="策略梯度损失类型")
    parser.add_argument("--use_std_normalization", action="store_true", default=True,
                        help="组内归一化时是否除以标准差")
    parser.add_argument("--cliprange", type=float, default=0.2,
                        help="GRPO-Clip 的 clip 参数 ε")

    # 日志和评估
    parser.add_argument("--eval_every", type=int, default=5,
                        help="每隔多少步做一次验证")
    parser.add_argument("--save_every", type=int, default=50,
                        help="每隔多少步保存 checkpoint")
    parser.add_argument("--n_eval_examples", type=int, default=1024,
                        help="验证时使用多少条样本（至少 1024 以减少噪声）")
    parser.add_argument("--num_log_examples", type=int, default=10,
                        help="记录多少条生成样例到 wandb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="gsm8k-grpo")

    # 消融实验参数
    parser.add_argument("--length_norm", type=str, default="masked_mean",
                        choices=["masked_mean", "masked_normalize"],
                        help="序列长度聚合方式：masked_mean（除以response token数）或 masked_normalize（除以最大序列长度）")
    parser.add_argument("--prompt_type", type=str, default="r1_zero",
                        choices=["r1_zero", "question_only"],
                        help="prompt 模板 + reward 函数的联合选择。"
                             "r1_zero：使用预格式化的 sft jsonl + r1_zero_reward_fn（要求 </think><answer> 格式）；"
                             "question_only：从原始 jsonl 加载 + question_only_reward_fn（只检查 \\boxed{} 格式）")

    args = parser.parse_args()
    train(args)



'''
 GRPO 参数之间的制衡关系                                           
                                                                    
  核心等式链                                                        
                  
  rollout_batch_size = n_prompts × group_size                       
  train_batch_size = micro_batch_size × gradient_accumulation_steps
  n_microbatches_per_rollout = rollout_batch_size / micro_batch_size

  1. 显存相关（硬约束）

  micro_batch_size = train_batch_size / gradient_accumulation_steps

  ┌────────────────────────┬────────────────────────────────────┐
  │          参数          │                影响                │
  ├────────────────────────┼────────────────────────────────────┤
  │ micro_batch_size       │ 直接决定单次前向/反向传播的显存占  │
  │                        │ 用。3090 跑 1.5B 模型，micro=1~2   │
  ├────────────────────────┼────────────────────────────────────┤
  │ gradient_accumulation_ │ 越大 → micro 越小 → 显存越省，但训 │
  │ steps                  │ 练越慢（更多次前向传播）           │
  ├────────────────────────┼────────────────────────────────────┤
  │ gpu_memory_utilization │ vLLM 端的显存比例。太高会          │
  │                        │ OOM，太低浪费 KV cache 空间        │
  ├────────────────────────┼────────────────────────────────────┤
  │ sampling_max_tokens    │ 越长 → vLLM KV cache 占用越大 +    │
  │                        │ 训练时 seq_len 越长占显存越多      │
  └────────────────────────┴────────────────────────────────────┘

  制衡：想要更大 train_batch_size，要么增大
  gradient_accumulation_steps（更慢），要么减小
  sampling_max_tokens（回答更短）。

  2. 统计质量 vs 速度

  rollout_batch_size = n_prompts × group_size

  参数: group_size
  增大的好处: 组内 advantage 估计更准（方差更低）
  增大的代价: rollout 时间线性增长，同样的 rollout_batch_size
    下能看的 prompt 更少
  ────────────────────────────────────────
  参数: n_prompts (= rollout_batch_size/group_size)
  增大的好处: 覆盖更多题目，梯度估计更稳定
  增大的代价: rollout 时间增长
  ────────────────────────────────────────
  参数: rollout_batch_size
  增大的好处: 同时增加 prompt 数和统计质量
  增大的代价: rollout 时间和训练时间都增长

  制衡：固定 rollout_batch_size=128 时，group_size=8 意味着 16 个
  prompt，group_size=16 意味着只有 8 个 prompt。group_size
  太大→题目多样性不够；太小→advantage 估计噪声大。

  3. On-policy vs Off-policy

  epochs_per_rollout_batch = 1  → on-policy（每批 rollout 只训一步）
  epochs_per_rollout_batch > 1  → off-policy（同一批 rollout
  训多步，需要 grpo_clip）

  ┌──────────────────┬────────────────────┬────────────────────┐
  │       参数       │   on-policy (=1)   │  off-policy (>1)   │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ epochs_per_rollo │ 数据利用率低，但策 │ 数据利用率高，但策 │
  │ ut_batch         │ 略梯度无偏         │ 略偏移需要 clip    │
  │                  │                    │ 修正               │
  ├──────────────────┼────────────────────┼────────────────────┤
  │                  │ no_baseline 或     │                    │
  │ loss_type        │ reinforce_with_bas │ 必须用 grpo_clip   │
  │                  │ eline              │                    │
  ├──────────────────┼────────────────────┼────────────────────┤
  │                  │                    │ 太小→更新太保守；  │
  │ cliprange        │ 不需要             │ 太大→等于没        │
  │                  │                    │ clip，策略可能崩   │
  └──────────────────┴────────────────────┴────────────────────┘

  制衡：rollout 很贵（需要 vLLM 推理），所以 off-policy 多训几步能省
   rollout 时间，但需要 clip 来保证稳定性。

  4. 学习动态

  ┌───────────────┬───────────────────────┬────────────────────┐
  │     参数      │         作用          │        制衡        │
  ├───────────────┼───────────────────────┼────────────────────┤
  │ learning_rate │ 越大→学得越快，但可能 │ 需要配合 gradient  │
  │               │ 不稳定                │ clipping (1.0)     │
  ├───────────────┼───────────────────────┼────────────────────┤
  │               │                       │ 但受限于           │
  │ train_batch_s │ 越大→梯度估计越准，可 │ rollout_batch_size │
  │ ize           │ 以用更大 lr           │ （on-policy        │
  │               │                       │ 时两者相等）       │
  ├───────────────┼───────────────────────┼────────────────────┤
  │ use_std_norma │ True→advantage        │ False→只减均值，梯 │
  │ lization      │ 被标准化到均值0方差1  │ 度尺度随 reward    │
  │               │ ，梯度尺度稳定        │ 分布变化           │
  ├───────────────┼───────────────────────┼────────────────────┤
  │               │ 防止 std=0            │ 太大会压缩         │
  │ advantage_eps │ 时除零（组内所有      │ advantage          │
  │               │ reward 相同）         │                    │
  ├───────────────┼───────────────────────┼────────────────────┤
  │ sampling_temp │ 高→rollout 多样性好， │ 太高→生成质量差，r │
  │ erature       │ 组内有区分度          │ eward 信号弱       │
  └───────────────┴───────────────────────┴────────────────────┘

  5. 一图总结关系

  sampling_temperature ──→ rollout 多样性 ──→ advantage 区分度
                                                ↓
  n_prompts ←── rollout_batch_size/group_size   advantage 质量
      ↓              ↓                              ↓
  题目覆盖       group_size ──→ 组内统计质量      梯度信号质量
                                                     ↓
  micro_batch_size ←── train_batch_size/grad_accum  策略更新
      ↓                                              ↓
    显存占用      epochs_per_rollout_batch ──→ 数据利用 vs 偏移
                          ↓
                     cliprange ──→ 更新幅度限制

  你当前的 3090 配置解读

  rollout_batch_size=128, group_size=8 
  → 16 个 prompt，每个 8 个response
  
  train_batch_size=128, grad_accum=128 → micro=1（最省显存）
  epochs=1 → on-policy，不需要 clip

  如果觉得训练太慢，可以尝试 micro=2（gradient_accumulation_steps=64
  ），前向传播次数减半，但显存翻倍——3090 上 1.5B 模型 micro=2
  大概率也能跑
'''

'''

on-policy
group_size: 8
无 clip（reinforce_with_baseline 不用重要性权重）
std_normalization 

CUDA_VISIBLE_DEVICES=0,1 uv run python grpo_scripts/train_grpo.py \--gradient_accumulation_steps 256 \--gpu_memory_utilization 0.75
wandb:                grpo_step ▁▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇███
wandb:     train/advantage_mean ▆▅█▆▆▃▂▄▄▅▄▄▄▄▁▅▅▃▄▄▄▄▄▆▄▄▄▃▅▄▄▄▄▄▄▄▄▄▅▄
wandb:      train/advantage_std ▆███▇▇▄▇▇▇▇▅▅▆▅▅▄▅▅▄▆▅▄▅▄▄▄▅▅▁▄▂▃▅▂▃▂▃▃▂
wandb: train/answer_reward_mean ▁▁▂▃▄▅▆▆▆▆▇▇▇▆▇▆▇▇▇▇▇▆▇▇▇█▇█▇▇▇█▇▇▇▇▇██▇
wandb: train/format_reward_mean ▁▂▄▅▆▇▇▇▆▇▇▇▇▇█▇▇▇▇▇█████▇█▇████████████
wandb:          train/grad_norm ▁▂▂▂▂▃▂▃▃▃▃▄▄▃▆▅▃▄▃▄▄▅▅▅▄▆█▅▇▇▆▄▄▅▆▇▅▆▄▅
wandb:               train/loss ▇▇▆▆▄▅▅▇▄▄▇▆█▆▄▆▆▆▇▄▄▃▆▅▇▄▆▄▆▇▄▄▁▅▃▆▄▄▄▄
wandb:         train/reward_max ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:        train/reward_mean ▁▁▂▃▄▅▆▆▆▆▆▇▇▆▇▆▇▇▇▆▇█▇▇▇▇▇▇▇▆▇█▇▇█▇██▇▇
wandb:         train/reward_min ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         train/reward_std ▆██▇▇█▇██▇▅▆▆▆▆▆▇▇▅▅▃▃▅▇▆▅▃▄▄▄▁▄▅▅▅▅▄▃▄▅
wandb:      train/token_entropy █▄▃▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▂▂▂▁▂▂▁▁▂▂
wandb:             val/accuracy ▁▂▃▄▅▅▅▆▆▆▇▇▇▇▇▆▇▇▇▇▇▇▇▇█▇▇██▇█▇▇▇▇████▇
wandb:          val/avg_entropy █▇▄▄▅▂▂▂▂▁▃▁▁▁▁▂▁▁▃▁▁▁▁▂▁▂▃▄▄▂▃▁▄▂▃▃▃▂▂▂
wandb:  val/avg_response_length ▇▆▂▃▂▂▂▄▄▂█▂▁▂▁▆▂▂▆▂▃▁▂▂▃▂▂▂▂▃▃▃▁▂▃▂▁▃▁▂
wandb:           val/avg_reward ▁▂▃▄▅▅▅▆▆▆▇▇▇▇▇▆▇▇▇▇▇▇▇▇█▇▇██▇█▇▇▇▇████▇
wandb:        val/format_reward ▁▃▄▅▅▆▆▆▇▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇██▇██▇████▇███
wandb: 
wandb: Run summary:
wandb:                grpo_step 200
wandb:     train/advantage_mean -0.0
wandb:      train/advantage_std 0.61993
wandb: train/answer_reward_mean 0.77344
wandb: train/format_reward_mean 0.98047
wandb:          train/grad_norm 10.8125
wandb:               train/loss -0.00055
wandb:         train/reward_max 1
wandb:        train/reward_mean 0.77344
wandb:         train/reward_min 0
wandb:         train/reward_std 0.41943
wandb:      train/token_entropy 0.13225
wandb:             val/accuracy 0.43262
wandb:          val/avg_entropy 0.22302
wandb:  val/avg_response_length 108.9
wandb:           val/avg_reward 0.43262
wandb:        val/format_reward 0.81836

'''