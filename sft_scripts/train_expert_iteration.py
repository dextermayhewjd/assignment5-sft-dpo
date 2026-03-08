"""
Expert Iteration (EI) 训练脚本 (Algorithm 2)

核心思路（对比 SFT）：
    SFT：用固定的人工标注数据训练。数据从头到尾不变。
    EI ：让模型自己做题 → 只留答对的 → 用答对的做 SFT → 循环。
         数据是模型自己生成的，且每一轮都在变。

算法伪代码：
    for ei_step in 1..n_ei_steps:
        1. 从题库中采样一批 question
        2. 用当前模型对每个 question 生成 G 个回答（rollout）
        3. 用 reward_fn 检查哪些回答是对的
        4. 过滤，只保留答对的 (prompt, response) 对
        5. 在这些正确数据上做 SFT（和之前的 SFT 完全一样的流程）

用法:
    CUDA_VISIBLE_DEVICES=0,1 uv run python sft_scripts/train_expert_iteration.py \
        --batch_size_db 512 --n_ei_steps 5 --G 4 --sft_epochs 1
"""

import argparse
import json
import os
import sys
import random
from unittest.mock import patch

# 把项目根目录加到 sys.path，让 sft_scripts 和 cs336_alignment 能被 import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
# 以下四个是之前 SFT 实现的模块，EI 的 SFT 部分完全复用
from sft_scripts.tokenize_prompt_and_output import tokenize_prompt_and_output
from sft_scripts.get_response_log_probs import get_response_log_probs
from sft_scripts.sft_microbatch_train_step import sft_microbatch_train_step
from sft_scripts.log_generations import log_generations


# ======================== vLLM 工具函数（和 SFT 脚本完全一样）========================

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.70):
    """在指定 GPU 上启动 vLLM 推理引擎（和 SFT 脚本一样）。"""
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
    """把训练中的 policy 权重同步到 vLLM（和 SFT 脚本一样）。"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_with_vllm(llm, prompts, ground_truths, reward_fn, sampling_params):
    """全量验证集评估（和 SFT 脚本一样）。"""
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

def load_questions(path: str, prompt_template: str):
    """加载原始 GSM8K 数据。

    【对比 SFT】SFT 加载的是预先构造好的 (prompt, response) 对；
    EI 只需要 question 和 ground_truth，response 由模型自己生成。
    """
    import re
    prompts, ground_truths, questions = [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            question = d["question"]
            # 清理 GSM8K 的计算标记 <<48/2=24>>，提取 #### 后的最终答案
            gt = re.sub(r"<<.*?>>", "", d["answer"]).split("####")[-1].strip()
            # 套入 r1_zero prompt 模板
            prompt = prompt_template.format(question=question)
            prompts.append(prompt)
            ground_truths.append(gt)
            questions.append(question)
    return prompts, ground_truths, questions

# 注意：这里只有 prompt（问题）和 ground_truth（标准答案的数字），
# 没有 response——response 要靠模型自己生成，这是和 SFT 最大的不同。


# ======================== EI 独有：rollout + filter ========================
# 这个函数是 EI 和 SFT 的核心区别所在。SFT 没有这一步。

def rollout_and_filter(
    llm,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn,
    sampling_params: SamplingParams,
) -> tuple[list[str], list[str], dict]:
    """Algorithm 2 第 5-7 行：对每个 question 生成 G 个回答，只保留答对的。

    【对比 SFT】SFT 直接从文件读 (prompt, response)，不需要这一步。
    EI 的训练数据来自这里：模型自己生成 → 过滤正确的 → 作为 SFT 数据。
    """

    # 用 vLLM 批量生成。SamplingParams 里的 n=G 表示每个 prompt 采样 G 个不同回答
    # 比如 512 个 question × G=4 = 2048 个回答
    outputs = llm.generate(prompts, sampling_params)

    filtered_prompts = []    # 存答对的 prompt
    filtered_responses = []  # 存答对的 response
    total_rollouts = 0       # 总生成数
    correct_rollouts = 0     # 答对的数

    # 遍历每个 question 的所有 G 个候选回答
    for prompt, gt, output in zip(prompts, ground_truths, outputs):
        for candidate in output.outputs:  # output.outputs 是长度为 G 的列表
            response_text = candidate.text  # 模型生成的回答文本
            total_rollouts += 1

            # 用 reward_fn 判断这个回答对不对
            reward_info = reward_fn(response_text, gt)

            # 只保留答对的（answer_reward == 1.0）
            # 答错的直接丢弃——这就是 "filter" 的含义
            if reward_info["answer_reward"] == 1.0:
                correct_rollouts += 1
                filtered_prompts.append(prompt)     # 同一个 question 可能有多条正确回答
                filtered_responses.append(response_text)

    # pass_rate = 答对比例，随着模型变强会升高（自举效应）
    pass_rate = correct_rollouts / total_rollouts if total_rollouts > 0 else 0.0
    metadata = {
        "total_rollouts": total_rollouts,
        "correct_rollouts": correct_rollouts,
        "pass_rate": pass_rate,                      # 关键指标：模型当前能力的体现
        "filtered_dataset_size": len(filtered_prompts),  # 本步 SFT 的训练数据量
    }
    return filtered_prompts, filtered_responses, metadata


# ======================== SFT 部分（和 SFT 脚本的训练循环完全一样）========================

def sft_on_filtered(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    optimizer,
    policy_device: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    seed: int,
    global_step: int,
) -> tuple[int, list[float]]:
    """Algorithm 2 第 8 行：在过滤后的正确样本上做 SFT。

    【对比 SFT】这个函数内部的逻辑和 train_sft.py 的训练循环一模一样：
    tokenize → get_log_probs → microbatch_train_step → clip_grad → optimizer.step
    唯一区别是输入数据来自 rollout_and_filter()，而不是固定文件。
    """
    losses = []
    for epoch in range(num_epochs):
        # 打乱数据顺序
        indices = list(range(len(prompts)))
        random.seed(seed + epoch)
        random.shuffle(indices)

        # effective_batch_size = 每次 optimizer.step() 看到的总样本数
        effective_batch_size = batch_size * gradient_accumulation_steps

        for batch_start in range(0, len(indices), effective_batch_size):
            batch_indices = indices[batch_start:batch_start + effective_batch_size]
            if len(batch_indices) == 0:
                break

            # 最后一个 batch 可能不满，动态计算实际 accumulation 步数
            actual_accum = (len(batch_indices) + batch_size - 1) // batch_size

            optimizer.zero_grad()  # 清空梯度
            total_loss = 0.0
            micro_count = 0

            # ---- 以下和 SFT 完全一样 ----
            for micro_start in range(0, len(batch_indices), batch_size):
                micro_idx = batch_indices[micro_start:micro_start + batch_size]
                micro_prompts = [prompts[i] for i in micro_idx]
                micro_responses = [responses[i] for i in micro_idx]

                # tokenize：和 SFT 一样，把 (prompt, response) 变成 (input_ids, labels, mask)
                batch = tokenize_prompt_and_output(micro_prompts, micro_responses, tokenizer)
                input_ids = batch["input_ids"].to(policy_device)
                labels = batch["labels"].to(policy_device)
                response_mask = batch["response_mask"].to(policy_device)

                # 前向传播：和 SFT 一样，获取每个 token 的 log 概率
                result = get_response_log_probs(
                    model=model, input_ids=input_ids, labels=labels,
                    return_token_entropy=False,
                )

                # 计算 loss + backward：和 SFT 一样
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=result["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=actual_accum,
                )
                total_loss += loss.item()
                micro_count += 1

            # 梯度裁剪 + 参数更新：和 SFT 一样
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            losses.append(total_loss / micro_count)

    return global_step, losses


# ======================== 主循环 ========================

def train(args):
    # --- 设备分配（和 SFT 一样）---
    policy_device = "cuda:0"   # GPU 0：训练 policy 模型
    vllm_device = "cuda:1"     # GPU 1：vLLM 推理（rollout + 评估）
    torch.cuda.set_device(policy_device)  # 确保默认 CUDA 设备是 GPU 0

    # --- 加载 prompt 模板 ---
    prompt_template_path = os.path.join(
        os.path.dirname(__file__), "..", "cs336_alignment", "prompts", "r1_zero.prompt"
    )
    with open(prompt_template_path) as f:
        prompt_template = f.read()

    # --- 加载数据 ---
    # 【对比 SFT】SFT 加载的是 sft_train.jsonl（已有 prompt + response）
    # EI 加载的是原始 train.jsonl（只有 question + answer），response 要模型自己生成
    train_prompts, train_gts, train_questions = load_questions(args.train_data, prompt_template)
    val_prompts, val_gts, _ = load_questions(args.val_data, prompt_template)
    print(f"Train questions: {len(train_prompts)}, Val questions: {len(val_prompts)}")

    # --- 加载模型和 tokenizer（和 SFT 一样）---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=policy_device,
    )

    # --- 初始化 vLLM（和 SFT 一样）---
    print("Initializing vLLM on", vllm_device)
    llm = init_vllm(args.model_id, vllm_device, seed=args.seed)

    # --- 采样参数 ---
    # 【EI 独有】rollout 采样参数：
    #   n=G 表示每个 prompt 生成 G 个不同回答（SFT 不需要这个）
    #   min_tokens=4 避免生成空字符串导致下游 NaN
    #   temperature 控制多样性，太低则 G 个回答太相似，太高则质量差
    rollout_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,  # 0.8，比评估时 (1.0) 稍低
        max_tokens=args.sampling_max_tokens,
        min_tokens=4,       # 至少生成 4 个 token，防止空输出
        n=args.G,           # 每个 prompt 生成 G 个候选回答
        stop=["</answer>"],                 # 遇到 </answer> 停止
        include_stop_str_in_output=True,    # 保留 </answer> 在输出中
        seed=args.seed,
    )
    # 评估用（和 SFT 一样，n=1 只生成一个回答）
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        min_tokens=4, stop=["</answer>"], include_stop_str_in_output=True,
    )

    # --- 初始化 wandb ---
    wandb.init(
        project=args.wandb_project,
        name=f"ei_db{args.batch_size_db}_G{args.G}_ep{args.sft_epochs}",
        config=vars(args),
    )
    # 【对比 SFT】SFT 用 train_step 和 eval_step 两个 x 轴
    # EI 多了一个 ei_step 作为外层循环的 x 轴
    wandb.define_metric("train_step")
    wandb.define_metric("ei_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("ei/*", step_metric="ei_step")

    # 优化器（和 SFT 一样）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    global_step = 0  # SFT 更新的总步数（跨所有 EI step 累积）

    # ======================== EI 主循环（SFT 没有这个外层循环）========================
    # 【对比 SFT】SFT 就是：加载固定数据 → 训练 num_epochs 轮 → 结束
    # EI 则是：循环 n_ei_steps 次，每次用当前模型生成新数据再训练
    for ei_step in range(1, args.n_ei_steps + 1):
        print(f"\n{'='*50}")
        print(f"EI Step {ei_step}/{args.n_ei_steps}")
        print(f"{'='*50}")

        # --- Algorithm 2 第 3 行：从题库中随机采样 batch_size_db 个 question ---
        # 【对比 SFT】SFT 用全部训练数据；EI 每步只抽一个子集
        random.seed(args.seed + ei_step)
        db_indices = random.sample(
            range(len(train_prompts)),
            min(args.batch_size_db, len(train_prompts))  # 比如采 512 个 question
        )
        db_prompts = [train_prompts[i] for i in db_indices]  # 这一步要用的 question
        db_gts = [train_gts[i] for i in db_indices]          # 对应的标准答案

        # --- Algorithm 2 第 4 行：把当前 policy 权重同步到 vLLM ---
        # 【对比 SFT】SFT 只在 eval 时同步；EI 每步 rollout 前都要同步
        # 因为 EI 需要用"当前"模型生成，而模型每步都在更新
        model.eval()
        load_policy_into_vllm_instance(model, llm)

        # --- Algorithm 2 第 5-7 行：rollout G 次 + filter ---
        # 【SFT 没有这一步】这是 EI 独有的：让模型自己做题，只留答对的
        print(f"Rolling out {len(db_prompts)} questions x G={args.G}...")
        filtered_prompts, filtered_responses, rollout_meta = rollout_and_filter(
            llm, db_prompts, db_gts, r1_zero_reward_fn, rollout_sampling_params,
        )
        print(f"  pass_rate={rollout_meta['pass_rate']:.3f}, "
              f"filtered={rollout_meta['filtered_dataset_size']} / {rollout_meta['total_rollouts']}")
        # pass_rate 体现模型当前能力：
        #   初期模型弱 → pass_rate 低（比如 5%）→ 过滤后数据少
        #   后期模型强 → pass_rate 高（比如 30%）→ 更多数据，且质量更好

        # --- 评估当前 policy ---
        print("Evaluating...")
        eval_metrics = evaluate_with_vllm(
            llm, val_prompts, val_gts, r1_zero_reward_fn, eval_sampling_params,
        )
        # log_generations：抽样几条看看模型生成了什么 + 计算 entropy
        gen_result = log_generations(
            policy_model=model, llm=llm, tokenizer=tokenizer,
            prompts=val_prompts, ground_truths=val_gts,
            reward_fn=r1_zero_reward_fn,
            sampling_params=eval_sampling_params,
            num_examples=args.num_log_examples,
        )
        # 记录到 wandb
        wandb.log({
            "ei/accuracy": eval_metrics["avg_answer_reward"],     # 验证准确率
            "ei/format_reward": eval_metrics["avg_format_reward"], # 格式合规率
            "ei/pass_rate": rollout_meta["pass_rate"],            # EI 独有：rollout 通过率
            "ei/filtered_size": rollout_meta["filtered_dataset_size"],  # EI 独有：过滤后数据量
            "ei/avg_entropy": gen_result["summary"]["avg_entropy"],     # 模型不确定性
            "ei/avg_response_length": gen_result["summary"]["avg_response_length"],
            "ei/avg_correct_length": gen_result["summary"]["avg_correct_response_length"],
            "ei/avg_incorrect_length": gen_result["summary"]["avg_incorrect_response_length"],
            "ei/generations": gen_result["table"],   # wandb 表格，可直接看生成样例
            "ei_step": ei_step,
        })
        print(f"  accuracy={eval_metrics['avg_answer_reward']:.4f}, "
              f"entropy={gen_result['summary']['avg_entropy']:.4f}")

        # 如果模型太弱，一个都没答对，没数据可训，跳过
        if len(filtered_prompts) == 0:
            print("  No correct rollouts, skipping SFT step.")
            continue

        # --- Algorithm 2 第 8 行：在 filtered 数据上做 SFT ---
        # 【对比 SFT】SFT 在固定数据上训 num_epochs 轮
        # EI 在每步新生成的正确数据上训 sft_epochs 轮（通常 1 轮就够）
        model.train()
        print(f"SFT on {len(filtered_prompts)} correct examples for {args.sft_epochs} epoch(s)...")
        global_step, step_losses = sft_on_filtered(
            model=model,
            tokenizer=tokenizer,
            prompts=filtered_prompts,     # 来自 rollout_and_filter，不是固定文件
            responses=filtered_responses,  # 模型自己生成的正确回答
            optimizer=optimizer,
            policy_device=policy_device,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.sft_epochs,
            seed=args.seed + ei_step * 100,
            global_step=global_step,
        )
        avg_loss = sum(step_losses) / len(step_losses) if step_losses else 0.0
        wandb.log({
            "train/loss": avg_loss,
            "train/ei_step": ei_step,
            "train_step": global_step,
        })
        print(f"  SFT done, avg_loss={avg_loss:.4f}, global_step={global_step}")

    # --- 最终评估（和 SFT 一样）---
    print("\nFinal evaluation...")
    model.eval()
    load_policy_into_vllm_instance(model, llm)
    eval_metrics = evaluate_with_vllm(
        llm, val_prompts, val_gts, r1_zero_reward_fn, eval_sampling_params,
    )
    gen_result = log_generations(
        policy_model=model, llm=llm, tokenizer=tokenizer,
        prompts=val_prompts, ground_truths=val_gts,
        reward_fn=r1_zero_reward_fn,
        sampling_params=eval_sampling_params,
        num_examples=args.num_log_examples,
    )
    wandb.log({
        "ei/accuracy": eval_metrics["avg_answer_reward"],
        "ei/format_reward": eval_metrics["avg_format_reward"],
        "ei/avg_entropy": gen_result["summary"]["avg_entropy"],
        "ei/avg_response_length": gen_result["summary"]["avg_response_length"],
        "ei/generations": gen_result["table"],
        "ei_step": args.n_ei_steps + 1,
    })
    print(f"Final: accuracy={eval_metrics['avg_answer_reward']:.4f}, "
          f"entropy={gen_result['summary']['avg_entropy']:.4f}")

    # --- 保存模型（和 SFT 一样）---
    save_dir = os.path.join(args.output_dir, f"ei_db{args.batch_size_db}_G{args.G}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/home/fredkeira/Data/models/Qwen/Qwen2.5-Math-1.5B")
    # 【对比 SFT】SFT 用 sft_train.jsonl（已有 response）；EI 用原始 train.jsonl（只有 question）
    parser.add_argument("--train_data", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    # === EI 独有的超参（SFT 没有这些）===
    parser.add_argument("--n_ei_steps", type=int, default=5,
                        help="EI 外层循环次数。每步：rollout → filter → SFT")
    parser.add_argument("--batch_size_db", type=int, default=512,
                        help="每步从题库中采样多少个 question（作业要求试 512/1024/2048）")
    parser.add_argument("--G", type=int, default=4,
                        help="每个 question 生成多少个 rollout（作业要求试不同值）")
    parser.add_argument("--sft_epochs", type=int, default=1,
                        help="每步 EI 内部做几轮 SFT（作业要求试不同值）")
    parser.add_argument("--sampling_temperature", type=float, default=0.8,
                        help="rollout 温度。太低→回答太相似，太高→质量差")
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)

    # === SFT 超参（和 SFT 脚本一样）===
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2, help="SFT microbatch size（24GB GPU 用 2）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # === logging ===
    parser.add_argument("--num_log_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="gsm8k-ei")
    args = parser.parse_args()
    train(args)


"""
========================================
EI vs SFT 总结
========================================

代码层面的区别只有两个地方：

1. 数据来源：
   SFT：load_sft_data("sft_train.jsonl") → 固定的 (prompt, response) 对
   EI ：rollout_and_filter()             → 每步用当前模型生成，只留答对的

2. 外层循环：
   SFT：一个 for epoch 循环，数据不变
   EI ：for ei_step 循环，每步数据都是新生成的

内层 SFT 部分（tokenize → log_probs → loss → backward → clip → step）完全一样。

有种变相强化学习的样子 只不过是通过reward 来选择筛选条件                         
  但这种方法是否会造成overfitting呢?                                               
                                                                                   
● 对，你的直觉很准——EI 本质上就是一种简化版的强化学习。                            
                                                                                   
  为什么像强化学习                                                                 
                                                                                   
  - RL：生成 → 用 reward 打分 → 根据 reward 更新梯度（直接优化）                   
  - EI：生成 → 用 reward 过滤 → 只在答对的样本上做 SFT（间接优化）                 
                                                                                   
  区别只是 EI 用"过滤"代替了梯度中的 reward 加权。                                 

  会 overfit 吗？

  会，但有几个机制在缓解它：

  会 overfit 的方向：
  - 模型反复在自己生成的"正确格式"上训练，可能会对特定的推理风格或措辞过度拟合
  - 如果某类题目每次都答对，模型会对这类题的解法记得很熟，但泛化性下降

  缓解的机制：
  1. 每步数据都是新生成的 — 不像 SFT 反复看同一批数据，EI 每步的 response
  都不同，天然有多样性
  2. filter 本身是正则 — 只有高质量（答对的）样本才能进入训练，噪声样本被丢弃
  3. pass_rate 越高风险越大 — 模型早期 pass_rate 低（数据少），训得慢；后期
  pass_rate 高时模型已经比较强了，overfit 的危害相对小

  真正的风险在于：
  - 如果 batch_size_db 太小，每步题目重复率高，模型会对这几百道题过拟合
  - 如果 sft_epochs 太大，在同一批 rollout 上训太多轮，相当于 SFT 的 overfit
  - EI 无法产生超过模型当前能力的样本——如果某类题一直答不对，模型永远学不到这类题

  这也是为什么作业要求 batch_size_db 试 512/1024/2048，sft_epochs 也要试不同值——大
  batch 和少 epoch 是降低 overfit 风险的标准手段。
"""
