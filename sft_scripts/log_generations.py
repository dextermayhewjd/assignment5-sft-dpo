import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Callable
from vllm import LLM, SamplingParams

from sft_scripts.get_response_log_probs import get_response_log_probs


def log_generations(
    policy_model: PreTrainedModel,
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: SamplingParams,
    num_examples: int = 10,
) -> dict:
    """在训练过程中对验证集 prompt 做生成，记录各种统计信息用于 logging。

    用 vLLM (GPU 1) 批量生成回答，用 policy_model (GPU 0) 计算 entropy。
    这样生成和训练互不抢显存。

    Args:
        policy_model: 训练中的 policy 模型（用于计算 entropy，在 GPU 0 上）。
        llm: vLLM 实例（用于快速批量生成，在 GPU 1 上）。
        tokenizer: tokenizer。
        prompts: 验证集 prompt 列表。
        ground_truths: 对应的标准答案列表。
        reward_fn: 打分函数，(response, ground_truth) -> dict。
        sampling_params: vLLM 生成参数。
        num_examples: 抽取多少条做 logging（避免每次评估太慢）。

    Returns:
        dict:
            "examples": 每条的详细信息
            "summary": 汇总统计
            "table": wandb.Table 对象（可直接 log 到 wandb）
    """
    import wandb

    # 从验证集中均匀抽样 num_examples 条
    step = max(1, len(prompts) // num_examples)
    sample_indices = list(range(0, len(prompts), step))[:num_examples]
    sample_prompts = [prompts[i] for i in sample_indices]
    sample_gts = [ground_truths[i] for i in sample_indices]

    # ---- 第一步：用 vLLM 批量生成 ----
    # vLLM 在 GPU 1 上，批量生成比逐条快得多
    outputs = llm.generate(sample_prompts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]

    examples = []
    all_rewards = []
    all_entropies = []
    all_lengths = []
    correct_lengths = []
    incorrect_lengths = []

    policy_device = next(policy_model.parameters()).device

    for prompt, gen_text, gt in zip(sample_prompts, generated_texts, sample_gts):
        # ---- 第二步：计算 reward ----
        reward_info = reward_fn(gen_text, gt)

        # ---- 第三步：计算 response 部分的平均 token entropy ----
        # 把完整序列（prompt + response）编码后送入 policy_model
        full_text = prompt + gen_text
        full_ids = tokenizer.encode(full_text, return_tensors="pt").to(policy_device)
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = prompt_ids.shape[1]
        response_length = full_ids.shape[1] - prompt_len

        avg_entropy = 0.0
        if response_length > 0:
            with torch.no_grad():
                labels = torch.cat([
                    full_ids[:, 1:],
                    torch.zeros(1, 1, dtype=torch.long, device=policy_device)
                ], dim=1)
                result = get_response_log_probs(
                    model=policy_model,
                    input_ids=full_ids,
                    labels=labels,
                    return_token_entropy=True,
                )
            # 只取 response 部分的 entropy
            response_entropy = result["token_entropy"][:, prompt_len - 1: prompt_len - 1 + response_length]
            avg_entropy = response_entropy.mean().item()

        # ---- 第四步：收集统计 ----
        all_rewards.append(reward_info["reward"])
        all_entropies.append(avg_entropy)
        all_lengths.append(response_length)

        is_correct = reward_info["answer_reward"] > 0.5
        if is_correct:
            correct_lengths.append(response_length)
        else:
            incorrect_lengths.append(response_length)

        examples.append({
            "prompt": prompt[-100:],  # 截取末尾，避免 wandb table 太宽
            "response": gen_text[:500],  # 截取开头
            "ground_truth": gt,
            "reward": reward_info["reward"],
            "format_reward": reward_info["format_reward"],
            "answer_reward": reward_info["answer_reward"],
            "avg_token_entropy": round(avg_entropy, 4),
            "response_length": response_length,
        })

    # ---- 第五步：汇总统计 ----
    summary = {
        "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "avg_entropy": sum(all_entropies) / len(all_entropies) if all_entropies else 0.0,
        "avg_response_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0.0,
        "avg_correct_response_length": sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0.0,
        "avg_incorrect_response_length": sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0.0,
        "num_correct": len(correct_lengths),
        "num_incorrect": len(incorrect_lengths),
    }

    # 构造 wandb Table，方便在 dashboard 上直接查看生成样例
    table = wandb.Table(
        columns=["prompt", "response", "ground_truth", "reward", "format_reward",
                 "answer_reward", "avg_entropy", "response_length"],
        data=[[
            ex["prompt"], ex["response"], ex["ground_truth"],
            ex["reward"], ex["format_reward"], ex["answer_reward"],
            ex["avg_token_entropy"], ex["response_length"],
        ] for ex in examples],
    )

    return {
        "examples": examples,
        "summary": summary,
        "table": table,
    }
