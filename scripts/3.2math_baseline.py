"""
qwen1.5b 数学模型
在/home/fredkeira/Data/models/Qwen/Qwen2.5-Math-1.5B

数学数据
在/home/fredkeira/Data/datasets/gsm8k

任务要求
Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. This script
should
(1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl,
(2) format them as string prompts to the language model using the r1_zero prompt, and
(3) generate outputs for each example.
(4) calculate evaluation metrics and
(5) serialize the examples, model generations, and corresponding evaluation scores to disk
for analysis in subsequent problems.


3.2 Zero-shot MATH baseline

评估 Qwen 2.5 Math 1.5B 在 GSM8K 测试集上的 zero-shot 性能。

步骤：
(1) 加载 GSM8K 测试集（用 main，不用 socratic）
    - main: 标准问答格式，适合做 zero-shot 评估
    - socratic: 苏格拉底式分步引导格式，不适合我们的目的
(2) 用 r1_zero prompt 模板格式化每个问题
(3) 用 vLLM 批量生成模型输出
(4) 用 r1_zero_reward_fn 计算评估指标
(5) 将结果序列化到磁盘
"""

import json
import os
from typing import Callable, List

import pandas as pd
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    questions: List[str],
    eval_sampling_params: SamplingParams,
) -> dict:
    """
    Evaluate a language model on a list of prompts,
    compute reward metrics, and return detailed results.
    """
    # (3) vLLM 批量生成
    print("Generating outputs...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]

    # (4) 计算评估指标
    print("Evaluating...")
    results = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    for question, gen_text, gt in zip(questions, generated_texts, ground_truths):
        reward_info = reward_fn(gen_text, gt)
        results.append(
            {
                "question": question,
                "ground_truth": gt,
                "generated_text": gen_text,
                "reward": reward_info["reward"],
                "format_reward": reward_info["format_reward"],
                "answer_reward": reward_info["answer_reward"],
            }
        )
        total_reward += reward_info["reward"]
        total_format_reward += reward_info["format_reward"]
        total_answer_reward += reward_info["answer_reward"]

    n = len(results)
    metrics = {
        "num_examples": n,
        "avg_reward": total_reward / n,
        "avg_format_reward": total_format_reward / n,
        "avg_answer_reward": total_answer_reward / n,
    }

    print(f"\n========== Results ==========")
    print(f"Total examples: {n}")
    print(f"Average reward (accuracy): {metrics['avg_reward']:.4f}")
    print(f"Average format reward:     {metrics['avg_format_reward']:.4f}")
    print(f"Average answer reward:     {metrics['avg_answer_reward']:.4f}")

    return {"results": results, "metrics": metrics}


if __name__ == "__main__":
    # ======================== 配置 ========================
    MODEL_PATH = "/home/fredkeira/Data/models/Qwen/Qwen2.5-Math-1.5B"
    DATA_PATH = "/home/fredkeira/Data/datasets/gsm8k/main/test-00000-of-00001.parquet"
    PROMPT_PATH = os.path.join(
        os.path.dirname(__file__), "..", "cs336_alignment", "prompts", "r1_zero.prompt"
    )
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "math_baseline")

    # ======================== (1) 加载数据 ========================
    print("Loading GSM8K test set...")
    df = pd.read_parquet(DATA_PATH)

    questions = df["question"].tolist()

    # 从 answer 列提取 ground truth（#### 后面的数字）
    ground_truths = []
    for ans in df["answer"]:
        # GSM8K answer 格式: 推理过程...\n#### <answer>
        gt = ans.split("####")[-1].strip()
        ground_truths.append(gt)

    print(f"Loaded {len(questions)} examples")

    # ======================== (2) 格式化 prompt ========================
    print("Formatting prompts with r1_zero template...")
    with open(PROMPT_PATH, "r") as f:
        prompt_template = f.read()

    prompts = [prompt_template.format(question=q) for q in questions]

    # 打印一个样例看看格式对不对
    print(f"\n--- Example prompt ---\n{prompts[0]}\n--- End example ---\n")

    # ======================== (3)+(4) 生成 + 评估 ========================
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,        # 双卡张量并行
        gpu_memory_utilization=0.8,    # 每张卡用 80% 显存
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    eval_output = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        questions=questions,
        eval_sampling_params=sampling_params,
    )

    # ======================== (5) 序列化到磁盘 ========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, "zero_shot_results.json")
    with open(output_path, "w") as f:
        json.dump(eval_output["results"], f, indent=2, ensure_ascii=False)

    metrics = eval_output["metrics"]
    metrics.update(
        {
            "model": MODEL_PATH,
            "dataset": DATA_PATH,
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 1024,
                "stop": ["</answer>"],
            },
        }
    )
    metrics_path = os.path.join(OUTPUT_DIR, "zero_shot_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")

    # ======================== 按类别统计 + 打印 case 供分析 ========================
    results = eval_output["results"]

    # 三类: (1) format=1, answer=1  (2) format=1, answer=0  (3) format=0, answer=0
    cat1 = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0]
    cat2 = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0]
    cat3 = [r for r in results if r["format_reward"] == 0.0 and r["answer_reward"] == 0.0]

    print(f"\n========== Category Counts ==========")
    print(f"(1) Correct (format=1, answer=1): {len(cat1)}")
    print(f"(2) Format ok, wrong answer (format=1, answer=0): {len(cat2)}")
    print(f"(3) Bad format (format=0, answer=0): {len(cat3)}")

    # 打印 10 个 format_reward=0 的 case，方便观察是模型问题还是 parser 问题
    print(f"\n========== 10 cases with format_reward=0 ==========")
    for i, r in enumerate(cat3[:10]):
        print(f"\n--- Case {i+1} ---")
        print(f"Question: {r['question'][:100]}...")
        print(f"Ground truth: {r['ground_truth']}")
        print(f"Generated (last 300 chars): ...{r['generated_text'][-300:]}")

    # 打印 10 个 format=1 but answer=0 的 case
    print(f"\n========== 10 cases with format=1, answer=0 ==========")
    for i, r in enumerate(cat2[:10]):
        print(f"\n--- Case {i+1} ---")
        print(f"Question: {r['question'][:100]}...")
        print(f"Ground truth: {r['ground_truth']}")
        # 提取模型给出的 answer
        gen = r["generated_text"]
        if "<answer>" in gen:
            model_ans = gen.split("<answer>")[-1].replace("</answer>", "").strip()
            print(f"Model answer: {model_ans}")
        print(f"Generated (last 300 chars): ...{gen[-300:]}")
