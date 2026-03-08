"""
将 GSM8K 原始数据转为 SFT 训练格式。

原始格式 (data/gsm8k/train.jsonl):
    {"question": "...", "answer": "推理过程...\n#### 42"}

目标格式 (data/gsm8k/sft_train.jsonl):
    {"prompt": "<r1_zero模板填入question>", "response": "<think>推理过程</think> <answer>42</answer>"}

即把 question 套入 r1_zero prompt 模板作为 prompt，
把 answer 拆成推理过程和最终答案，包装成 <think>...</think> <answer>...</answer> 格式作为 response。
"""

import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")

PROMPT_TEMPLATE_PATH = os.path.join(PROJECT_DIR, "cs336_alignment", "prompts", "r1_zero.prompt")
INPUT_TRAIN = os.path.join(PROJECT_DIR, "data", "gsm8k", "train.jsonl")
INPUT_TEST = os.path.join(PROJECT_DIR, "data", "gsm8k", "test.jsonl")
OUTPUT_TRAIN = os.path.join(PROJECT_DIR, "data", "gsm8k", "sft_train.jsonl")
OUTPUT_VAL = os.path.join(PROJECT_DIR, "data", "gsm8k", "sft_val.jsonl")


def convert(input_path: str, output_path: str, prompt_template: str):
    """把原始 GSM8K jsonl 转成 SFT 格式。"""
    examples = []
    with open(input_path, "r") as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            answer_raw = item["answer"]

            # 清理 GSM8K 的计算注释标记 <<48/2=24>>，模型不需要学这些
            answer_clean = re.sub(r"<<.*?>>", "", answer_raw)

            # 解析 answer: "推理过程...\n#### 42"
            # #### 前面是 chain-of-thought，后面是最终数字答案
            parts = answer_clean.split("####")
            reasoning = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""

            # 构造 prompt: r1_zero 模板，以 <think> 结尾（模型从这里开始续写）
            prompt = prompt_template.format(question=question)

            # 构造 response: 推理过程 + 答案，用 <think></think> <answer></answer> 包裹
            response = f" {reasoning}\n</think> <answer> {final_answer} </answer>"

            examples.append({"prompt": prompt, "response": response})

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Converted {len(examples)} examples: {input_path} -> {output_path}")


if __name__ == "__main__":
    with open(PROMPT_TEMPLATE_PATH, "r") as f:
        prompt_template = f.read()

    convert(INPUT_TRAIN, OUTPUT_TRAIN, prompt_template)
    convert(INPUT_TEST, OUTPUT_VAL, prompt_template)

    # 打印一个样例验证格式
    with open(OUTPUT_TRAIN, "r") as f:
        ex = json.loads(f.readline())
    print(f"\n--- Example ---")
    print(f"Prompt:\n{ex['prompt']}")
    print(f"\nResponse:\n{ex['response']}")
