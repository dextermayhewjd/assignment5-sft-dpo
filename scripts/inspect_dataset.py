'''
这个代码用来查看这个数据集究竟是如何构成的
uv run python scripts/inspect_dataset.py /home/fredkeira/Data/datasets/gsm8k/main/test-00000-of-00001.parquet

观察后可得 这里的回答有俩部分
- question：题目
- answer：标准解答文本，
    但它不是已经带好 <think>...</think> <answer>...</answer> 标签的
    
  {
    "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
  },
  比如这一条 
  答案的部分而#### 之前是解释 最后是结果
    Janet sells 16 - 3 - 4 = 9 duck eggs a day.
  She makes 9 * 2 = 18 every day at the farmer’s market.
  #### 18
  
    如果你想把它改造成更符合你这个模板的形式：

  A conversation between User and Assistant...
  User: {question}
  Assistant: <think>

  那么应该这样拆：

  - question 直接放进 User: {question}
  - answer 里 #### 前面的部分，作为 <think>...</think> 中的 reasoning
  - #### 后面的部分，作为 <answer>...</answer> 中的 final answer

  也就是：

  raw_answer = example["answer"]
  reasoning = raw_answer.split("####")[0].strip()
  final_answer = raw_answer.split("####")[-1].strip()

  target = f"{reasoning}</think> <answer>{final_answer}</answer>"
  
'''


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    """解析命令行参数，告诉脚本应该检查哪个 parquet 文件，以及打印几条样本。"""
    parser = argparse.ArgumentParser(
        description="Inspect dataset content and format for a parquet dataset file."
    )
    # `path` 是必填位置参数。
    # 用户可以传一个 parquet 文件，也可以传一个只包含 parquet 的目录。
    parser.add_argument(
        "path",
        help="Path to a parquet file, or a directory containing parquet files.",
    )
    # `--samples` 控制最后打印多少条样本，默认打印 3 条。
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample rows to print. Default: 3.",
    )
    return parser.parse_args()


def resolve_dataset_path(path: Path) -> Path:
    """把用户传入的路径解析成最终真正要读取的 parquet 文件路径。"""
    # 如果用户传进来的是一个目录，就尝试在当前目录下找 parquet 文件。
    if path.is_dir():
        # `glob("*.parquet")` 只查当前目录这一层，不递归查更深层目录。
        parquet_files = sorted(path.glob("*.parquet"))
        # 如果当前目录里根本没有 parquet 文件，就直接报错。
        if not parquet_files:
            raise ValueError(f"No .parquet files found in directory: {path}")
        # 如果当前目录里有多个 parquet 文件，脚本无法替用户决定用哪一个。
        # 这里会把所有候选文件列出来，让用户显式传入具体文件路径。
        if len(parquet_files) > 1:
            choices = "\n".join(f"- {file}" for file in parquet_files)
            raise ValueError(
                "Multiple .parquet files found in directory. "
                "Please pass one file explicitly:\n"
                f"{choices}"
            )
        # 如果目录下刚好只有一个 parquet 文件，就直接使用它。
        return parquet_files[0]

    # 如果传入的不是目录，那就按普通文件处理。
    suffix = path.suffix.lower()
    # 这里只支持 parquet，所以后缀不是 `.parquet` 就报错。
    if suffix != ".parquet":
        raise ValueError(f"Unsupported file type: {suffix}. Only .parquet is supported.")
    # 文件后缀合法时，直接返回这个文件路径。
    return path


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取 parquet 数据，并返回 DataFrame 和一些元信息。"""
    # 先把输入路径规范化成最终可读取的 parquet 文件路径。
    dataset_path = resolve_dataset_path(path)
    # 用 pandas 读取 parquet。
    # 返回值里的 metadata 主要用于后面打印“实际读的是哪个文件”。
    return pd.read_parquet(dataset_path), {"format": "parquet", "dataset_path": dataset_path}


def truncate(value: Any, max_chars: int = 300) -> str:
    """把任意值转成字符串；如果太长，就截断，避免终端输出过长。"""
    # 无论原始值是数字、字符串还是别的对象，都先转成字符串。
    text = str(value)
    # 如果长度本来就不长，直接原样返回。
    if len(text) <= max_chars:
        return text
    # 如果太长，就保留前面部分，最后加 `...`。
    return text[: max_chars - 3] + "..."


def print_basic_summary(df: pd.DataFrame) -> None:
    """打印数据集最基础的结构信息：行数、列名、每列类型、空值数量。"""
    print("\n=== Basic Summary ===")
    # `len(df)` 是总行数。
    print(f"Rows: {len(df)}")
    # `df.columns` 是列名列表，方便先看这个数据集有哪些字段。
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    # 如果是空表，后面逐列统计没有意义，直接返回。
    if df.empty:
        return

    print("\n=== Column Summary ===")
    # 逐列检查每一列的数据类型和缺失情况。
    for column in df.columns:
        # `series` 是当前这一列。
        series = df[column]
        # `notna().sum()` 统计非空数量。
        non_null = int(series.notna().sum())
        # `isna().sum()` 统计空值数量。
        null_count = int(series.isna().sum())
        # 找当前列第一个非空值，用来推断这一列的 Python 实际类型。
        sample_value = next((value for value in series if pd.notna(value)), None)
        # 如果找到了非空值，就取它的 Python 类型名；否则记成 None。
        sample_type = type(sample_value).__name__ if sample_value is not None else "None"
        # 打印 pandas dtype、Python 类型、非空数量和空值数量。
        print(
            f"- {column}: pandas_dtype={series.dtype}, "
            f"python_type={sample_type}, non_null={non_null}, null={null_count}"
        )


def print_text_stats(df: pd.DataFrame) -> None:
    """只针对文本列打印长度统计，帮助判断问题和答案大概有多长。"""
    # 先收集哪些列是“文本列”。
    text_columns = []
    for column in df.columns:
        # 先去掉空值，避免后面拿第一个元素时报错。
        series = df[column].dropna()
        # 如果这一列去掉空值后已经没有内容，就跳过。
        if series.empty:
            continue
        # 取这一列第一个非空值，粗略判断它是不是字符串。
        first_value = series.iloc[0]
        # 如果是字符串，我们就把它视为文本列。
        if isinstance(first_value, str):
            text_columns.append(column)

    # 如果没有任何文本列，就不用打印文本长度统计。
    if not text_columns:
        return

    print("\n=== Text Length Stats ===")
    # 对每个文本列统计最短长度、平均长度和最长长度。
    for column in text_columns:
        # 先统一转成字符串，再计算每个样本的字符长度。
        lengths = df[column].dropna().astype(str).str.len()
        # 打印长度统计，便于快速判断文本规模。
        print(
            f"- {column}: min={int(lengths.min())}, "
            f"mean={lengths.mean():.1f}, max={int(lengths.max())}"
        )


def print_gsm8k_answer_check(df: pd.DataFrame) -> None:
    """检查 `answer` 列是否符合 GSM8K 常见格式：推理过程 + `####` + 最终答案。"""
    # 如果根本没有 `answer` 列，或者这一列全是空，就直接返回。
    if "answer" not in df.columns or df["answer"].dropna().empty:
        return

    # 去掉空值并转成字符串，确保后面字符串操作稳定。
    answers = df["answer"].dropna().astype(str)
    # 检查每一条 answer 是否包含 `####`，这是 GSM8K 常见的终答案分隔符。
    has_delimiter = answers.str.contains("####", regex=False)
    # 对包含 `####` 的样本，按 `####` 切开并取最后一段，作为最终答案。
    extracted = answers[has_delimiter].str.split("####").str[-1].str.strip()

    print("\n=== Answer Format Check ===")
    # 打印有多少条 answer 包含 `####`。
    print(f"Rows with `####`: {int(has_delimiter.sum())}/{len(answers)}")
    # 如果存在不包含 `####` 的样本，就打印第一条坏样本的索引位置。
    if (~has_delimiter).any():
        first_bad_index = answers[~has_delimiter].index[0]
        print(f"First row without `####`: index={first_bad_index}")

    # 如果成功提取出了 final answer，就打印前几条预览。
    if not extracted.empty:
        preview = [truncate(value, max_chars=60) for value in extracted.head(5).tolist()]
        print(f"Sample extracted final answers: {preview}")


def print_samples(df: pd.DataFrame, num_samples: int) -> None:
    """打印前几条样本，让用户直观看到每一行数据长什么样。"""
    # 空表没有样本可打印，直接返回。
    if df.empty:
        return

    print("\n=== Sample Rows ===")
    # 取前 `num_samples` 条样本。
    sample_df = df.head(num_samples).copy()
    # 把 DataFrame 逐行转成 dict，方便以 JSON 形式打印。
    sample_records = []
    for record in sample_df.to_dict(orient="records"):
        # 为了避免输出过长，对每个字段都做一次截断。
        sample_records.append({key: truncate(value) for key, value in record.items()})
    # `ensure_ascii=False` 可以保证中文正常显示。
    # `indent=2` 让输出更易读。
    print(json.dumps(sample_records, ensure_ascii=False, indent=2))


def main() -> None:
    """
    脚本整体流程：
    1. 解析命令行参数，拿到用户输入的路径和样本数量。
    2. 检查路径是否存在。
    3. 把目录或文件路径解析成最终 parquet 文件路径。
    4. 用 pandas 读取 parquet 数据。
    5. 依次打印：
       - 基本结构信息
       - 文本列长度统计
       - answer 列的 GSM8K 格式检查
       - 前几条样本内容
    """
    # 解析命令行参数。
    args = parse_args()
    # 规范化路径：
    # - `expanduser()` 处理 `~`
    # - `resolve()` 转成绝对路径
    path = Path(args.path).expanduser().resolve()

    # 如果路径不存在，直接报错，避免后面读取失败。
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # 读取数据并拿到元信息。
    df, metadata = load_dataset(path)

    # 打印当前实际读取的是哪个 parquet 文件。
    print(f"Inspecting: {metadata['dataset_path']}")
    # 当前脚本只支持 parquet，所以这里会打印 parquet。
    print(f"Detected format: {metadata['format']}")

    # 先看整体结构：多少行、多少列、各列类型如何。
    print_basic_summary(df)
    # 再看文本长度，帮助判断 question / answer 的规模。
    print_text_stats(df)
    # 然后专门检查 `answer` 列是否符合 GSM8K 的 `####` 格式。
    print_gsm8k_answer_check(df)
    # 最后打印前几条样本，直接观察数据内容。
    print_samples(df, args.samples)


if __name__ == "__main__":
    main()
