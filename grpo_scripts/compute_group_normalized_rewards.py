import torch
from typing import Callable


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],  # 奖励函数，输入(响应, 真实答案)，输出包含reward等key的字典
    rollout_responses: list[str],  # 策略模型生成的所有rollout响应，长度 = n_prompts * group_size
    repeated_ground_truths: list[str],  # 重复了group_size次的真实答案，与rollout_responses等长
    group_size: int,  # 每个prompt对应的响应数量（即一组的大小）
    advantage_eps: float,  # 防止除零的小常数
    normalize_by_std: bool,  # 是否除以组内标准差（True=z-score归一化，False=只减均值）
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    计算每组rollout响应的奖励，并在组内进行归一化。

    GRPO（Group Relative Policy Optimization）的核心思想：
    对同一个prompt的多个响应，在组内做相对比较而非使用绝对奖励值，
    这样每个prompt都有自己的baseline，能有效降低方差。

    参数:
        reward_fn: 奖励函数，对响应和真实答案打分，返回包含 "reward"、"format_reward"、"answer_reward" 的字典
        rollout_responses: 策略模型的rollout响应列表，长度为 rollout_batch_size = n_prompts_per_rollout_batch * group_size
        repeated_ground_truths: 真实答案列表，每个答案重复了group_size次，与rollout_responses等长
        group_size: 每个问题（组）的响应数量
        advantage_eps: 归一化时防止除零的小常数epsilon
        normalize_by_std: 若为True，则用 (reward - mean) / (std + eps) 做z-score归一化；否则只减去组均值

    返回:
        advantages: 形状 (rollout_batch_size,)，组内归一化后的奖励（即优势值）
        raw_rewards: 形状 (rollout_batch_size,)，未归一化的原始奖励
        metadata: 字典，包含用于日志记录的各种统计量
    """
    rollout_batch_size = len(rollout_responses)  # 总的rollout响应数量

    # ===== 第1步：计算每个响应的原始奖励 =====
    raw_rewards_list = []  # 存储每个响应的总奖励
    format_rewards_list = []  # 存储每个响应的格式奖励
    answer_rewards_list = []  # 存储每个响应的答案奖励
    for response, gt in zip(rollout_responses, repeated_ground_truths):  # 遍历每个(响应, 真实答案)对
        result = reward_fn(response, gt)  # 调用奖励函数，得到包含三种奖励的字典
        raw_rewards_list.append(result["reward"])  # 取出总奖励
        format_rewards_list.append(result["format_reward"])  # 取出格式奖励
        answer_rewards_list.append(result["answer_reward"])  # 取出答案奖励

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)  # 转为tensor，形状 (rollout_batch_size,)

    # ===== 第2步：按组reshape =====
    n_prompts = rollout_batch_size // group_size  # 计算prompt的数量
    grouped_rewards = raw_rewards.view(n_prompts, group_size)  # reshape为 (n_prompts, group_size)，每行是同一个prompt的一组奖励

    # ===== 第3步：计算每组的均值和标准差 =====
    group_means = grouped_rewards.mean(dim=1, keepdim=True)  # 每组的均值，形状 (n_prompts, 1)
    group_stds = grouped_rewards.std(dim=1, keepdim=True)    # 每组的标准差，形状 (n_prompts, 1)

    # ===== 第4步：组内归一化，得到优势值 =====
    if normalize_by_std:  # 如果需要除以标准差
        normalized = (grouped_rewards - group_means) / (group_stds + advantage_eps)  # z-score归一化：(reward - mean) / (std + eps)
    else:  # 如果不除以标准差
        normalized = grouped_rewards - group_means  # 只减去组均值作为baseline

    advantages = normalized.view(rollout_batch_size)  # 展平回 (rollout_batch_size,)

    # ===== 第5步：收集用于日志记录的元数据 =====
    metadata = {
        "reward/mean": raw_rewards.mean().item(),  # 所有响应的平均奖励
        "reward/std": raw_rewards.std().item(),  # 所有响应的奖励标准差
        "reward/max": raw_rewards.max().item(),  # 最大奖励
        "reward/min": raw_rewards.min().item(),  # 最小奖励
        "format_reward/mean": sum(format_rewards_list) / len(format_rewards_list),  # 平均格式奖励
        "answer_reward/mean": sum(answer_rewards_list) / len(answer_rewards_list),  # 平均答案奖励
        "advantages/mean": advantages.mean().item(),  # 优势值的均值
        "advantages/std": advantages.std().item(),  # 优势值的标准差
    }

    return advantages, raw_rewards, metadata  # 返回：优势值、原始奖励、元数据
