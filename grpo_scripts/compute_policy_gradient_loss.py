import torch
from typing import Literal

from grpo_scripts.compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss
from grpo_scripts.compute_grpo_clip_loss import compute_grpo_clip_loss
from grpo_scripts.compute_grpo_no_clip_loss import compute_grpo_no_clip_loss


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,  # 当前策略的逐token对数概率，形状 (batch_size, sequence_length)
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],  # 损失类型选择
    raw_rewards: torch.Tensor | None = None,  # 原始奖励，形状 (batch_size, 1)，no_baseline时需要
    advantages: torch.Tensor | None = None,  # 组内归一化后的优势值，形状 (batch_size, 1)，reinforce_with_baseline和grpo_clip时需要
    old_log_probs: torch.Tensor | None = None,  # 旧策略的逐token对数概率，形状 (batch_size, sequence_length)，grpo_clip时需要
    cliprange: float | None = None,  # clip参数ε，grpo_clip时需要
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    策略梯度损失的统一包装器，根据loss_type分发到对应的损失计算函数。

    三种模式:
      - no_baseline: 朴素策略梯度，直接用原始奖励作为优势值 A = R(q, o)
      - reinforce_with_baseline: 朴素策略梯度，但用组内归一化后的奖励作为优势值 A = r̄
      - grpo_clip: GRPO-Clip损失，带ratio截断的策略梯度

    参数:
        policy_log_probs: 形状 (batch_size, sequence_length)，当前策略的逐token对数概率
        loss_type: 三种损失类型之一
        raw_rewards: 形状 (batch_size, 1)，no_baseline时必须提供
        advantages: 形状 (batch_size, 1)，reinforce_with_baseline和grpo_clip时必须提供
        old_log_probs: 形状 (batch_size, sequence_length)，grpo_clip时必须提供
        cliprange: clip参数ε，grpo_clip时必须提供

    返回:
        loss: 形状 (batch_size, sequence_length)，逐token损失
        metadata: 字典，包含底层函数返回的统计信息（如grpo_clip的clip比例）
    """
    if loss_type == "no_baseline":  # 朴素策略梯度，不用baseline
        assert raw_rewards is not None, "no_baseline模式需要提供raw_rewards"  # 参数检查
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)  # 直接用原始奖励计算
        metadata = {}  # 朴素模式没有额外元数据

    elif loss_type == "reinforce_with_baseline":  # 带baseline的REINFORCE
        assert advantages is not None, "reinforce_with_baseline模式需要提供advantages"  # 参数检查
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)  # 用组内归一化的优势值计算
        metadata = {}  # 朴素模式没有额外元数据

    elif loss_type == "grpo_clip":  # GRPO-Clip损失
        assert advantages is not None, "grpo_clip模式需要提供advantages"  # 参数检查
        assert old_log_probs is not None, "grpo_clip模式需要提供old_log_probs"  # 参数检查
        assert cliprange is not None, "grpo_clip模式需要提供cliprange"  # 参数检查
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)  # 调用GRPO-Clip计算

    elif loss_type == "grpo_no_clip":  # GRPO-No-Clip（消融实验：去掉clip）
        assert advantages is not None, "grpo_no_clip模式需要提供advantages"
        assert old_log_probs is not None, "grpo_no_clip模式需要提供old_log_probs"
        loss, metadata = compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs)

    else:  # 不支持的损失类型
        raise ValueError(f"不支持的loss_type: {loss_type}")  # 抛出错误

    return loss, metadata  # 返回逐token损失和元数据
