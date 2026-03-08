import torch


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,  # 原始奖励或已归一化的优势值，形状 (batch_size, 1)
    policy_log_probs: torch.Tensor,  # 策略模型对每个token的对数概率，形状 (batch_size, sequence_length)
) -> torch.Tensor:
    """
    计算朴素策略梯度的逐token损失。

    公式: -A_t * log p_θ(o_t | q, o_{<t})
    即: 取负号 × 优势值 × 策略的对数概率

    这不是传统意义上的"损失"，不应作为评估指标使用。
    在RL中应该跟踪训练和验证的回报(return)等指标。

    参数:
        raw_rewards_or_advantages: 形状 (batch_size, 1)，每个rollout响应的标量奖励/优势值
        policy_log_probs: 形状 (batch_size, sequence_length)，每个token的对数概率

    返回:
        形状 (batch_size, sequence_length) 的tensor，逐token的策略梯度损失
        （后续在训练循环中会在batch和sequence维度上进行聚合）
    """
    # raw_rewards_or_advantages 形状为 (batch_size, 1)，会自动广播到 (batch_size, sequence_length)
    # 取负号：因为我们要最大化期望回报，而优化器做的是最小化，所以加负号
    loss = -raw_rewards_or_advantages * policy_log_probs  # 形状 (batch_size, sequence_length)

    return loss  # 返回逐token的策略梯度损失
'''
  原理：策略梯度定理告诉我们，要最大化期望回报，梯度方向是 A_t *    
  ∇log p_θ(o_t)。但优化器做的是最小化，所以加负号变成损失：-A_t *   
  log p_θ(o_t)。raw_rewards_or_advantages 形状 (batch_size, 1)      
  会自动广播到 (batch_size, sequence_length)，使得同一个响应的所有  
  token 共享同一个优势值。 
'''