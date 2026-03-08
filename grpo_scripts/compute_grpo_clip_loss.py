import torch


def compute_grpo_clip_loss(
    advantages: torch.Tensor,  # 每个样本的优势值，形状 (batch_size, 1)
    policy_log_probs: torch.Tensor,  # 当前策略的逐token对数概率，形状 (batch_size, sequence_length)
    old_log_probs: torch.Tensor,  # 旧策略的逐token对数概率，形状 (batch_size, sequence_length)
    cliprange: float,  # clip参数 ε，例如0.2
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算GRPO-Clip的逐token损失。

    公式: -min( ratio * A_t,  clip(ratio, 1-ε, 1+ε) * A_t )
    其中 ratio = π_θ(o_t|q,o_{<t}) / π_θ_old(o_t|q,o_{<t})
              = exp(log_π_θ - log_π_θ_old)

    clip的作用是限制策略更新的幅度，防止新策略偏离旧策略太远，
    这是PPO/GRPO的核心思想，保证训练稳定性。

    参数:
        advantages: 形状 (batch_size, 1)，每个样本的优势值A
        policy_log_probs: 形状 (batch_size, sequence_length)，当前正在训练的策略的逐token对数概率
        old_log_probs: 形状 (batch_size, sequence_length)，旧策略的逐token对数概率
        cliprange: clip参数ε

    返回:
        loss: 形状 (batch_size, sequence_length)，逐token的clipped损失
        metadata: 字典，包含每个token是否被clip的信息
    """
    # ===== 第1步：计算新旧策略的概率比 ratio =====
    # 在log空间做减法等价于概率空间做除法：log(π_θ/π_old) = log π_θ - log π_old
    log_ratio = policy_log_probs - old_log_probs  # 形状 (batch_size, sequence_length)
    ratio = torch.exp(log_ratio)  # 转回概率比，形状 (batch_size, sequence_length)

    # ===== 第2步：计算未clip的策略梯度项 =====
    # 左边项：ratio * A_t，advantages会从(batch_size,1)广播到(batch_size,sequence_length)
    unclipped = ratio * advantages  # 形状 (batch_size, sequence_length)

    # ===== 第3步：计算clip后的策略梯度项 =====
    # 将ratio限制在 [1-ε, 1+ε] 范围内，防止策略更新太激进
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)  # 形状 (batch_size, sequence_length)
    clipped = clipped_ratio * advantages  # clip后的ratio乘以优势值

    # ===== 第4步：取min并取负，得到最终损失 =====
    # 取min是悲观策略：选择改进较小的那个，进一步保证保守更新
    loss = -torch.min(unclipped, clipped)  # 取负号因为优化器最小化，但我们要最大化回报

    # ===== 第5步：记录哪些token被clip了 =====
    # 当clipped项 < unclipped项时，说明该token被clip了（即min选择了右边的clipped项）
    clipped_mask = (clipped < unclipped).float()  # 1.0表示被clip，0.0表示未被clip

    metadata = {
        "clipped": clipped_mask,  # 每个token是否被clip的mask，形状 (batch_size, sequence_length)
    }

    return loss, metadata  # 返回逐token损失和元数据

'''
核心逻辑：                                                        
                                                                    
  1. 计算ratio：exp(log π_θ - log π_old) — 新旧策略的概率比         
  2. 未clip项：ratio * A_t                                          
  3. clip项：clamp(ratio, 1-ε, 1+ε) * A_t — 把ratio限制在 [1-ε, 1+ε]
   范围                                                             
  4. 取min再取负：-min(未clip项, clip项) —                          
  悲观策略，选改进较小的那个                                        
                                                                    
  clip的意义：当ratio偏离1太远（即新策略和旧策略差距太大），就截断它
  ，防止单步更新太激进导致训练崩溃。这是PPO/GRPO保证训练稳定性的核心
  机制。
'''