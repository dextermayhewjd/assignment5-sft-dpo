import torch
from typing import Literal

from grpo_scripts.compute_policy_gradient_loss import compute_policy_gradient_loss
from grpo_scripts.masked_mean import masked_mean
from sft_scripts.masked_normalize import masked_normalize


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,  # 当前策略的逐token对数概率，形状 (batch_size, sequence_length)
    response_mask: torch.Tensor,  # response token的mask，1表示response，0表示prompt/padding，形状 (batch_size, sequence_length)
    gradient_accumulation_steps: int,  # 梯度累积步数，用于缩放loss
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],  # 策略梯度损失类型
    raw_rewards: torch.Tensor | None = None,  # 原始奖励，形状 (batch_size, 1)，no_baseline时需要
    advantages: torch.Tensor | None = None,  # 优势值，形状 (batch_size, 1)，reinforce_with_baseline和grpo_clip时需要
    old_log_probs: torch.Tensor | None = None,  # 旧策略的逐token对数概率，形状 (batch_size, sequence_length)，grpo_clip时需要
    cliprange: float | None = None,  # clip参数ε，grpo_clip时需要
    length_norm: str = "masked_mean",  # 序列长度聚合方式：masked_mean 或 masked_normalize
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行一个microbatch的GRPO前向+反向传播。

    流程:
      1. 调用compute_policy_gradient_loss得到逐token损失 (batch_size, sequence_length)
      2. 在sequence维度上聚合（masked_mean 或 masked_normalize，由 length_norm 控制），只对response token计算，得到每个样本的标量损失 (batch_size,)
      3. 对batch维度取均值，得到标量loss
      4. 除以gradient_accumulation_steps进行缩放
      5. 调用backward()反向传播

    参数:
        policy_log_probs: 形状 (batch_size, sequence_length)，当前策略的逐token对数概率
        response_mask: 形状 (batch_size, sequence_length)，1表示response token，0表示prompt/padding
        gradient_accumulation_steps: 梯度累积步数，loss会除以此值
        loss_type: 三种策略梯度损失类型之一
        raw_rewards: 形状 (batch_size, 1)，no_baseline时必须提供
        advantages: 形状 (batch_size, 1)，reinforce_with_baseline和grpo_clip时必须提供
        old_log_probs: 形状 (batch_size, sequence_length)，grpo_clip时必须提供
        cliprange: clip参数ε，grpo_clip时必须提供

    返回:
        loss: 标量tensor，经过梯度累积缩放后的microbatch损失（用于日志记录）
        metadata: 字典，包含底层损失函数返回的统计信息
    """
    # ===== 第1步：计算逐token的策略梯度损失 =====
    # 根据loss_type分发到对应的损失函数，得到形状 (batch_size, sequence_length) 的逐token损失
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,  # 当前策略的对数概率
        loss_type=loss_type,  # 损失类型
        raw_rewards=raw_rewards,  # 原始奖励（可能为None）
        advantages=advantages,  # 优势值（可能为None）
        old_log_probs=old_log_probs,  # 旧策略的对数概率（可能为None）
        cliprange=cliprange,  # clip参数（可能为None）
    )  # per_token_loss 形状 (batch_size, sequence_length)

    # ===== 第2步：在sequence维度上聚合 =====
    # 根据 length_norm 选择聚合方式
    if length_norm == "masked_mean":
        # 只对response token求均值（除以response token数量）
        per_sample_loss = masked_mean(
            tensor=per_token_loss,
            mask=response_mask,
            dim=-1,
        )  # 形状 (batch_size,)
    elif length_norm == "masked_normalize":
        # 对response token求和后除以最大序列长度（Dr. GRPO做法）
        # 短序列的梯度被缩小，消除长度对梯度大小的影响
        seq_len = per_token_loss.shape[1]
        per_sample_loss = masked_normalize(
            tensor=per_token_loss,
            mask=response_mask,
            normalize_constant=float(seq_len),
            dim=-1,
        )  # 形状 (batch_size,)
    else:
        raise ValueError(f"不支持的length_norm: {length_norm}")

    # ===== 第3步：对batch维度取均值 =====
    # 多个样本的loss取均值，每个样本对loss的贡献相等
    loss = per_sample_loss.mean()  # 标量

    # ===== 第4步：除以gradient_accumulation_steps进行缩放 =====
    # 梯度累积时，多个microbatch的backward()会自动累加梯度
    # 除以累积步数使得累积后的梯度等价于大batch的平均梯度
    scaled_loss = loss / gradient_accumulation_steps  # 缩放后的标量loss

    # ===== 第5步：反向传播 =====
    # 计算梯度并累积到各参数的.grad上
    # 不做zero_grad，因为梯度累积需要多个microbatch的梯度叠加
    scaled_loss.backward()  # 反向传播

    return scaled_loss, metadata  # 返回缩放后的loss（用于logging）和元数据

'''
  流程总结：                                                        
  1. compute_policy_gradient_loss → 逐token损失 (batch_size,        
  seq_len)                                                          
  2. masked_mean(dim=-1) → 只对response                             
  token求均值，得到每个样本的损失 (batch_size,)                     
  3. .mean() → batch维度取均值，得到标量                            
  4. / gradient_accumulation_steps →                                
  缩放，使累积梯度等价于大batch平均梯度                             
  5. .backward() → 反向传播，梯度累积到 .grad 上   
'''