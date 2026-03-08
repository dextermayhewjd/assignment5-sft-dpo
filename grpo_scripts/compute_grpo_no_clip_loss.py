import torch


def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,   # (batch_size, 1)
    policy_log_probs: torch.Tensor,   # (batch_size, sequence_length)
    old_log_probs: torch.Tensor,      # (batch_size, sequence_length)
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    GRPO-No-Clip：不做截断的 importance-weighted 策略梯度损失。

    公式: -ratio * A_t
    其中 ratio = π_θ / π_old = exp(log π_θ - log π_old)

    和 grpo_clip 的区别：去掉 min(ratio*A, clamp(ratio)*A) 中的 clamp，
    直接用 ratio * A_t。用于消融实验，观察 clip 是否真的必要。
    """
    log_ratio = policy_log_probs - old_log_probs   # (batch_size, seq_len)
    ratio = torch.exp(log_ratio)
    loss = -(ratio * advantages)                   # advantages 广播到 seq_len

    metadata = {}
    return loss, metadata
