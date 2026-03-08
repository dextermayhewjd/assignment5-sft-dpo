import torch

from sft_scripts.masked_normalize import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """执行一个 microbatch 的 SFT 前向 + 反向传播。

    SFT 的 loss 就是交叉熵，即 -log p(实际 token) 在 response 部分的加权和。
    policy_log_probs 已经是每个位置的 log p，所以 loss = -sum(log_probs * mask) / constant。

    Args:
        policy_log_probs: (batch_size, seq_len) 每个 token 的 log 概率。
        response_mask: (batch_size, seq_len) 1 表示 response token，0 表示 prompt/padding。
        gradient_accumulation_steps: 梯度累积步数，用于缩放 loss。
        normalize_constant: 归一化常数，对 loss 求和后除以它。

    Returns:
        (loss, metadata):
            loss: 标量，经过梯度累积缩放后的 microbatch loss。
            metadata: 额外的统计信息字典。
    """

    # 第一步：计算每个样本的 NLL
    # --------------------------
    # SFT 的目标是最大化 response token 的 log 概率，等价于最小化负 log 概率（NLL）。
    # -policy_log_probs 就是每个位置的 NLL。
    # 用 masked_normalize 在 seq 维度 (dim=-1) 上对每个样本的 response token 求和，
    # 再除以 normalize_constant，得到每个样本的归一化 NLL。
    per_sample_loss = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )  # (batch_size,)

    # 第二步：对 batch 维度取平均
    # --------------------------
    # 多个样本的 loss 取均值，这是标准做法——每个样本对 loss 的贡献相等，
    # 不会因为某个样本的 response 更长就主导梯度方向。
    loss = per_sample_loss.mean()  # 标量

    # 第三步：除以 gradient_accumulation_steps 并反向传播
    # --------------------------------------------------
    # 在梯度累积中，我们会连续调用多个 microbatch 的 backward()，梯度会自动相加。
    # 为了让累积后的梯度等价于"把所有 microbatch 合并成一个大 batch 再算"，
    # 需要把每个 microbatch 的 loss 除以累积步数。
    # 这样 accumulated_grad = sum(grad_i / N) = (1/N) * sum(grad_i)，就是平均梯度。
    scaled_loss = loss / gradient_accumulation_steps

    # 第四步：反向传播
    # ---------------
    # 调用 backward() 计算梯度并累积到 policy_log_probs.grad 上。
    # 注意：这里不做 zero_grad，因为梯度累积需要多个 microbatch 的梯度叠加，
    # zero_grad 由外部的训练循环在 optimizer.step() 之后负责。
    scaled_loss.backward()

    # 返回缩放后的 loss（用于 logging）和 metadata
    metadata = {}
    return scaled_loss, metadata
