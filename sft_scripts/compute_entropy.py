import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算每个 token 位置上 next-token 预测的熵（即在 vocab 维度上的熵）。

    熵的定义: H(p) = -sum_i p(x_i) * log p(x_i)
    衡量的是模型对下一个 token 的"不确定性"——熵越高，模型越不确定。

    Args:
        logits: torch.Tensor
            形状为 (batch_size, sequence_length, vocab_size) 的未归一化 logits。
            这是模型最后一层的原始输出，还没有经过 softmax 归一化。

    Returns:
        torch.Tensor
            形状为 (batch_size, sequence_length)，每个位置的预测熵。
    """

    # 第一步：用 log_softmax 将 logits 转为 log 概率
    # -----------------------------------------------
    # 为什么不直接 softmax 再取 log？因为 softmax 中的 exp(logit) 可能溢出（logit 很大时）
    # 或下溢（logit 很小时）。
    # 先找到最大值 $m = \max(z)$，然后每个 logit都减去它：
    # log_softmax 内部使用了 logsumexp 技巧：
    #   log_softmax(x_i) = x_i - log(sum_j exp(x_j))
    #                    = x_i - logsumexp(x)
    # 其中 logsumexp 会先减去 max(x) 来避免数值溢出，保证数值稳定。
    # dim=-1 表示在最后一个维度（vocab_size）上做归一化。
    # log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # 手动实现 log_softmax（等价于上面注释掉的一行）：
    # 1) 找到每个位置上所有 vocab logit 的最大值，用于数值稳定
    #    keepdim=True 保持维度，方便后面广播减法
    m = logits.max(dim=-1, keepdim=True).values  # (batch_size, seq_len, 1)

    # 2) 每个 logit 减去最大值，这样最大的变成 0，其余都 <= 0
    #    exp 之后值域在 (0, 1]，不会溢出
    shifted = logits - m  # (batch_size, seq_len, vocab_size)

    # 3) 计算 logsumexp = m + log(sum(exp(shifted)))
    #    这就是 log(sum(exp(logits))) 的数值稳定版本
    logsumexp = m + torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))  # (batch_size, seq_len, 1)

    # 4) log_softmax = logits - logsumexp
    #    即 log(p_i) = z_i - log(sum_j exp(z_j))
    log_probs = logits - logsumexp  # (batch_size, seq_len, vocab_size)

    # 第二步：从 log 概率恢复出概率值 p(x)
    # ------------------------------------
    # 因为 log_probs 已经是数值稳定计算的结果，直接 exp 即可得到正确的概率值。
    # 这样得到的 probs 保证 >= 0 且 sum = 1。
    probs = torch.exp(log_probs)  # (batch_size, seq_len, vocab_size)

    # 第三步：计算熵 H = -sum_i p(x_i) * log p(x_i)
    # -----------------------------------------------
    # 对 vocab 维度 (dim=-1) 求和，得到每个 token 位置的熵标量。
    # 关于 p≈0 的边界情况：当某个 token 的概率极小时，
    # log_prob → -inf，但 prob → 0，所以 prob * log_prob → 0（因为 0 * (-inf) = 0）。
    # PyTorch 的 float 运算会正确处理这一点，不会产生 NaN。
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size, seq_len)

    return entropy
