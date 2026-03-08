import torch


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """对 tensor 中 mask==1 的元素求和，然后除以 normalize_constant 进行归一化。

    Args:
        tensor: 要处理的张量。
        mask: 与 tensor 同形状的掩码，1 表示参与计算，0 表示忽略。
        normalize_constant: 归一化常数，求和后除以它。
        dim: 沿哪个维度求和。如果为 None，则对所有维度求和（得到标量）。

    Returns:
        归一化后的结果。mask==0 的位置不参与求和。
    """

    # 第一步：将 mask==0 的位置清零
    # -----------------------------
    # tensor * mask：mask 为 1 的位置保留原值，为 0 的位置变成 0
    # 这样后面求和时，被 mask 掉的元素贡献为 0，不影响结果
    masked_tensor = tensor * mask  # 与 tensor 同形状

    # 第二步：沿指定维度求和
    # --------------------
    # dim=None 时 torch.sum 会对所有元素求和，返回标量
    # dim=0/1/-1 时沿对应维度求和，降掉该维度
    summed = torch.sum(masked_tensor, dim=dim)

    # 第三步：除以归一化常数
    # --------------------
    # 比如在 SFT 中，normalize_constant 可能是 response token 的总数，
    # 这样就得到了"每个 token 的平均 loss"
    return summed / normalize_constant
