import torch


def masked_mean(
    tensor: torch.Tensor,  # 待求均值的数据tensor
    mask: torch.Tensor,  # 与tensor同形状的mask，1表示参与计算，0表示忽略
    dim: int | None = None,  # 沿哪个维度求均值，None表示对所有被mask选中的元素求全局均值
) -> torch.Tensor:
    """
    计算tensor在指定维度上的masked均值，只对mask==1的位置求平均。

    用途：在RL训练中，我们的损失tensor形状为 (batch_size, sequence_length)，
    但只需要对response部分（mask==1）的token求平均，忽略prompt和padding部分。
    也可用于计算response token上的平均entropy、clip比例等统计量。

    参数:
        tensor: 待求均值的数据
        mask: 与tensor同形状，1的位置参与均值计算，0的位置被忽略
        dim: 沿哪个维度求均值。若为None，则对所有mask==1的元素求全局标量均值

    返回:
        masked均值，形状与 tensor.mean(dim) 的语义一致
    """
    # 将mask转为与tensor相同的dtype，方便做乘法和求和
    mask = mask.to(tensor.dtype)  # 确保mask和tensor类型一致，例如都是float32

    if dim is None:  # 全局均值模式：对所有被mask选中的元素求一个标量均值
        # 分子：所有mask==1位置的元素之和
        numerator = (tensor * mask).sum()  # 先逐元素乘mask（将mask==0的位置置零），再全局求和
        # 分母：mask中1的总数（即参与计算的元素个数）
        denominator = mask.sum()  # 统计有多少个位置被选中
        return numerator / denominator  # 求均值并返回标量
    else:  # 沿指定维度求均值
        # 分子：沿dim维度，对mask==1位置的元素求和
        numerator = (tensor * mask).sum(dim=dim)  # 逐元素乘mask后沿dim求和
        # 分母：沿dim维度，统计每个切片中mask==1的个数
        denominator = mask.sum(dim=dim)  # 沿dim维度统计选中元素数
        return numerator / denominator  # 逐元素除法，得到每个切片的masked均值
'''
  核心逻辑：                                                        
  - (tensor * mask).sum(dim) —                                      
  先把mask==0的位置置零，再沿指定维度求和（分子）                   
  - mask.sum(dim) — 统计每个切片中mask==1的个数（分母）             
  - 两者相除得到masked均值                                          
  - dim=None 时对所有选中元素求全局标量均值                         
'''