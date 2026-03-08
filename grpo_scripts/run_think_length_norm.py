"""
think_about_length_normalization

用代码演示 masked_mean vs masked_normalize 的梯度差异，无需训练。
对应作业 Problem (think_about_length_normalization)。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from grpo_scripts.masked_mean import masked_mean
from sft_scripts.masked_normalize import masked_normalize

# ===== 复现作业中的示例 =====
# batch=2：第一条 response 4 个 token，第二条 7 个 token
ratio = torch.tensor([
    [1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1.],
], requires_grad=True)

advs = torch.tensor([
    [2., 2., 2., 2., 2., 2., 2.],
    [2., 2., 2., 2., 2., 2., 2.],
])

masks = torch.tensor([
    [1., 1., 1., 1., 0., 0., 0.],   # response 1：4 tokens
    [1., 1., 1., 1., 1., 1., 1.],   # response 2：7 tokens
])

max_gen_len = 7  # padding 后的最大序列长度

# ===== masked_mean =====
mean_result = masked_mean(ratio * advs, masks, dim=-1)
print(f"masked_mean 结果: {mean_result}")
# 对两个样本都是 2.0（只对有效token求均值）
# 短序列(4tok)和长序列(7tok)的loss值相同

mean_result.mean().backward()
print(f"masked_mean 梯度:\n{ratio.grad}")
# 短序列每个token梯度 = 2/(4*2) = 0.25（被4个token平均）
# 长序列每个token梯度 = 2/(7*2) = 0.1429（被7个token平均）
# → 短序列的每个token梯度更大，等价于对短序列的"奖励"

ratio.grad.zero_()

# ===== masked_normalize =====
norm_result = masked_normalize(ratio * advs, masks, normalize_constant=float(max_gen_len), dim=-1)
print(f"\nmasked_normalize 结果: {norm_result}")
# 短序列: 4*2/7 = 1.1429，长序列: 7*2/7 = 2.0
# 短序列的loss值更小，因为分母统一是 max_gen_len=7

norm_result.mean().backward()
print(f"masked_normalize 梯度:\n{ratio.grad}")
# 两条序列每个有效token的梯度都是 1/(7*2) = 0.1429
# 无论序列长短，每个token梯度相同

print("""
===== 分析 =====

masked_mean（当前默认）:
  优点：
    - 对每个样本的loss归一化，不同长度样本对梯度的贡献相等（以样本为单位）
    - 实现简单，是大多数 NLL loss 的标准做法
  缺点：
    - 短序列每个 token 的梯度比长序列大（因为同样的 loss 被更少的 token 平均）
    - 模型可能倾向于生成更短的回答来获得更大梯度，导致长度偏置
    - 在强化学习中，这意味着短的正确回答会得到更强的梯度信号

masked_normalize（Dr. GRPO 做法）:
  优点：
    - 每个 token 的梯度大小与序列长度无关（统一除以 max_gen_len）
    - 更公平地对待不同长度的回答，避免长度偏置
    - 梯度更稳定（分母固定，不随样本变化）
  缺点：
    - 短序列的整体 loss 权重更小（分母相同但有效token更少）
    - 需要额外传入 max_gen_len 作为归一化常数
    - 如果序列长度分布差异很大，归一化效果有限

适合 masked_normalize 的场景：
    - 训练过程中发现模型倾向生成很短的回答（为了获得更大梯度）
    - 希望梯度稳定，不希望短 rollout 主导训练
    - off-policy 训练（epochs>1）时更需要稳定的梯度

适合 masked_mean 的场景：
    - 希望每个样本等权（不同长度样本平等贡献 loss）
    - 数据集中回答长度比较均匀
    - 与 SFT 阶段保持一致的 loss 计算方式
""")
