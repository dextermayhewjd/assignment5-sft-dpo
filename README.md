  ---                                                                                                               
  基于 CoT 的数学推理后训练：SFT / STaR / GRPO
                                                                                                  
  围绕"如何让语言模型学会链式推理（Chain-of-Thought）"，在 Qwen2.5-Math-1.5B 上从零实现并对比三条后训练路线，GSM8K
  数据集：

  - 监督式 CoT（SFT）：在人工标注的 <think> 格式推理数据上做监督微调，从零实现梯度累积与 response mask
  损失；数据量从 128 扩展至 7,473 条，准确率 2.3% → 47.3%
  - 自举式 CoT（Expert Iteration / STaR）：模型自己生成 CoT 推理 → 筛选答案正确的样本 →
  迭代微调，无需额外人工标注；2,048 条自生成数据达 41.9%，验证了 CoT 能力的自我迭代路径
  - 强化式 CoT（GRPO）：以答案正确性为 reward，用 GRPO 直接强化 CoT 推理过程；实现三种 loss
  变体（reinforce_with_baseline / grpo_clip / grpo_no_clip），on-policy 基线 43.3%，改进版（off-policy + clip + Dr.
  GRPO 归一化）[TBD]%
  - 工程实现：双卡异构架构（GPU 0 训练 / GPU 1 vLLM 推理），权重实时同步；系统性消融实验覆盖 CoT
  提示词格式、长度归一化、组内方差归一化、off-policy 数据复用共 5 组对照
