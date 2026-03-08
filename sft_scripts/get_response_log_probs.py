import torch
from transformers import PreTrainedModel

from sft_scripts.compute_entropy import compute_entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """获取每个 token 位置的条件 log 概率 log p(x_t | x_{<t})，以及可选的 token 熵。

    Args:
        model: HuggingFace 因果语言模型，用于产生 logits。
        input_ids: (batch_size, sequence_length) 拼接好的 prompt + response token ids。
        labels: (batch_size, sequence_length) 右移一位的 token ids（模型需要预测的目标）。
        return_token_entropy: 是否同时返回每个位置的预测熵。

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": (batch_size, sequence_length) 每个位置的条件 log 概率。
            "token_entropy": 可选，(batch_size, sequence_length) 每个位置的预测熵。
    """

    # 第一步：前向传播，获取模型输出的 logits
    # ----------------------------------------
    # model(input_ids) 返回一个对象，其中 .logits 形状为 (batch_size, seq_len, vocab_size)
    # logits[b, t, v] 表示在位置 t，给定 x_{<t+1} 的条件下，词表中 token v 的未归一化分数
    logits = model(input_ids).logits  # (batch_size, seq_len, vocab_size)

    # 第二步：计算 log_softmax，得到 log 概率分布
    # --------------------------------------------
    # log_softmax 在 vocab 维度上做归一化，数值稳定（内部用 logsumexp 技巧）
    # log_probs_all[b, t, v] = log p(v | x_{<=t})
    log_probs_all = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # 第三步：用 gather 取出 labels 对应的 log 概率
    # -----------------------------------------------
    # labels[b, t] 告诉我们位置 t 实际出现的 token id
    # 我们要从 log_probs_all[b, t, :] 中取出第 labels[b, t] 个元素
    # gather(dim=-1, index) 要求 index 形状与输出一致，所以 unsqueeze 再 squeeze
    #
    # 直觉：log_probs_all 是一个 "每个位置上所有可能 token 的 log 概率表"，
    # gather 就是在这张表里按 labels 查表，取出实际 token 的 log 概率
    log_probs = log_probs_all.gather(
        dim=-1,
        index=labels.unsqueeze(-1)  # (batch_size, seq_len, 1)
    ).squeeze(-1)  # (batch_size, seq_len)

    result = {"log_probs": log_probs}

    # 第四步（可选）：计算每个位置的预测熵
    # ------------------------------------
    # 复用之前实现的 compute_entropy，输入是 logits，输出是每个位置的熵
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)  # (batch_size, seq_len)

    return result
'''
  这个函数到底在算什么                                        
   
  假设 prompt = "1 2"，output = "3 4"，拼接后：               
                  
  input_ids:  [1, 2, 3, 4]
  labels:     [2, 3, 4, 0]   （右移一位，最后补0）

  第一步：模型前向传播

  模型看到 input_ids，在每个位置产出一个 vocab 大小的 logits
  向量：

  位置0: 模型看到 [1]       → logits[0] = 对整个 vocab 的打分
  位置1: 模型看到 [1,2]     → logits[1] = 对整个 vocab 的打分
  位置2: 模型看到 [1,2,3]   → logits[2] = 对整个 vocab 的打分
  位置3: 模型看到 [1,2,3,4] → logits[3] = 对整个 vocab 的打分

  第二步：log_softmax 归一化

  把每个位置的 logits 转成 log 概率分布，得到一张"概率表"。

  第三步：用 labels 查表

  labels 告诉我们每个位置实际出现的下一个 token 是什么，gather
   就是去概率表里查这个 token 的 log 概率：

  位置0: labels=2 → 查出 log p(token 2 | 看到 [1])        ←
  prompt 内部预测
  位置1: labels=3 → 查出 log p(token 3 | 看到 [1,2])      ←
  预测第一个 output token
  位置2: labels=4 → 查出 log p(token 4 | 看到 [1,2,3])    ←
  预测第二个 output token
  位置3: labels=0 → 查出 log p(token 0 | 看到 [1,2,3,4])  ←
  padding，后续会被 mask 掉

  所以本质就是

  给定前文，模型认为实际出现的那个 token 的概率有多大？

  - 概率高（log_prob 接近 0）→ 模型对这个 token 预测得很准
  - 概率低（log_prob 是很大的负数）→ 模型没预测对，loss 会大

  这就是为什么 SFT 的 loss 直接就是 -log_prob 取平均——让模型在
   response 部分每个位置都尽量给出高概率。至于 prompt 和
  padding 位置的 log_prob，虽然这里也算了，但后面训练时会用
  response_mask 把它们遮掉，不参与 loss 计算。
'''