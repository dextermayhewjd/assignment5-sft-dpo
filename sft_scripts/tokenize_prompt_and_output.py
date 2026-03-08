import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """分别对 prompt 和 output 进行 tokenize，拼接后构造 input_ids、labels 和 response_mask。

    Args:
        prompt_strs: 提示字符串列表
        output_strs: 输出字符串列表
        tokenizer: 用于分词的 tokenizer

    Returns:
        dict[str, torch.Tensor]:
            "input_ids":      (batch_size, max_len - 1) 去掉最后一个 token 的完整序列
            "labels":         (batch_size, max_len - 1) 去掉第一个 token 的完整序列（右移一位）
            "response_mask":  (batch_size, max_len - 1) 在 labels 中 output 部分为1，其余为0
    """
    batch_size = len(prompt_strs)

    # 第一步：分别对 prompt 和 output 编码，不添加特殊 token
    prompt_token_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs
    ]
    output_token_ids = [
        tokenizer.encode(o, add_special_tokens=False) for o in output_strs
    ]

    # 第二步：拼接 prompt + output 的 token ids，记录各自长度
    combined_token_ids = [
        p + o for p, o in zip(prompt_token_ids, output_token_ids)
    ]
    prompt_lens = [len(p) for p in prompt_token_ids]
    combined_lens = [len(c) for c in combined_token_ids]
    max_len = max(combined_lens)

    # 第三步：用 pad_token_id 将所有序列填充到 max_len
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    padded = [
        c + [pad_token_id] * (max_len - len(c)) for c in combined_token_ids
    ]
    # 每一个prompt + output 组合在一起之后加入pad pad到最长的组合的长度
    
    full_ids = torch.tensor(padded, dtype=torch.long)  # (batch_size, max_len)
    # 把list变成tensor

    # 第四步：构造 input_ids 和 labels
    # input_ids = 去掉最后一个 token（模型的输入）              1 2 3 4 pad pad
    # labels    = 去掉第一个 token（模型要预测的目标，即右移一位） 2 3 4 pad pad pad
    input_ids = full_ids[:, :-1]  # (batch_size, max_len - 1)
    labels = full_ids[:, 1:]      # (batch_size, max_len - 1)

    # 第五步：构造 response_mask
    #
    # 原始序列 full_ids: [p0, p1, ..., p_{L-1}, o0, o1, ..., o_{M-1}, pad, ...]
    # output token 在 full_ids 中的下标范围: [prompt_len, combined_len - 1]
    #
    # labels[j] = full_ids[j+1]，所以 labels 第 j 个位置对应 full_ids 第 j+1 个 token
    # 要让 labels[j] 落在 output 区间，需要:
    #   prompt_len <= j+1 <= combined_len - 1
    #   => j 的范围: [prompt_len - 1, combined_len - 2]
    #
    # 因此 response_mask 在这个范围内置1，其余为0
    response_mask = torch.zeros_like(labels, dtype=torch.float)
    for i in range(batch_size):
        resp_start = prompt_lens[i] - 1 
        #这里是prompt的最后一个token 也就是第一个input 对应label里面是repsonse的部分
        
        resp_end = combined_lens[i] - 2  # 包含此位置
        # 这是一个prompt+response 中最后一个还要产生label的地方 即len - 1 -1
        if resp_start <= resp_end:
            response_mask[i, resp_start : resp_end + 1] = 1.0

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


'''
response_mask 的作用是：告诉训练过程，labels 中哪些位置是   
  output（需要计算 loss），哪些是 prompt 或 padding（不算     
  loss）。                                                    
                                                              
  用一个具体例子来说明：                                      
                  
  假设 prompt = "1 2"，output = "3 4"，pad 补到长度 8         
                                                              
  full_ids:  [1, 2, 3, 4, pad, pad, pad, pad]
              ---- ----  --------
              prompt     output     padding

  input_ids: [1, 2, 3, 4,   pad, pad, pad]   （去掉最后一个）
  labels:    [2, 3, 4, pad, pad, pad, pad]   （去掉第一个）

  现在问题是：labels 里的每个位置，模型都会预测一个
  token，但我们 只想对 output 部分算 loss。

  - labels[0] = 2：这是在"已知 prompt token 1，预测下一个
  prompt token 2"——属于 prompt 内部的预测，不算 loss
  - labels[1] = 3：这是在"已知 prompt 1 2，预测第一个 output
  token 3"——算 loss
  - labels[2] = 4：预测第二个 output token——算 loss
  - labels[3] = pad：output 结束了——不算 loss

  所以：
  response_mask: [0, 1, 1, 0, 0, 0, 0]

  数学上怎么算这个范围：
  - output 在 full_ids 中的下标是 [prompt_len,
  combined_len-1]，即 [2, 3]
  - labels[j] = full_ids[j+1]，所以要让 full_ids[j+1] 落在
  output 范围内：
    - prompt_len <= j+1 → j >= prompt_len - 1 = 1
    - j+1 <= combined_len - 1 → j <= combined_len - 2 = 2
  - 所以 response_mask[1:3] = 1，其余为 0

  为什么要这样做？ 因为 SFT 的目标是让模型学会"给定
  prompt，生成正确的 output"。prompt
  本身不是模型需要学习生成的内容，所以不应该对 prompt
  部分的预测计算 loss。
'''