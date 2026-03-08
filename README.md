  ---                                                                                        
  nano-vLLM：从零实现高性能 LLM 推理引擎                                                     
                                                                                             
  技术栈：Python / PyTorch / Triton / CUDA                                                   
                                                                                             
  项目描述：从零实现一个面向 Qwen3-8B（8B 参数，GQA 架构）的高性能 LLM 推理引擎，聚焦 vLLM   
  核心优化技术的原理复现，在单张 RTX 3090 上实现 1100+ tok/s 的总吞吐。                      
                                                                                             
  核心工作：                                                                                 
                                                                                             
  - PagedAttention 内存管理：实现基于 Block 的 KV Cache 分页管理系统（BlockAllocator），通过 
  block_table 间接寻址将逻辑 token 位置映射到物理 slot，消除 KV Cache
  的内存碎片问题，单卡可管理约 2400 个 block（~38400 tokens）
  - Triton Paged Attention Kernel：使用 Triton 编写 decode 阶段的分页注意力核函数，支持 GQA  
  分组映射与 online softmax，直接从分散的物理 block 中读取 KV，避免 gather 带来的显存拷贝开销
  - Continuous Batching 调度：设计 waiting → running → finished 三队列调度器，支持动态 batch 
  组装；实现 prefill 端准入控制（block 容量检查）与 decode 端 LIFO 抢占策略，保障显存不溢出  
  - CUDA Graph 加速：对 decode 阶段（S_q=1）按 bucket size [1,2,4,8,16,32] 预捕获 CUDA       
  Graph，消除 kernel launch overhead，复用共享 memory pool                                   
  - Chunked Prefill：将长 prompt 按 512 token 分块处理，每个 step 交替执行一个 prefill chunk 
  和全量 decode batch，显著降低排队场景下的 TTFT；解决了 SDPA 在 S_q ≠ S_k 时的 causal mask  
  对齐问题及 FlashAttention/Math backend 精度不一致问题（通过 pad Q 保证走相同计算路径）

  性能指标（ShareGPT 数据集，200 并发请求）：
  - 输出吞吐：591.8 tok/s ｜ 总吞吐：1106.0 tok/s
