'''
这个测试文件成功导入了qwen2.5并且 generate prompt
'''
from vllm import LLM, SamplingParams
from pathlib import Path
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 两张 3090
warnings.filterwarnings("ignore", category=UserWarning)  # 过滤 shared_memory 泄漏警告

MODEL_DIR = Path("/home/fredkeira/Data/models/Qwen/Qwen2.5-Math-1.5B")  # 改成你的路径

prompts = [
    "你是谁",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)
# Create an LLM.
llm = LLM(
    model=str(MODEL_DIR),
    # tensor_parallel_size=2,       # 两张卡做 tensor parallelism
    gpu_memory_utilization=0.9,  # 每张卡留 10% 余量，避免 OOM
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    
