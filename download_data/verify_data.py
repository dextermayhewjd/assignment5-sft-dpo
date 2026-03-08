from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/fredkeira/Data/models/Qwen/Qwen2.5-Math-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

print("Model loaded successfully ✅")
