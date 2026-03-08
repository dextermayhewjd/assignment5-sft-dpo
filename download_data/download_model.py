from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    local_dir="/home/fredkeira/Data/models/Qwen/Qwen2.5-Math-1.5B",
    local_dir_use_symlinks=False  # 建议关掉，避免软链接坑
)
print("Saved to:", local_dir)
