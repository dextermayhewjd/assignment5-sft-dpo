# download_dataset.py 完整代码（适配 Parquet 格式）
from datasets import load_dataset

def download_gsm8k_parquet():
    try:
        # 直接指定 Parquet 格式和具体配置版本
        # main 版本：基础版 GSM8K
        # socratic 版本：苏格拉底式对话版 GSM8K（可选）
        ds = load_dataset(
            "openai/gsm8k",
            name="main",  # 可替换为 "socratic" 切换版本
            data_format="parquet",  # 显式指定 Parquet 格式
            trust_remote_code=True  # 兼容新版数据集加载
        )
        
        # 保存到本地（Parquet 格式，占用空间更小）
        ds.save_to_disk("./gsm8k_parquet_dataset")
        print("✅ GSM8K 数据集（Parquet 格式）下载成功！")
        print(f"📊 训练集样本数：{len(ds['train'])}")
        print(f"📊 测试集样本数：{len(ds['test'])}")
        return ds
    except Exception as e:
        print(f"❌ 下载失败：{e}")
        raise

if __name__ == "__main__":
    download_gsm8k_parquet()