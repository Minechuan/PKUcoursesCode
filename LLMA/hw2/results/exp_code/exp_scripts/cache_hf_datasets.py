import os
from datasets import load_dataset

# 设置清华 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置本地缓存目录
local_dir = "./datasets"

# 加载 PKU-Alignment 的 text-to-text 数据集
dataset = load_dataset(
    path="PKU-Alignment/Align-Anything",
    name="text-to-text",
    cache_dir=local_dir
)


print(dataset)
