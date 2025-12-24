from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 使用 Hugging Face 清华镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
local_dir = "./model/Qwen2.5-0.5B-Instruct"

# 下载模型和分词器到本地指定文件夹
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
