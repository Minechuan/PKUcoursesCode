from datasets import load_dataset
from transformers import AutoTokenizer
import json

# 加载数据集
dataset = load_dataset("/data/align_anything_t2t", split="validation")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/Qwen2.5-0.5B-Instruct")

# 创建一个空列表来保存每个 text
prompts = []

# 遍历validation集中的每个条目
for example in dataset:
    prompt = example['question']
    
    # 构建 messages（如果需要的话，可以在这里扩展或修改信息）
    prompt_entry = {"prompt": prompt}
    
    # 将 prompt 添加到列表
    prompts.append(prompt_entry)

# 将所有文本保存到 prompts.json 文件
with open("./prompts.json", "w") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)

print("All prompts have been saved to prompts.json!")
