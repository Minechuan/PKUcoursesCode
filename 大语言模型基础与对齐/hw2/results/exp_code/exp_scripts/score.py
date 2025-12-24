import sys
import os

# 添加 align-anything 项目根目录到 Python 路径
sys.path.append("/data/align-anything")

from align_anything.models.pretrained_model import load_pretrained_models
from transformers import AutoTokenizer
import torch
import json  # 从你的训练代码库导入
import numpy as np

# 加载 Reward Model 和 Tokenizer（对齐训练配置）
REWARD_MODEL_PATH = "/data/align-anything/outputs/part1/qwen_2_5_hw/slice_end"
# REWARD_MODEL_PATH = "/data/Qwen2.5-0.5B-Instruct"
device = torch.device("npu:0")





# 使用与训练时相同的加载函数
model, tokenizer, _ = load_pretrained_models(
    REWARD_MODEL_PATH,
    is_reward_model=True,  # 关键参数，确保模型结构一致
    padding_side='right',   # 与训练时对齐
    trust_remote_code=True  # 如果模型有自定义代码
)
model = model.to(device)

# 加载生成的 responses
print(f"Scoring responses with reward model: {REWARD_MODEL_PATH}")
with open("generated_responses.json", "r") as f:
    data = json.load(f)

score_accu=[]
scored_data = []
for item in data:
    prompt = item["prompt"]
    response = item["response"]
    
    # 构建输入文本（根据训练时的格式）
    input_text = f"Question: {prompt}\nAnswer: {response}"
    
    # Tokenize（对齐训练时的处理）
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True  # 确保与训练时一致
    ).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 提取最终奖励分数（对齐训练时的 end_scores）
    end_scores = outputs.end_scores  # 形状: (batch_size=1, 1)
    score = end_scores.squeeze().item()  # 标量值
    
    scored_data.append({
        "prompt": prompt,
        "response": response,
        "reward_score": score
    })
    score_accu.append(score)

# 保存结果
with open("scores_rlhf_dpo.json", "w") as f:
    json.dump(scored_data, f, indent=4)

print("Scoring completed! Results saved to scores.json.")
print(f"The largest score is {max(score_accu)}, smallest score is {min(score_accu)}, mean is {np.mean(score_accu)}")