'''
Load: question, response 1, response 2, over_all_respon
input_text = f"Question: {prompt}\nAnswer: {response}"
Store:  response 1's score, responses 2's score, over_all_id

Then use these Data to visualize the distribution of the score.
'''
import sys
import os
sys.path.append("/data/align-anything")
from align_anything.models.pretrained_model import load_pretrained_models
from transformers import AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
import json

device = torch.device("npu:0")
dataset = load_dataset("/data/align_anything_t2t", split="validation")
# REWARD_MODEL_PATH = "/data/align-anything/outputs/part1/qwen_2_5_hw/slice_end"
REWARD_MODEL_PATH = "/data/Qwen2.5-0.5B-Instruct"
model, tokenizer, _ = load_pretrained_models(
    REWARD_MODEL_PATH,
    is_reward_model=True,  # 关键参数，确保模型结构一致
    padding_side='right',   # 与训练时对齐
    trust_remote_code=True  # 如果模型有自定义代码
)
model = model.to(device)


data_entry=[]

for example in dataset:
    question = example['question']
    response_1=example['response_1']
    response_2=example['response_2']
    win_id=example['overall_response']
    val_data_entry = {"prompt": question,"response_1":response_1,"response_2":response_2,"win_id":win_id}

    data_entry.append(val_data_entry)


scored_data = []
for item in data_entry:
    prompt = item["prompt"]
    response_1 = item["response_1"]
    response_2 = item["response_2"]

    input_text_1 = f"Question: {prompt}\nAnswer: {response_1}"
    input_text_2 = f"Question: {prompt}\nAnswer: {response_2}"


    # Tokenize（对齐训练时的处理）
    inputs = tokenizer(
        input_text_1,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True  # 确保与训练时一致
    ).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    

    end_scores = outputs.end_scores  # 形状: (batch_size=1, 1)
    score_1 = end_scores.squeeze().item()  # 标量值
    


    inputs = tokenizer(
        input_text_2,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True  # 确保与训练时一致
    ).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 提取最终奖励分数（对齐训练时的 end_scores）
    end_scores = outputs.end_scores  
    score_2 = end_scores.squeeze().item() 
    
    scored_data.append({
        "prompt": prompt,
        "response_1": response_1,
        "score_1": score_1,
        "response_2": response_2,
        "score_2": score_2,
        "win_id": item["win_id"]
    })


with open("./base_score.json", "w") as f:
    json.dump(scored_data, f, ensure_ascii=False, indent=4)

print("All validation data has been scored and stores in val_score.json!")
