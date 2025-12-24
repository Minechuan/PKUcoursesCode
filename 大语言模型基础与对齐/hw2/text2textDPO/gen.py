import os
# 使用 hf-mirror 下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer

from tqdm import tqdm


device=torch.device("npu:0")
print("my device is:",device)

model_name = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 显存占用 50 G

model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
# ref_model = T5ForConditionalGeneration.from_pretrained(ref_model_name).eval().to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


model.eval()
prompt = "I like this moive because "
inputs = tokenizer(prompt, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    # ref_output_ids = ref_model.generate(**inputs, max_new_tokens=100, do_sample=False)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# ref_output_text = tokenizer.decode(ref_output_ids[0], skip_special_tokens=True)

print("=== Model Output ===")
print(output_text)
