import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 设置 NPU 设备
device = torch.device("npu:0")

model_path = "/data/Qwen2.5-0.5B-Instruct"  # 本地模型路径

# 加载模型（禁用 FlashAttention，避免 TBE 问题）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager", 
).to(device)

# 加载 tokenizer，并确保 pad_token 设置
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 读取 prompts.json
with open("prompts.json", "r") as f:
    data = json.load(f)

prompts = []
responses = []

for count, entry in enumerate(data, start=1):
    print(f"Current processing sample {count}")
    prompt = entry["prompt"]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # 使用模板生成输入文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入，确保包含 attention_mask 且使用 padding
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,  # 避免超长输入
        return_attention_mask=True
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=False,  # 不使用采样，避免 TBE 中非法数值
            num_beams=1,      # beam search 设置为1，简化执行路径
        )

    # 去除输入部分，仅保留新生成内容
    new_tokens = output_ids[:, model_inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    prompts.append(prompt)
    responses.append(response)

# 保存结果
output_data = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)]
with open("qwen_gen.json", "w") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("Generated responses saved successfully!")
