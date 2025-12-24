import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化NPU环境
torch.npu.set_compile_mode(jit_compile=True)  # 启用NPU图编译
device = torch.device("npu:0")

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "/data/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
).to(device)
model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(
    "/data/Qwen2.5-0.5B-Instruct",
    padding_side="left"
)

# 批量处理配置
BATCH_SIZE = 8
MAX_LENGTH = 512

# 读取数据
with open("prompts.json", "r") as f:
    data = json.load(f)

all_messages = []
for entry in data:
    all_messages.append([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": entry['prompt']}
    ])

# 批量生成函数
def generate_batch(messages_batch):
    texts = [tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    ) for msg in messages_batch]
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True
    ).to(device)
    
    # NPU专用混合精度上下文
    with torch.npu.amp.autocast(enabled=True, dtype=torch.float16), torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    responses = []
    for i in range(len(outputs)):
        output = outputs[i][len(inputs["input_ids"][i]):]
        responses.append(tokenizer.decode(output, skip_special_tokens=True))
    
    return responses

# 分批次处理
results = []
for i in range(0, len(all_messages), BATCH_SIZE):
    batch = all_messages[i:i+BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(all_messages)-1)//BATCH_SIZE + 1}")
    results.extend(generate_batch(batch))

# 保存结果
output_data = [{"prompt": data[i]["prompt"], "response": results[i]} 
              for i in range(len(data))]
with open("responses_qwen.json", "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Generated {len(results)} responses successfully.")