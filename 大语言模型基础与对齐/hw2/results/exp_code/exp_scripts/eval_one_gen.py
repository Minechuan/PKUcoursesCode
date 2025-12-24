from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("npu:0")  # 明确指定昇腾 NPU

model = AutoModelForCausalLM.from_pretrained(
    "/data/align-anything/outputs/part2DPO/DPO/slice_end",
    torch_dtype=torch.float16  # 昇腾推荐使用 float16
).to(device)  # 手动迁移到 NPU

tokenizer = AutoTokenizer.from_pretrained("/data/Qwen2.5-0.5B-Instruct")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt")
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}  # 数据迁移到 NPU

with torch.no_grad():
    generated_ids = model.generate(
        model_inputs["input_ids"],
        max_new_tokens=512
    )

# 去除 prompt 的部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
