import os
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

device = torch.device("npu:0")
print("my device is:", device)

# 配置参数
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
ref_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_name = "Intel/orca_dpo_pairs"
save_dir = "./My_sec/2"
max_length = 512

grad_accum_steps = 4  # 模拟 batchsize=32
batch_size = 8
learning_rate = 1e-5
num_epochs = 1
beta = 0.1

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 添加
tokenizer.padding_side = "left"            # 左填充（适合decoder）
tokenizer.truncation_side = "left"


model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name).eval().to(device)
# model.config.sliding_window = None
# ref_model.config.sliding_window = None
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 冻结参考模型
for param in ref_model.parameters():
    param.requires_grad = False

# 数据集类
class PreferenceDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset(dataset_name, split=split)
        
        # 预处理：过滤掉text_len > max_length的样本
        self.valid_indices = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            messages = [
                {"role": "system", "content": sample["system"]},
                {"role": "user", "content": sample["question"]}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            text_tokens = self.tokenizer(text, add_special_tokens=False)
            if len(text_tokens["input_ids"]) <= max_length/2:
                self.valid_indices.append(idx)

            
                
        print(f"Max length is: {max_length}. Filtered dataset: {len(self.valid_indices)}/{len(self.dataset)} samples remaining")

    def __len__(self):
        return len(self.valid_indices)  # 返回过滤后的长度

    def __getitem__(self, idx):
        # 使用过滤后的索引获取原始数据
        original_idx = self.valid_indices[idx]
        sample = self.dataset[original_idx]
        
        system_msg = sample["system"]
        user_msg = sample["question"]
        better_response = sample["chosen"]
        worse_response = sample["rejected"]

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 动态计算prompt长度（已确保<=max_length）
        text_tokens = self.tokenizer(text, add_special_tokens=False)
        prompt_len = len(text_tokens["input_ids"])

        # 处理better和worse样本
        better_full = text + better_response
        better_enc = self.tokenizer(
            better_full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        worse_full = text + worse_response
        worse_enc = self.tokenizer(
            worse_full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids_better": better_enc["input_ids"].squeeze(0),
            "attention_mask_better": better_enc["attention_mask"].squeeze(0),
            "input_ids_worse": worse_enc["input_ids"].squeeze(0),
            "attention_mask_worse": worse_enc["attention_mask"].squeeze(0),
            "text_len": prompt_len
        }

# 加载数据集
train_dataset = PreferenceDataset(tokenizer, split="train", max_length=max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# DPO损失函数
def dpo_loss(logp_theta_w, logp_ref_w, logp_theta_l, logp_ref_l, beta):
    log_ratio_w = logp_theta_w - logp_ref_w
    log_ratio_l = logp_theta_l - logp_ref_l
    
    # 限制差异范围（防止数值不稳定）
    log_ratio_w = torch.clamp(log_ratio_w, -10, 10)
    log_ratio_l = torch.clamp(log_ratio_l, -10, 10)
    
    diff = beta * (log_ratio_w - log_ratio_l)
    diff = torch.clamp(diff, -20, 20)  # 进一步限制
    
    loss = -F.logsigmoid(diff).mean()
    return loss

# 计算log概率
def get_log_probs(model, input_ids, attention_mask, labels, is_ref_model=False):
    if is_ref_model:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
    else:
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100)
    return (selected_log_probs * mask).sum(dim=1)

# 测试生成函数
def generate_sample(model, tokenizer, device, prompt_text=None):
    if prompt_text is None:
        system_msg = "You are an AI assistant that helps people find information."
        user_msg = "Where would the best place to drive over the speed limit be?"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return output_text

# 训练循环
best_loss = float('inf')
os.makedirs(save_dir, exist_ok=True)

def get_log_probs(model, input_ids, attention_mask, labels, is_ref_model=False):
    # 如果是参考模型，强制禁用梯度
    if is_ref_model:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
    else:
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100)
    return (selected_log_probs * mask).sum(dim=1)


model.train()
for epoch in range(num_epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(progress_bar):
        # 准备数据
        better_ids = batch["input_ids_better"].to(device)
        better_att = batch["attention_mask_better"].to(device)
        worse_ids = batch["input_ids_worse"].to(device)
        worse_att = batch["attention_mask_worse"].to(device)
        prompt_len = batch["text_len"].to(device)

        # 创建标签掩码
        B, L = better_ids.shape
        w_labels = better_ids.clone()
        l_labels = worse_ids.clone()
        token_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        w_labels[token_pos < prompt_len.unsqueeze(1)] = -100
        l_labels[token_pos < prompt_len.unsqueeze(1)] = -100

        logp_w_theta = get_log_probs(model, better_ids, better_att, w_labels, is_ref_model=False)
        logp_l_theta = get_log_probs(model, worse_ids, worse_att, l_labels, is_ref_model=False)

        # 计算参考模型的 log-probs（禁用梯度）
        logp_w_ref = get_log_probs(ref_model, better_ids, better_att, w_labels, is_ref_model=True)
        logp_l_ref = get_log_probs(ref_model, worse_ids, worse_att, l_labels, is_ref_model=True)

        # 计算损失并更新
        loss = dpo_loss(logp_w_theta, logp_w_ref, logp_l_theta, logp_l_ref, beta)
        optimizer.zero_grad()
        loss.backward()


        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            # 'logp_w_theta':f'{logp_w_theta:.4f}',
            # 'logp_l_theta':f'{logp_l_theta:.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

        # 每100个step测试生成并保存最佳模型
        if batch_idx % 100 == 0:
            print(f"\nStep {batch_idx} Generation Test:")
            generated_text = generate_sample(model, tokenizer, device)
            print("Generated:", generated_text)

            current_loss = avg_loss
            if current_loss < best_loss:
                best_loss = current_loss
                model.save_pretrained(os.path.join(save_dir, "best_model"))
                tokenizer.save_pretrained(os.path.join(save_dir, "best_model"))
                print(f"New best model saved with loss: {best_loss:.4f}")

# 保存最终模型
model.save_pretrained(os.path.join(save_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(save_dir, "final_model"))

# 最终测试生成
print("\nFinal Generation Test:")
final_output = generate_sample(model, tokenizer, device)
print("Final Output:", final_output)