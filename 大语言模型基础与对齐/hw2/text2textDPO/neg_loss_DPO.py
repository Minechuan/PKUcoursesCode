import os
# 使用 hf-mirror 下载模型
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
import wandb
wandb.login(key="My key")



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("npu:0")
print("my device is:",device)



model_name = "Qwen/Qwen2.5-0.5B-Instruct"
ref_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 参考策略
# dataset_name="Dahoas/rm-static"
dataset_name="Intel/orca_dpo_pairs" # 采用单轮对话
save_dir = "./qwen-ft"
max_length=512

batch_size = 8
learning_rate = 8e-5
num_epochs = 1
beta = 0.01  # DPO 温度参数


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device) # 显存占用 50 G
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name).eval().to(device)
model.config.sliding_window = None  # 显式禁用
ref_model.config.sliding_window = None  # 显式禁用
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

from datasets import load_dataset


class PreferenceDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = load_dataset(dataset_name, split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
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
        text_tokens = self.tokenizer(text, add_special_tokens=False)
        text_len = len(text_tokens["input_ids"])

        # 拼接 better
        better_full = text + better_response
        better_enc = self.tokenizer(
            better_full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # 使用了 apply_chat_template，它会自动在结尾加上 <|assistant|>\n
        # 拼接 worse
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
            # "text_ids": text_tokens["input_ids"].squeeze(0),
            "text_len": text_len  
        }
train_dataset = PreferenceDataset(tokenizer,split="train",max_length=max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



# 初始化 wandb
wandb.init(
    project="dpo-training",
    name="run_exp",
    config={
        "model_name": model_name,
        "ref_model_name": ref_model_name,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "beta": beta,
        "max_length": max_length, # in dataset prompt 256+ response 256
    }
)


def dpo_loss(logp_theta_w, logp_ref_w, logp_theta_l, logp_ref_l, beta):
    
    diff = beta * ((logp_theta_w - logp_ref_w) - (logp_theta_l - logp_ref_l))
    # 负对数 sigmoid
    loss = -F.logsigmoid(diff).mean()
    return loss

model.train()
for epoch in range(num_epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    total_loss = 0.0
    total_batches = len(train_dataloader)
    for batch_idx, batch in enumerate(progress_bar):
        

        better_ids   = batch["input_ids_better"].to(device)      # [B, Lp]
        better_att  = batch["attention_mask_better"].to(device)
        worse_ids    = batch["input_ids_worse"].to(device)      # [B, Lc]
        worse_att   = batch["attention_mask_worse"].to(device)
        # prompt_ids   = batch["text_ids"].to(device)    # [B, Lr]
        prompt_len   = batch["text_len"].to(device)

        # make labels
        B, L = better_ids.shape
        w_labels = better_ids.clone()
        token_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        w_labels[token_pos < prompt_len.unsqueeze(1)] = -100
        l_labels = worse_ids.clone()
        l_labels[token_pos < prompt_len.unsqueeze(1)] = -100


        # label_mask = (w_labels != -100)
        # effective_tokens = label_mask.sum()
        # total_tokens = torch.numel(w_labels)
        # print("Effective token ratio:", effective_tokens.item() / total_tokens)
        # === Get better log-probs ===
        out_better = model(
            input_ids=better_ids,
            attention_mask=better_att,
            labels= w_labels,
            return_dict=True
        )
        # token_w = (token_pos >= prompt_len.unsqueeze(1)).sum(dim=1).float().clamp(min=1.)
        # average loss for every token
        logp_w_theta = -out_better.loss

        with torch.no_grad():
            out_better_ref = ref_model(
                input_ids=better_ids,
                attention_mask=better_att,
                labels= w_labels,
                return_dict=True
            )
            logp_w_ref = -out_better_ref.loss

        # === Get better log-probs ===
        out_worse = model(
            input_ids=worse_ids,
            attention_mask=worse_att,
            labels= l_labels,
            return_dict=True
        )
        # token_l = (token_pos >= prompt_len.unsqueeze(1)).sum(dim=1).float().clamp(min=1.)
        logp_l_theta = -out_worse.loss

        with torch.no_grad():
            out_worse_ref = ref_model(
                input_ids=worse_ids,
                attention_mask=worse_att,
                labels= l_labels,
                return_dict=True
            )
            logp_l_ref = -out_worse_ref.loss

        # ===== DPO loss + backward =====

        loss = dpo_loss(logp_w_theta, logp_w_ref, logp_l_theta, logp_l_ref, beta)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

        if batch_idx % 10 == 0:
            wandb.log({"loss": loss.item(), "avg_loss": avg_loss})



os.makedirs(save_dir, exist_ok=True)


wandb.save(os.path.join(save_dir, "pytorch_model.bin"))
wandb.finish()


model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


# Generation Evaluation
system_msg = "You are an AI assistant that helps people find information."
user_msg   = "Where would the best place to drive over the speed limit be?"

messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    ref_output_ids = ref_model.generate(**inputs, max_new_tokens=100, do_sample=False)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
ref_output_text = tokenizer.decode(ref_output_ids[0], skip_special_tokens=True)

print("=== Model Output ===")
print(output_text)
print("\n=== Reference Model Output ===")
print(ref_output_text)

