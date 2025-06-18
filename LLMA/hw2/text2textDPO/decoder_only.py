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
dataset_name="Dahoas/rm-static"
save_dir = "./qwen-ft"
max_length=256 # prompt 

batch_size = 8
learning_rate = 2e-5
num_epochs = 1
beta = 0.1  # DPO 温度参数
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 显存占用 50 G

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name).eval().to(device)
model.config.sliding_window = None  # 显式禁用
ref_model.config.sliding_window = None  # 显式禁用
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

from datasets import load_dataset


class PreferenceDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = load_dataset(dataset_name, split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # def extract_important_parts(prompt: str, max_char_len=800):
        #     """
        #     仅保留 prompt 的关键部分，比如：开头+结尾、特定关键词所在句子等
        #     """
        #     prompt = prompt.strip()
        #     # 示例策略：保留前500字符 + 后500字符
        #     front = prompt[:max_char_len // 2]
        #     back = prompt[-max_char_len // 2:]
        #     return front + "\n...\n" + back

        # processed_prompt = extract_important_parts(sample["prompt"])

        prompt_enc = self.tokenizer(
            sample["prompt"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        chosen_enc = self.tokenizer(
            sample["chosen"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            sample["rejected"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "prompt_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_mask": prompt_enc["attention_mask"].squeeze(0),
            "chosen_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_mask": rejected_enc["attention_mask"].squeeze(0),
        }

train_dataset = PreferenceDataset(tokenizer,split="train",max_length=max_length)
test_dataset = PreferenceDataset(tokenizer,split="test",max_length=max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 初始化 wandb
wandb.init(
    project="dpo-training",
    name="Qwen-2.5-text2text",
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

        pid = batch["prompt_ids"].to(device)      # [B, Lp]
        pm  = batch["prompt_mask"].to(device)
        cid = batch["chosen_ids"].to(device)      # [B, Lc]
        cm  = batch["chosen_mask"].to(device)
        rid = batch["rejected_ids"].to(device)    # [B, Lr]
        rm  = batch["rejected_mask"].to(device)

        # ===== chosen 拼接 =====
        in_w  = torch.cat([pid, cid], dim=1)                  # [B, Lp+Lc]
        att_w = torch.cat([pm, cm], dim=1)
        lbl_w = torch.cat([
            torch.full_like(pid, -100),  # prompt 部分不计 loss
            cid
        ], dim=1)

        out_w_theta = model(input_ids=in_w, attention_mask=att_w, labels=lbl_w, return_dict=True)
        token_w = (cid != tokenizer.pad_token_id).sum(dim=1).float().clamp(min=1.)
        logp_w_theta = -out_w_theta.loss * token_w

        with torch.no_grad():
            out_w_ref = ref_model(input_ids=in_w, attention_mask=att_w, labels=lbl_w, return_dict=True)
            logp_w_ref = -out_w_ref.loss * token_w

        # ===== rejected 拼接 =====
        in_l  = torch.cat([pid, rid], dim=1)
        att_l = torch.cat([pm, rm], dim=1)
        lbl_l = torch.cat([
            torch.full_like(pid, -100),
            rid
        ], dim=1)

        out_l_theta = model(input_ids=in_l, attention_mask=att_l, labels=lbl_l, return_dict=True)
        token_l = (rid != tokenizer.pad_token_id).sum(dim=1).float().clamp(min=1.)
        logp_l_theta = -out_l_theta.loss * token_l

        with torch.no_grad():
            out_l_ref = ref_model(input_ids=in_l, attention_mask=att_l, labels=lbl_l, return_dict=True)
            logp_l_ref = -out_l_ref.loss * token_l

        # ===== DPO loss + backward =====
        loss = dpo_loss(logp_w_theta, logp_w_ref, logp_l_theta, logp_l_ref, beta)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })

        wandb.log({"loss": loss.item()})



os.makedirs(save_dir, exist_ok=True)


wandb.save(os.path.join(save_dir, "pytorch_model.bin"))
wandb.finish()


model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


def evaluate(model, ref_model, dataloader, tokenizer, beta, device):
    model.eval()
    ref_model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # 1. 取出 prompt 与 chosen/rejected
            pid = batch["prompt_ids"].to(device)      # [B, Lp]
            pm  = batch["prompt_mask"].to(device)
            cid = batch["chosen_ids"].to(device)      # [B, Lc]
            cm  = batch["chosen_mask"].to(device)
            rid = batch["rejected_ids"].to(device)    # [B, Lr]
            rm  = batch["rejected_mask"].to(device)

            # —— chosen 拼接 & 计算 log-prob —— 
            in_w  = torch.cat([pid, cid], dim=1)       # [B, Lp+Lc]
            att_w = torch.cat([pm, cm], dim=1)
            lbl_w = torch.cat([
                torch.full_like(pid, -100),  # prompt 部分不计 loss
                cid
            ], dim=1)

            out_w_theta = model(input_ids=in_w, attention_mask=att_w, labels=lbl_w, return_dict=True)
            token_w = (cid != tokenizer.pad_token_id).sum(dim=1).float().clamp(min=1.)
            logp_w_theta = -out_w_theta.loss * token_w

            out_w_ref = ref_model(input_ids=in_w, attention_mask=att_w, labels=lbl_w, return_dict=True)
            logp_w_ref = -out_w_ref.loss * token_w

            # —— rejected 拼接 & 计算 log-prob —— 
            in_l  = torch.cat([pid, rid], dim=1)
            att_l = torch.cat([pm, rm], dim=1)
            lbl_l = torch.cat([
                torch.full_like(pid, -100),
                rid
            ], dim=1)

            out_l_theta = model(input_ids=in_l, attention_mask=att_l, labels=lbl_l, return_dict=True)
            token_l = (rid != tokenizer.pad_token_id).sum(dim=1).float().clamp(min=1.)
            logp_l_theta = -out_l_theta.loss * token_l

            out_l_ref = ref_model(input_ids=in_l, attention_mask=att_l, labels=lbl_l, return_dict=True)
            logp_l_ref = -out_l_ref.loss * token_l

            # —— DPO Loss 累计 —— 
            loss = dpo_loss(logp_w_theta, logp_w_ref, logp_l_theta, logp_l_ref, beta)
            batch_size = pid.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    print(f"Evaluation DPO Loss: {avg_loss:.4f}")
    return avg_loss

# 调用
eval_loss = evaluate(model, ref_model, test_dataloader, tokenizer, beta, device)


prompt = "Human: Where would the best place to drive over the speed limit be?\n\nAssistant:"
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

