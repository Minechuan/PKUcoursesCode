'''Choose T5 as base model: design train data for Encoder-Decoder Architecture'''



import os
# 使用 hf-mirror 下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer

from tqdm import tqdm

import wandb
wandb.login(key="My key")



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device=torch.device("npu:0")
print("my device is:",device)

model_name = "t5-small"
ref_model_name = "t5-small"  # 参考策略
batch_size = 16
learning_rate = 5e-5
num_epochs = 1
beta = 0.1  # DPO 温度参数
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 显存占用 50 G

model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
ref_model = T5ForConditionalGeneration.from_pretrained(ref_model_name).eval().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

from datasets import load_dataset


class PreferenceDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = load_dataset("Dahoas/rm-static", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prompt = sample["prompt"]


        input_enc = self.tokenizer(
            prompt,
            # 75% 的 prompt<1024
            max_length=self.max_length*2,
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
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "chosen_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }

train_dataset = PreferenceDataset(tokenizer,split="train")
test_dataset = PreferenceDataset(tokenizer,split="test")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 初始化 wandb
wandb.init(
    project="t5-dpo-training",
    name="dpo-t5-small",
    config={
        "model_name": model_name,
        "ref_model_name": ref_model_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "beta": beta,
        "max_length": 512, # in dataset prompt i
    }
)


def dpo_loss(logp_theta_w, logp_ref_w, logp_theta_l, logp_ref_l, beta):
    
    diff = beta * ((logp_theta_w - logp_ref_w) - (logp_theta_l - logp_ref_l))
    # 负对数 sigmoid
    loss = -F.logsigmoid(diff).mean()
    return loss

model.train()

for epoch in range(num_epochs):

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
    total_loss = 0.0
    total_batches = len(train_dataloader)

    for batch_idx, batch in enumerate(progress_bar):
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels_w = batch["chosen_ids"].to(device)
        labels_w[labels_w == tokenizer.pad_token_id] = -100
        out_theta_w = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels_w, return_dict=True)
        
        # loss 是平均每个 token 的负对数似然, 这里要还原总的 loss
        token_count_w = torch.clamp((labels_w != -100).sum(dim=1).float(), min=1.)
        logp_theta_w = -out_theta_w.loss * token_count_w


        '''reject response'''
        labels_l = batch["rejected_ids"].to(device)
        labels_l[labels_l == tokenizer.pad_token_id] = -100
        out_theta_l = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels_l, return_dict=True)

        token_count_l = torch.clamp((labels_l != -100).sum(dim=1).float(), min=1.)
        logp_theta_l = -out_theta_l.loss * token_count_l
        # compute ref log-prob
        with torch.no_grad():
            out_ref_w = ref_model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels_w, return_dict=True)
            logp_ref_w = -out_ref_w.loss * token_count_w

            out_ref_l = ref_model(input_ids=input_ids, attention_mask=attention_mask,
                                labels=labels_l, return_dict=True)
            logp_ref_l = -out_ref_l.loss * token_count_l


        loss = dpo_loss(logp_theta_w, logp_ref_w, logp_theta_l, logp_ref_l, beta)

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

        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
            "logp_theta_w": logp_theta_w.mean().item(),
            "logp_ref_w": logp_ref_w.mean().item(),
            "logp_theta_l": logp_theta_l.mean().item(),
            "logp_ref_l": logp_ref_l.mean().item(),
        })


save_dir = "./dpo_model"
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Chosen
            labels_chosen = batch["chosen_ids"].to(device)
            labels_chosen[labels_chosen == tokenizer.pad_token_id] = -100
            token_count_chosen = (labels_chosen != -100).sum(dim=1).float()

            out_theta_chosen = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_chosen, return_dict=True)
            out_ref_chosen = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_chosen, return_dict=True)

            logp_theta_chosen = -out_theta_chosen.loss * token_count_chosen
            logp_ref_chosen = -out_ref_chosen.loss * token_count_chosen

            # Rejected
            labels_rejected = batch["rejected_ids"].to(device)
            labels_rejected[labels_rejected == tokenizer.pad_token_id] = -100
            token_count_rejected = (labels_rejected != -100).sum(dim=1).float()

            out_theta_rejected = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_rejected, return_dict=True)
            out_ref_rejected = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_rejected, return_dict=True)

            logp_theta_rejected = -out_theta_rejected.loss * token_count_rejected
            logp_ref_rejected = -out_ref_rejected.loss * token_count_rejected

            # DPO Loss
            loss = dpo_loss(logp_theta_chosen, logp_ref_chosen, logp_theta_rejected, logp_ref_rejected, beta)
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    
    
    print(f"Evaluation DPO Loss: {avg_loss:.4f}")
    return avg_loss


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
