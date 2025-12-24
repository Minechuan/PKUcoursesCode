import torch
import argparse
import transformers
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import os


# —— 2. 全局配置 —— 
working_dir       = './'
output_directory  = os.path.join(working_dir, "peft_lab_outputs")
device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_NAME   = "bigscience/bloomz-560m"
LORA_MODEL_PATH   = ""
# LORA_MODEL_PATH   = "./peft_lab_outputs/checkpoint-16"
MAX_LENGTH        = 512
MAX_NEW_TOKENS    = 50
DATA_TEST_NUM   = 512

print(f"Using device: {device}")


# —— 3. 加载基础模型与 LoRA adapter —— 
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
if LORA_MODEL_PATH:
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH, is_trainable=False).to(device)

# —— 4. 生成函数 —— 
def get_outputs_batch(model, prompts, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# —— 5. Prompt 列表 —— 
def get_inputs(cate):
    if cate == 1:
        return [
            "I love this movie because",
            "A slow-burn thriller that",
            "The plot was boring, but"
        ]
    elif cate == 2:
        return [
            "Here are three tips for a good meal: 1 fresh ingredients. 2",
            "Write a poem about",
            "Pretend you are an alien visiting Earth. You would"
        ]
    elif cate == 3:
        common = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately completes the request.\n\n"
        )
        return [
            common +
            "### Instruction:\nRead the movie review provided below and summarize the author's opinion in one sentence.\n\n"
            "### Input:\nThe movie had a stunning visual style and a compelling performance from the lead actor, "
            "but the plot felt disjointed and the pacing was slow.\n\n### Response:\n",
            common +
            "### Instruction:\nWrite a comprehensive critique of the following movie review. Your response should include: "
            "an analysis of the review's strengths and weaknesses, a discussion of the reviewer's perspective, "
            "and a reflection on whether you agree or disagree with their assessment and why.\n\n"
            "### Input:\nWhile the movie attempts to tackle complex philosophical themes, its heavy-handed symbolism and sluggish pacing "
            "often distract from the emotional impact. The lead actress delivers a powerful performance, but the dialogue "
            "sometimes feels unnatural, and the final twist is more confusing than clever.\n\n### Response:\n",
            common +
            "### Instruction:\nGiven the list of tasks below, generate a prioritized to-do list for someone working from home.\n\n"
            "### Input:\n- Reply to emails\n- Complete project report\n- Attend team meeting at 3 PM\n- Take a 30-minute walk\n- Review budget for Q2\n\n### Response:\n",
            common +
            "### Instruction:\nTranslate the following English sentence into French.\n\n"
            "### Input:\nThe sun is shining and the birds are singing.\n\n### Response:\n"
        ]

# —— 6. 运行并打印生成结果 —— 
print("\nGenerating outputs (batched):")
count = 0
for i in range(3):
    prompts = get_inputs(i + 1)
    generated_list = get_outputs_batch(model, prompts)
    for prompt, gen in zip(prompts, generated_list):
        count += 1
        print(f"\n--- Example {count} ---")
        print(f"Prompt: {prompt}")
        print(f"Generation: {gen[len(prompt):]}")
print("-" * 80)

# —— 7. 定义预处理和评估函数 —— 

# Alpaca preprocess（Instruction fine-tuning 形式）
def build_alpaca_prompt(instr, inp):
    instr = instr.strip()
    inp   = inp.strip()
    prompt = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instr}"
    )
    if inp:
        prompt += f"\n\n### Input:\n{inp}"
    prompt += "\n\n### Response:\n"
    return prompt

def preprocess_alpaca(batch):
    prompts = [build_alpaca_prompt(ins, inp) for ins, inp in zip(batch["instruction"], batch.get("input", [""]*len(batch["instruction"])))]
    outs    = [o.strip() for o in batch["output"]]
    enc     = tokenizer(prompts, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    lab     = tokenizer(outs,    truncation=True, padding="max_length", max_length=MAX_LENGTH)
    # mask prompt tokens
    labels = []
    for i, p in enumerate(prompts):
        prompt_len = len(tokenizer(p, truncation=True, max_length=MAX_LENGTH).input_ids)
        seq_labels = [-100]*prompt_len + lab["input_ids"][i][prompt_len:]
        seq_labels = seq_labels[:MAX_LENGTH]
        # pad to length
        seq_labels += [-100] * (MAX_LENGTH - len(seq_labels))
        labels.append(seq_labels)
    enc["labels"] = labels
    return enc

# IMDB preprocess（自回归形式）
def preprocess_imdb(batch):
    enc = tokenizer(batch["review"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    # labels = input_ids
    enc["labels"] = enc["input_ids"].copy()
    return enc

def evaluate_loss(dataset_name, dataset, preprocess_fn, split="train"):
    # 取 512 条样本（Alpaca 用 train，IMDB 用 test）
    subset = dataset[split].select(range(DATA_TEST_NUM))
    # 预处理
    tokenized = subset.map(preprocess_fn, batched=True, remove_columns=subset.column_names)
    # DataLoader
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader   = DataLoader(tokenized, batch_size=8, collate_fn=collator)
    # 评估
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # batch["labels"] 中 -100 positions are ignored in loss_fn
            bsz = batch["labels"].shape[0]
            total_loss += outputs.loss.item() * bsz
            total_tokens += bsz
    avg_loss = total_loss / total_tokens
    print(f"[{dataset_name}] avg eval loss over 512 samples: {avg_loss:.4f}")

# —— 8. 加载数据集并评估 —— 
print("\nEvaluating model loss on 512 samples each from Alpaca and IMDB...")
alpaca_raw = load_dataset("tatsu-lab/alpaca")
evaluate_loss("Alpaca", alpaca_raw, preprocess_alpaca, split="train")
imdb_raw   = load_dataset("noob123/imdb_review_3000")
evaluate_loss("IMDB",   imdb_raw,   preprocess_imdb, split="train")
print("Evaluation complete.")
