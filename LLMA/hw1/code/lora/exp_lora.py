import torch
import argparse
import transformers
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from IPython.display import display
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


parser = argparse.ArgumentParser(description='get r and a')

parser.add_argument('--r', default=4, type=int, metavar='r')

parser.add_argument('--a', default=1, type=int, metavar='alpha')

args = parser.parse_args()



working_dir = './'
output_directory = os.path.join(working_dir, "peft_lab_outputs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

####################################################################
#       Change the Code here: hyperparameter & test strings  
###################################################################
'''base model'''
model_name = "bigscience/bloomz-560m"
# model_name="bigscience/bloom-1b1"

'''data set'''
dataset_name = "tatsu-lab/alpaca" # num_rows: 52002
#dataset_name = "noob123/imdb_review_3000" # IMDB电影评论数据集, 保留50条样本用于训练 num_rows: 2999

'''Lora hyperparameters: W' = W + alpha/r*delta W '''
# lora_r=args.r
# lora_a=args.a
lora_r=16
lora_a=1
lora_target_modules=["dense"]
# train_lr=3e-2
train_lr=3e-2
train_epoch_num=2
MAX_LENGTH = 512
MAX_NEW_TOKEN=50
data_sample_num=1024 # 2^10
lora_dropout=0.05
# Step nums=(epoch_num * data_sample_num) /  Batch_size
print(f"Training LoRA model: r={lora_r}, a={lora_a}...")
'''Generation test (3 different kinds)'''
# 1. without instruction: movie
gen_str="I love this movie because"

# 2. Without instruction: not movie
# gen_str = "I like studying mathematics because"

# 3. With instruction
instr = (
    "Read the movie review provided below and summarize the author's opinion in one sentence.\n\n"
    "### Input:\n"
    "The movie had a stunning visual style and a compelling performance from the lead actor, "
    "but the plot felt disjointed and the pacing was slow.\n\n"
    "### Response:\n"
)
prompt = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately completes the request.\n\n"
    f"### Instruction:\n{instr}"
)
instr_str=prompt


'''Address and path'''
sub_addr = f"{'alpaca' if dataset_name=='tatsu-lab/alpaca' else 'ori'}"
sub_addr = os.path.join(sub_addr, f"model_r={lora_r}_a={lora_a}")
tensor_load_dir="./logs/lora_"+sub_addr # store tensorboard records
model_weights_path=f"lora_"+sub_addr # store weights of the lora model
###################################################################
#
#################################################################



tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
foundation_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)



def get_outputs(model, inputs, max_new_tokens=MAX_NEW_TOKEN):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5, # 避免模型复读，默认值为1.0
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs
def one_test(model_path):
    print(f"Loading lora model from {model_path}")

    loaded_model = PeftModel.from_pretrained(foundation_model, model_path, is_trainable=False).to(device)
    ''' test the generation ability for no instruction prompt and instruction prompt'''
    print("Without an instruction")
    input_sentences = tokenizer(gen_str, return_tensors="pt")
    gen_output = get_outputs(loaded_model, input_sentences, max_new_tokens=MAX_NEW_TOKEN)
    print(tokenizer.batch_decode(gen_output, skip_special_tokens=True))

    print("\nWith an instruction")
    input_sentences_instr = tokenizer(instr_str, return_tensors="pt")
    instr_output = get_outputs(loaded_model, input_sentences_instr, max_new_tokens=MAX_NEW_TOKEN)
    print(tokenizer.batch_decode(instr_output, skip_special_tokens=True))

    del loaded_model
    torch.cuda.empty_cache()



input_sentences = tokenizer(gen_str, return_tensors="pt")
foundational_outputs_sentence = get_outputs(foundation_model, input_sentences, max_new_tokens=MAX_NEW_TOKEN)
print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True))


def load_process_ft_data(dataset_name):
    if dataset_name=="tatsu-lab/alpaca":
        '''Dataset Alpaca'''
        raw_data = load_dataset(dataset_name)

        # —— 1. 构造 prompt 的函数 —— 
        def build_prompt(instr: str, inp: str) -> str:
            instr = instr.strip()
            inp = inp.strip()
            prompt = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instr}"
            )
            if inp:
                prompt += f"\n\n### Input:\n{inp}"
            prompt += "\n\n### Response:\n"
            return prompt

        # —— 2. batched-tokenize + 拼接 + mask labels —— 
        def tokenize_and_mask(batch):
            instrs = batch["instruction"]
            inps   = batch.get("input", [""] * len(instrs))
            outs   = batch["output"]

            prompts = [build_prompt(i, j) for i, j in zip(instrs, inps)]

            input_ids_batch = []
            labels_batch = []

            for p, o in zip(prompts, outs):
                p_ids = tokenizer(p, truncation=True, max_length=512).input_ids
                o_ids = tokenizer(o.strip(), truncation=True, max_length=512).input_ids

                # 拼接
                input_ids = p_ids + o_ids
                labels    = [-100] * len(p_ids) + o_ids

                # 截断到 max_length
                input_ids = input_ids[:MAX_LENGTH]
                labels    = labels[:MAX_LENGTH]

                input_ids_batch.append(input_ids)
                labels_batch.append(labels)

                input_ids_tensor = [torch.tensor(x, dtype=torch.long) for x in input_ids_batch]
                labels_tensor    = [torch.tensor(x, dtype=torch.long) for x in labels_batch]

            return {"input_ids": input_ids_tensor, "labels": labels_tensor}



        # —— 3. 构造 train_dataset —— 
        train_dataset = raw_data["train"].select(range(data_sample_num)).map(
            tokenize_and_mask,
            batched=True,
            remove_columns=["instruction", "input", "output"]
        )
    elif dataset_name == "noob123/imdb_review_3000":
        dataset = "noob123/imdb_review_3000"

        #Create the Dataset to create prompts.
        data = load_dataset(dataset)
        data = data.map(lambda samples: tokenizer(samples['review']), batched=True)
        train_sample = data["train"].select(range(data_sample_num))

        train_dataset = train_sample.remove_columns('sentiment')
    else:
        Exception("Unknow Dataset!")
    
    print(f"Fine-Tuning base model with dataset {dataset_name}, use {data_sample_num} lines, the structure of the partial dataset:")
    display(train_dataset)
    return train_dataset

train_sample=load_process_ft_data(dataset_name)
# print(train_sample[:1])

lora_config = LoraConfig(
    r=lora_r, #As bigger the R bigger the parameters to train.
    lora_alpha=lora_a, # a scaling factor that adjusts the magnitude of the weight matrix. Usually set to 1
    target_modules=lora_target_modules, #You can obtain a list of target modules in the URL above.
    lora_dropout=lora_dropout, #Helps to avoid Overfitting.
    bias="lora_only", # this specifies if the bias parameter should be trained.
    task_type="CAUSAL_LM"
)

# for name, module in foundation_model.named_modules():
#     print(name)
peft_model = get_peft_model(foundation_model, lora_config)
print(peft_model.print_trainable_parameters())



# My preference batch size is 16
training_args = TrainingArguments(
    output_dir=output_directory,
    logging_dir=tensor_load_dir,  # 新增日志目录
    logging_steps=8,      # 每10步记录一次
    report_to="tensorboard", # 使用TensorBoard
    auto_find_batch_size=False,      # 关掉
    per_device_train_batch_size=1,   # 或者 2
    gradient_accumulation_steps=8,  # 保证等效大 batch
    # fp16=True,                       # 混合精度，节省一半显存
    # save_steps=100, # 每训练 100个 step（batch） 就保存一次模型检查点
    # save_total_limit=1,
    # auto_find_batch_size=True, # Find a correct bvatch size that fits the size of Data.
    learning_rate=train_lr, # Higher learning rate than full fine-tuning.
    num_train_epochs=train_epoch_num
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_sample,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
print("Auto batch size is: ",trainer.args.per_device_train_batch_size)
trainer.train()

peft_model_path = os.path.join(output_directory, model_weights_path)
trainer.model.save_pretrained(peft_model_path)
print(f"Store model in the dir: {peft_model_path}!\n")

del trainer.model
torch.cuda.empty_cache()

# test sample output
one_test(peft_model_path)