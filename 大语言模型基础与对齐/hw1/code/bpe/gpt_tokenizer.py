from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
'''
with open("encode_task_en.txt", "r", encoding="utf-8") as f:
    text = f.read()
'''
encoded1 = tokenizer.encode("OF", add_special_tokens=False) 
encoded2 = tokenizer.encode("IN", add_special_tokens=False) 
encoded3 = tokenizer.encode("专家", add_special_tokens=False) 
print(encoded1,encoded2,encoded3)
# decoded = tokenizer.decode(encoded)  
'''
with open("gpt_tokenizer_EN.txt", "w", encoding="utf-8") as f:
    f.write("Token IDs:\n")
    f.write(str(encoded) + "\n\n")
    f.write("Decoded Text:\n")
    f.write(decoded)
'''