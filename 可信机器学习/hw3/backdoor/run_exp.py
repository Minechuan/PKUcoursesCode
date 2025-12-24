#!/usr/bin/env python3
import subprocess, re, os
import numpy as np
import json
'''
python generate_mask.py --anp-alpha 0.8
python prune_network.py --mask-file=./mask_out/mask_values_0.5.txt


Because original eaasy always prune neurons by the threshold 0.2
So we find the great alpha 

then change threshold to find the best threshold
 "alpha = 0.2, layer3.1.bn1", "Neuron Idx": 153, "Mask": 0.25, "PoisonLoss": 3.6408, "PoisonACC": 0.033, "CleanLoss": 0.2727, "CleanACC": 0.9213}

'''

# def parse_output(output):
#     lines = output.strip().splitlines()
#     if not lines:
#         return None, None

#     last_line = lines[-1].strip()
#     fields = re.split(r'\s{2,}', last_line)

#     try:
#         ASR = float(fields[5])   # PoisonACC
#         ACC = float(fields[7])   # CleanACC
#         return ASR, ACC
#     except (IndexError, ValueError):
#         print("Error parsing ASR/ACC from last line:", last_line)
#         return None, None


def main():

#     best_alpha=0
#     best_score = -100
#     def eval_f(ASR,ACC):
#         return (0.05-ASR)+(ACC-0.92)
    

#     alpha_score = {}
#     # calculate mask with different: alpha
#     for alpha in np.arange(0.0, 1.0, 0.05):
#         alpha = round(alpha, 2)
        
#         # Step 1: generate mask
#         cmd = ["python", "generate_mask.py", f"--anp-alpha={alpha}"]
#         print("Running:", cmd)
#         result = subprocess.run(cmd, capture_output=True, text=True)
#         if result.returncode != 0:
#             print("generate_mask.py failed:", result.stderr)
#             continue

#         # Step 2: evaluate pruning
#         eval_cmd = [
#             "python", "prune_network.py",
#             f"--threshold={0.2}",
#             f"--mask-file=./mask_out/mask_values_{alpha}.txt"
#         ]
#         print("Evaluating:", eval_cmd)
#         result = subprocess.run(eval_cmd, capture_output=True, text=True)
#         if result.returncode != 0:
#             print("prune_network.py failed:", result.stderr)
#             continue

#         output = result.stdout
#         ASR, ACC = parse_output(output)
#         if ASR is None or ACC is None:
#             continue
#         score = eval_f(ASR, ACC)

#         if score > best_score:
#             best_alpha = alpha
#             best_score = score

#         alpha_score[alpha] = [ASR, ACC, score]

#     # Step 3: sort results and save
#     sorted_results = sorted(
#         [{"alpha": a, "ASR": v[0], "ACC": v[1], "score": v[2]} for a, v in alpha_score.items()],
#         key=lambda x: x["score"],
#         reverse=True
#     )

#     with open("stage1_result.jsonl", "w") as f:
#         for entry in sorted_results:
#             f.write(json.dumps(entry) + "\n")

#     print(f"\nBest alpha: {best_alpha}, Score: {best_score:.4f}")

    # Change threshold to check ASR and ACC
    best_alpha = 0.2
    for thr in np.arange(0.0, 1.0, 0.05):
        thr = round(thr, 2) 
        cmd = ["python", "prune_network.py", f"--threshold={thr}",f"--mask-file=./mask_out/mask_values_{best_alpha}.txt"]
        print("Running command:", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("prune_network.py failed:", result.stderr)
            continue

        output = result.stdout.strip().splitlines()

        # 寻找表头
        header_pattern = re.compile(r"No\.\s+Layer Name\s+Neuron Idx")
        header_line = None
        for i, line in enumerate(output):
            if header_pattern.search(line):
                header_line = i
                break
        if header_line is None:
            print("No table header found.")
            continue

        header_fields = re.split(r"\s{2,}", output[header_line].strip())
        last_data_line = output[-1].strip()
        values = re.split(r"\s{2,}", last_data_line)

        # 补全字段，如果列数不对则跳过
        if len(header_fields) != len(values):
            print("Mismatch between header and data columns.")
            continue

        entry = {k: v for k, v in zip(header_fields, values)}

        # 字段类型转换
        for key in entry:
            try:
                entry[key] = float(entry[key]) if '.' in entry[key] or 'e' in entry[key].lower() else int(entry[key])
            except:
                pass  # 保留字符串形式

        # 写入 JSONL
        with open(f"fin_result_{best_alpha}.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


# if __name__ == "__main__":
    # main()


thr = 0.7
for alpha in np.arange(0.0, 1.0, 0.05):
    alpha = round(alpha, 2) 
    cmd = ["python", "prune_network.py", f"--threshold={thr}",f"--mask-file=./mask_out/mask_values_{alpha}.txt"]
    print("Running command:", cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("prune_network.py failed:", result.stderr)
        continue

    output = result.stdout.strip().splitlines()

    # 寻找表头
    header_pattern = re.compile(r"No\.\s+Layer Name\s+Neuron Idx")
    header_line = None
    for i, line in enumerate(output):
        if header_pattern.search(line):
            header_line = i
            break
    if header_line is None:
        print("No table header found.")
        continue

    header_fields = re.split(r"\s{2,}", output[header_line].strip())
    last_data_line = output[-1].strip()
    values = re.split(r"\s{2,}", last_data_line)

    # 补全字段，如果列数不对则跳过
    if len(header_fields) != len(values):
        print("Mismatch between header and data columns.")
        continue

    entry = {k: v for k, v in zip(header_fields, values)}

    # 字段类型转换
    for key in entry:
        try:
            entry[key] = float(entry[key]) if '.' in entry[key] or 'e' in entry[key].lower() else int(entry[key])
        except:
            pass  # 保留字符串形式

    with open(f"./fin_2.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")