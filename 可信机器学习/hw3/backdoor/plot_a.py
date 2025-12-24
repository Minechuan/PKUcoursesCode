import json
import matplotlib.pyplot as plt

# 读取 jsonl 文件
threshold, ASRs, ACCs, scores = [], [], [], []

with open('./fin_result_0.2.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        threshold.append(data["Mask"])
        ASRs.append(data["PoisonACC"])
        ACCs.append(data["CleanACC"])
        # scores.append(data["score"])

# 按 alpha 升序排序（可选但推荐）
sorted_data = sorted(zip(threshold, ASRs, ACCs))
threshold, ASRs, ACCs = zip(*sorted_data)

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(threshold, ASRs, marker='o', label='ASR', color='red')
plt.plot(threshold, ACCs, marker='s', label='ACC', color='green')
# plt.plot(alphas, scores, marker='^', label='Score', color='blue')


plt.xlabel('threshold')
plt.ylabel('Value')
plt.title('ASR, ACC vs threshold')
plt.grid(True)
plt.legend()
plt.tight_layout()


plt.show()
