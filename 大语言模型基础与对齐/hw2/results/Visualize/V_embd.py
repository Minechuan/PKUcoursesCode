import json
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import numpy as np


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 1. 加载数据
with open('val_score.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

chosen_texts = []
rejected_texts = []

for item in data:
    if item['win_id'] == 1:
        chosen_texts.append(item['response_1'])
        rejected_texts.append(item['response_2'])
    else:
        chosen_texts.append(item['response_2'])
        rejected_texts.append(item['response_1'])

# 2. 可视化响应长度分布
chosen_lengths = [len(text.split()) for text in chosen_texts]
rejected_lengths = [len(text.split()) for text in rejected_texts]

plt.figure()
plt.hist(chosen_lengths, bins=30, alpha=0.5, label='Chosen', density=True)
plt.hist(rejected_lengths, bins=30, alpha=0.5, label='Rejected', density=True)
plt.xlabel('Number of words')
plt.ylabel('Density')
plt.title('Length Distribution of Responses')
plt.legend()
plt.grid(True)
plt.show()

# 3. 词云可视化
def generate_wordcloud(texts, title):
    text_combined = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# generate_wordcloud(chosen_texts, 'Word Cloud: Chosen Responses')
# generate_wordcloud(rejected_texts, 'Word Cloud: Rejected Responses')

# 4. 嵌入 + t-SNE 降维
model = SentenceTransformer('all-MiniLM-L6-v2')  # 或者其他轻量向量模型
texts_all = chosen_texts + rejected_texts
embeddings = model.encode(texts_all, show_progress_bar=True)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced = tsne.fit_transform(embeddings)

labels = ['Chosen'] * len(chosen_texts) + ['Rejected'] * len(rejected_texts)
colors = ['blue' if label == 'Chosen' else 'red' for label in labels]

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.6)
plt.title("t-SNE of Response Embeddings")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Chosen', markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Rejected', markerfacecolor='red', markersize=10)
])
plt.grid(True)
plt.show()
