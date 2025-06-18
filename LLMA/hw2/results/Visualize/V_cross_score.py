import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import json

def load_json_list(filepath):
    """加载一个标准 JSON 数组文件。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)  # 仅支持 JSON array 格式

def load_data(sftrm_dpo_path, sftrm_ori_path, orirm_dpo_path, orirm_ori_path):
    sftrm_dpo = load_json_list(sftrm_dpo_path)
    sftrm_ori = load_json_list(sftrm_ori_path)
    orirm_dpo = load_json_list(orirm_dpo_path)
    orirm_ori = load_json_list(orirm_ori_path)

    data = {
        'score': (
            [item['reward_score'] for item in sftrm_dpo] +
            [item['reward_score'] for item in sftrm_ori] +
            [item['reward_score'] for item in orirm_dpo] +
            [item['reward_score'] for item in orirm_ori]
        ),
        'reward_model': (
            ['SFTRM'] * (len(sftrm_dpo) + len(sftrm_ori)) +
            ['Ori_RM'] * (len(orirm_dpo) + len(orirm_ori))
        ),
        'generation_source': (
            ['DPO_gen'] * len(sftrm_dpo) +
            ['Ori_gen'] * len(sftrm_ori) +
            ['DPO_gen'] * len(orirm_dpo) +
            ['Ori_gen'] * len(orirm_ori)
        )
    }

    return data

data = load_data(
    'scores_RM_dpo.json',
    'scores_RM_qwen.json',
    'scores_qwen_dpo.json',
    'scores_qwen_qwen.json'
)
df = pd.DataFrame(data)
df['Model_Pair'] = df['reward_model'] + '→' + df['generation_source']

# Set style
sns.set(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("Set2", n_colors=len(df['Model_Pair'].unique()))  # Use only needed colors

# === 1. Boxplot with mean annotations ===
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Model_Pair', y='score', data=df, palette=palette)

# Calculate means and find box positions
means = df.groupby('Model_Pair')['score'].mean().values
boxes = [artist for artist in ax.artists if isinstance(artist, plt.Rectangle)]

for box, mean in zip(boxes, means):
    # Get box coordinates
    box_x = box.get_x() + box.get_width()/2
    box_top = box.get_y() + box.get_height()
    
    # Place text above the box
    ax.text(box_x, box_top + 0.5, f"{mean:.2f}", 
            ha='center', va='bottom', color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

plt.title('Comparison of Reward Scores Across RM and Generators (Boxplot)')
plt.xlabel('Reward Model → Generation Source')
plt.ylabel('Reward Score')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === 2. Bar plot showing means ===
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Model_Pair', y='score', data=df, palette=palette, estimator='mean', ci=None)

# Annotate bars with mean values
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points')

plt.title('Mean Reward Score by Model Pair (Bar Plot)')
plt.xlabel('Reward Model → Generation Source')
plt.ylabel('Mean Reward Score')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()