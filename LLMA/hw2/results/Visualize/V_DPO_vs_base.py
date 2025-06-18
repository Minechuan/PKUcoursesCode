import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set(style="whitegrid")  # 设置更精致的背景风格

def load_data(file1, file2):
    """Load data from two JSON list files and merge into DataFrame"""
    with open(file1, 'r', encoding='utf-8') as f:
        dpo_scores = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        original_scores = json.load(f)

    df_dpo = pd.DataFrame(dpo_scores)
    df_dpo['model_type'] = 'DPO'
    
    df_ori = pd.DataFrame(original_scores)
    df_ori['model_type'] = 'Original'

    df = pd.concat([df_dpo, df_ori], ignore_index=True)
    return df

def analyze_scores(df):
    """Print statistics and significance test"""
    grouped = df.groupby('model_type')['reward_score']
    print("\nBasic Statistics:")
    print(grouped.describe())

    dpo_scores = df[df['model_type'] == 'DPO']['reward_score']
    ori_scores = df[df['model_type'] == 'Original']['reward_score']

    u_stat, p_val = stats.mannwhitneyu(dpo_scores, ori_scores, alternative='two-sided')
    print(f"\nMann-Whitney U test: U={u_stat:.2f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("✅ Statistically significant difference (p < 0.05)")
    else:
        print("ℹ️ Not statistically significant")

def plot_box(df):
    """Plot refined boxplot with seaborn"""
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=df, x='model_type', y='reward_score', palette=['#8ecae6', '#90be6d'], width=0.5)
    sns.stripplot(data=df, x='model_type', y='reward_score', color='black', alpha=0.25, jitter=0.2, size=3)
    plt.title('Reward Score Distribution by Model', fontsize=14)
    plt.xlabel('Model Type')
    plt.ylabel('Reward Score')
    plt.tight_layout()
    plt.show()

def plot_histogram(df):
    """Plot histogram comparing score distributions"""
    plt.figure(figsize=(8, 6))
    sns.histplot(df[df['model_type']=='Original']['reward_score'], 
                 bins=25, color='#219ebc', label='Original', kde=True, stat='density', alpha=0.6)
    sns.histplot(df[df['model_type']=='DPO']['reward_score'], 
                 bins=25, color='#55a630', label='DPO', kde=True, stat='density', alpha=0.6)
    plt.title('Histogram of Reward Scores', fontsize=14)
    plt.xlabel('Reward Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_score_diff(df):
    """Plot histogram of score differences (DPO - Original)"""
    if 'prompt' not in df.columns:
        print("Missing 'prompt' key, cannot compare per-prompt.")
        return

    # Replace pivot with pivot_table to handle duplicates
    pivot_df = df.pivot_table(
        index='prompt',
        columns='model_type',
        values='reward_score',
        aggfunc='mean'  # or use 'first' if appropriate
    )
    pivot_df = pivot_df.dropna()  # drop rows missing either model
    pivot_df['score_diff'] = pivot_df['DPO'] - pivot_df['Original']

    plt.figure(figsize=(8, 6))
    sns.histplot(pivot_df['score_diff'], bins=30, color='purple', kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Score Differences (DPO - Original)', fontsize=14)
    plt.xlabel('Difference in Reward Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    print("\nScore Difference Summary:")
    print(pivot_df['score_diff'].describe())

def plot_response_length(df):
    """Compare generated response lengths (in words)"""
    dpo_lengths = df[df['model_type'] == 'DPO']['response'].dropna().apply(lambda x: len(x.split()))
    ori_lengths = df[df['model_type'] == 'Original']['response'].dropna().apply(lambda x: len(x.split()))

    plt.figure(figsize=(8, 6))
    plt.hist(dpo_lengths, bins=30, alpha=0.6, label='DPO gen', color='#2a9d8f', density=True)
    plt.hist(ori_lengths, bins=30, alpha=0.6, label='Ori gen', color='#e76f51', density=True)
    plt.xlabel('Number of Words')
    plt.ylabel('Density')
    plt.title('Response Length Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\nResponse Length Stats:")
    print(f"DPO mean length: {np.mean(dpo_lengths):.2f}")
    print(f"Original mean length: {np.mean(ori_lengths):.2f}")

if __name__ == "__main__":
    dpo_scores_file = "scores_RM_dpo.json"
    original_scores_file = "scores_RM_qwen.json"

    df = load_data(dpo_scores_file, original_scores_file)
    analyze_scores(df)

    # Draw figures separately
    plot_box(df)
    plot_histogram(df)
    plot_score_diff(df)
    plot_response_length(df)
