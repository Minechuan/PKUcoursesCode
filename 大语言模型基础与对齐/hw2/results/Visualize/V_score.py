import json
import matplotlib.pyplot as plt

def load_scored_data(path):
    """从 JSON 文件中加载 scored_data 列表。"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_scores(scored_data):
    """
    根据 win_id 拆分 chosen 和 rejected 的分数列表。
    假设 win_id == 1 表示 response_1 被选择，== 2 表示 response_2 被选择。
    """
    chosen_scores = []
    rejected_scores = []
    for item in scored_data:
        if item['win_id'] == 1:
            chosen_scores.append(item['score_1'])
            rejected_scores.append(item['score_2'])
        else:
            chosen_scores.append(item['score_2'])
            rejected_scores.append(item['score_1'])
    return chosen_scores, rejected_scores

def plot_histograms(chosen, rejected, bins=30):
    """绘制两个分数分布的叠加直方图。"""
    plt.figure()
    plt.hist(chosen, bins=bins, alpha=0.5, label='Chosen', density=True)
    plt.hist(rejected, bins=bins, alpha=0.5, label='Rejected', density=True)
    plt.title('Base Reward Model Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_boxplot(chosen, rejected):
    """绘制箱线图，直观比较两个分数分布的中位数和分散程度。"""
    plt.figure()
    plt.boxplot([chosen, rejected], labels=['Chosen', 'Rejected'])
    plt.title('Base Reward Model Score Comparison')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

def main():
    data_path = 'base_score.json'

    scored_data = load_scored_data(data_path)
    chosen_scores, rejected_scores = extract_scores(scored_data)

    # 绘制直方图
    plot_histograms(chosen_scores, rejected_scores, bins=50)

    # 绘制箱线图
    plot_boxplot(chosen_scores, rejected_scores)

if __name__ == '__main__':
    main()
