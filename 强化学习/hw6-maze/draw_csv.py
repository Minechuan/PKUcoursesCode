import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV 文件
dyna_q = np.loadtxt("dyna_q_mean.csv", delimiter=",")
dyna_q_plus = np.loadtxt("dyna_q_plus_mean.csv", delimiter=",")

# 构造 x 轴（每个点的索引）
x = np.arange(len(dyna_q))

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, dyna_q, color='blue', label='Dyna-Q', linewidth=2)
plt.plot(x, dyna_q_plus, color='red', label='Dyna-Q+', linewidth=2)

# 美化图像
plt.xlabel("Steps")
plt.ylabel("Mean Accumulated Reward")
plt.title("Dyna-Q vs Dyna-Q+")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图像
plt.show()
