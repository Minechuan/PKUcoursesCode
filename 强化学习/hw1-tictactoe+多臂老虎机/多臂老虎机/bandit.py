import numpy as np
from random import random, randint
from time import sleep
from matplotlib import pyplot as plt

class NormalDistBandit:
    def __init__(self, means, stds):
        assert len(means) == len(stds), "Means and stds must be the same length."
        self.n = len(means)
        self.means = np.array(means)
        self.stds = np.array(stds)
        assert all(self.stds >= 0), "Stds must be positive."
    
    def pull(self, k):
        assert 0 <= k < self.n, f"Invalid arm {k}."
        return np.random.normal(loc=self.means[k], scale=self.stds[k])

def epsilon_greedy(values, epsilon):
    assert len(values) > 1, "There should be 2 or more values."
    eps = epsilon * len(values) / (len(values) - 1)
    if random() <= eps:
        return randint(0, len(values)-1)
    return int(np.argmax(values))
   
        
if __name__ == "__main__":
    '''11 bandits, means from -5 to 5, std=1'''
    n = 5
    bandit = NormalDistBandit(means = np.array(range(-n, n+1)), stds = np.ones(11))
    
    # Task：把下面这几种epsilon的曲线画到一张图中，分析你所观察到的结果。
    iter = 10000
    
    x = np.array(range(iter))

    '''以下绘制epsilon-greedy的曲线图'''
    epsilons = [0.01, 0.05, 0.1, 0.2]
    y_eps_greedy = np.zeros((len(epsilons),iter), dtype=np.float64)
    for idx, eps in enumerate(epsilons):
        values = np.zeros(n*2+1, dtype=np.float64)
        counts = np.zeros(n*2+1, dtype=np.int64)
        for i in range(1, iter):
            action = epsilon_greedy(values, eps)
            counts[action] += 1
            value = bandit.pull(action)
            values[action] = (values[action] * (counts[action] - 1) + value) / counts[action]
            y_eps_greedy[idx][i] = (y_eps_greedy[idx][i-1] * (i-1) + value) / i

    


    '''以下绘制UCB的曲线图'''
    UCB_c = [0.01, 0.5, 2, 10]
    y_UCB = np.zeros((len(UCB_c),iter), dtype=np.float64)
    
    for idx, c in enumerate(UCB_c):
        values = np.zeros(n*2+1, dtype=np.float64)
        counts = np.zeros(n*2+1, dtype=np.int64)
        for i in range(1,iter):
            action = np.argmax(values + c * np.sqrt(np.log(i) / (counts + 1e-5)))
            counts[action] += 1
            value = bandit.pull(action)
            values[action] = (values[action] * (counts[action] - 1) + value) / counts[action]
            y_UCB[idx][i] = (y_UCB[idx][i-1] * (i-1) + value) / i




    '''以下绘制 Gradient Bandit 的曲线图'''
    alphas = [0.01, 0.1, 0.5, 1]
    y_gradient = np.zeros((len(alphas),iter), dtype=np.float64)
    for idx, alpha in enumerate(alphas):
        H = np.zeros(n*2+1, dtype=np.float64)
        avg_reward = 0.0
        for i in range(1,iter):
            exp_H = np.exp(H - np.max(H))  # for numerical stability
            action_probs = exp_H / np.sum(exp_H)
            action = np.random.choice(range(n*2+1), p=action_probs)
            value = bandit.pull(action) # reward
            avg_reward = (avg_reward * (i-1) + value) / i
            # update H
            one_hot = np.zeros(n*2+1, dtype=np.float64)
            one_hot[action] = 1.0
            H += alpha * (value - avg_reward) * (one_hot - action_probs)
            # if action: H[action] += alpha*(value-avg_reward)*(1-action_probs[action])
            # otherwise: H[k] += -alpha*(value-avg_reward)*action_probs[k] for all k != action
            y_gradient[idx][i] = (y_gradient[idx][i-1] * (i-1) + value) / i

    '''以下绘制优化初值的曲线图'''
    init_values = [0.0, 2.5, 5.0, 7.5]
    y_init_opt = np.zeros((len(init_values),iter), dtype=np.float64)
    for idx, ini_val in enumerate(init_values):

        values = np.zeros(n*2+1, dtype=np.float64)
        values += ini_val
        counts = np.zeros(n*2+1, dtype=np.int64)
        for i in range(1, iter):
            action = np.argmax(values)
            counts[action] += 1
            value = bandit.pull(action)
            values[action] = (values[action] * (counts[action] - 1) + value) / counts[action]
            y_init_opt[idx][i] = (y_init_opt[idx][i-1] * (i-1) + value) / i


    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    # Gradient Bandit plot
    axs[0][0].plot(x, y_eps_greedy.T)
    axs[0][0].set_xlabel('Iterations')
    axs[0][0].set_ylabel('Average reward')
    axs[0][0].set_title('Epsilon-Greedy')
    axs[0][0].legend([f'epsilon = {epsilons[i]}' for i in range(y_eps_greedy.shape[0])])

    # UCB plot
    axs[0][1].plot(x, y_UCB.T)
    axs[0][1].set_xlabel('Iterations')
    axs[0][1].set_ylabel('Average reward')
    axs[0][1].set_title('UCB')
    axs[0][1].legend([f'c = {UCB_c[i]}' for i in range(y_UCB.shape[0])])

    axs[1][0].plot(x, y_gradient.T)
    axs[1][0].set_xlabel('Iterationxs')
    axs[1][0].set_ylabel('Average reward')
    axs[1][0].set_title('Gradient Bandit')
    axs[1][0].legend([f'alpha = {alphas[i]}' for i in range(y_gradient.shape[0])])

    # UCB plot
    axs[1][1].plot(x, y_init_opt.T)
    axs[1][1].set_xlabel('Iterations')
    axs[1][1].set_ylabel('Average reward')
    axs[1][1].set_title('Optimistic Initial Values')
    axs[1][1].legend([f'initial value = {init_values[i]}' for i in range(y_init_opt.shape[0])])

    plt.tight_layout()  # Prevents overlap
    plt.show()

"""
Findings:
1. 较小的epsilon（如0.01）在初期表现较差，因为探索的机会较少，可能错过了更优的选项。
2. 较大的epsilon（如0.2）在初期表现较好，因为更多的探索帮助发现了更优的选项。但是收敛时 reward 较低，因为过多的探索导致未能充分利用已知的最佳选项。
3. 在测试的 4 组数据中最优的 epsilon 是 0.05，达到了最高的 reward。



"""