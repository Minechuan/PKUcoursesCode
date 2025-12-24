import gym
import numpy as np
import tqdm
import random
import math
import os
from termcolor import cprint
import matplotlib.pyplot as plt

# -----------------------
# Discrete wrapper
# -----------------------
class DiscreteCartPoleEnv(gym.Env):
    def __init__(self, intervals=8):
        self._env = gym.make('CartPole-v1', render_mode=None)
        self.intervals = intervals
        # observation discrete shape: 4 dims, each 0..intervals-1
        self.observation_space = gym.spaces.MultiDiscrete([intervals]*4)
        self.action_space = self._env.action_space

        # clipping ranges chosen reasonably for CartPole
        self._obs_low_high = [
            (-2.4, 2.4),   # cart position
            (-3.0, 3.0),   # cart velocity (clipped)
            (-0.5, 0.5),   # pole angle (clipped)
            (-2.0, 2.0)    # pole tip velocity (clipped)
        ]

    def _to_discrete(self, x, a, b):
        # map x in [a,b] to integer in [0, intervals-1]
        frac = (x - a) / (b - a)
        idx = int(np.clip(np.floor(frac * self.intervals), 0, self.intervals - 1))
        return idx

    def _discretize(self, obs):
        # support gym versions where reset returns (obs, info) or just obs
        if isinstance(obs, tuple):
            obs = obs[0]
        # ensure numpy array
        obs = np.asarray(obs, dtype=float)
        vals = []
        for i, (a, b) in enumerate(self._obs_low_high):
            vals.append(self._to_discrete(obs[i], a, b))
        return tuple(vals)

    def reset(self):
        rs = self._env.reset()
        return self._discretize(rs)

    def step(self, action: int):
        # handle gym versions returning (obs, reward, terminated, truncated, info)
        step_ret = self._env.step(action)
        if len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = terminated or truncated
        else:
            obs, reward, done, info = step_ret
        return self._discretize(obs), reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

# -----------------------
# n-step SARSA learner
# -----------------------
class NStepSarsaLearner:
    def __init__(self, config):
        # config keys directly set as attributes
        for k, v in config.items():
            setattr(self, k, v)

        # initialize dynamic params
        self.epsilon = self.epsilon_upper
        self.lr = self.lr_upper
        self.env_step_num = 0
        self.current_ckpt = 0
        cprint(f"Save frequency: every {self.save_freq} episodes", 'yellow')

    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q[state]))

    def epsilon_decay(self, total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)

    def lr_decay(self, total_step):
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)

    def save_model_and_test(self, episode_idx):
        if episode_idx % self.save_freq == 0:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
            np.save(os.path.join(self.save_path, f'{episode_idx}.npy'), self.q)
            _, _, mean = test_k_rounds(self.test_env, self.q, k=20)
            cprint(f'[ckpt {self.current_ckpt}] episode {episode_idx} | mean reward (20 trials) = {mean:.2f}', 'cyan')
            self.current_ckpt += 1

    def train(self):
        for ep in range(self.start_iter, self.iter + 1):
            # reset env
            S = []  # states
            A = []  # actions
            R = [0.0]  # rewards, R[0] is dummy per textbook indexing

            s0 = self.env.reset()
            a0 = self.epsilon_greedy(s0)
            S.append(s0)
            A.append(a0)

            T = float('inf')
            t = 0

            while True:
                # step if not yet terminal
                if t < T:
                    s_t = S[t]
                    a_t = A[t]
                    # take action
                    (s_tp1, r_tp1, done, _info) = self.env.step(a_t)
                    R.append(r_tp1)
                    self.env_step_num += 1
                    # if terminal
                    if done:
                        T = t + 1
                        S.append(None)  # placeholder for terminal
                        A.append(None)
                    else:
                        a_tp1 = self.epsilon_greedy(s_tp1)
                        S.append(s_tp1)
                        A.append(a_tp1)

                # update tau
                tau = t - self.n + 1
                if tau >= 0:
                    # compute G
                    G = 0.0
                    # sum rewards from tau+1 to min(tau+n, T)
                    upper = int(min(tau + self.n, T))
                    for i in range(tau + 1, upper + 1):
                        G += (self.gamma ** (i - (tau + 1))) * R[i]
                    # bootstrap if tau + n < T
                    if tau + self.n < T:
                        s_tn = S[tau + self.n]
                        a_tn = A[tau + self.n]
                        G += (self.gamma ** self.n) * self.q[s_tn][a_tn]

                    s_tau = S[tau]
                    a_tau = A[tau]
                    # update Q(s_tau, a_tau)
                    td_error = G - self.q[s_tau][a_tau]
                    self.q[s_tau][a_tau] += self.lr * td_error

                # decay schedules per environment steps (optional)
                self.epsilon_decay(self.env_step_num)
                self.lr_decay(self.env_step_num)

                if tau == T - 1:
                    break
                t += 1

            # end of episode: save & test checkpoint periodically
            self.save_model_and_test(ep)

# -----------------------
# Testing utilities
# -----------------------
def test(env, q_table, max_steps=500):
    state = env.reset()
    done = False
    total = 0.0
    steps = 0
    while not done and steps < max_steps:
        a = int(np.argmax(q_table[state]))
        state, r, done, _ = env.step(a)
        total += r
        steps += 1
    return total

def test_k_rounds(env, q_table, k=5):
    rewards = []
    for _ in range(k):
        rewards.append(test(env, q_table))
    max_r = max(rewards)
    min_r = min(rewards)
    if k >= 4:
        # trim top/bottom 25% as in your original code
        rm = k // 4
        rewards_sorted = sorted(rewards)
        trimmed = rewards_sorted[rm:-rm] if rm > 0 else rewards_sorted
        mean_r = sum(trimmed) / len(trimmed) if len(trimmed) > 0 else 0.0
    else:
        mean_r = sum(rewards) / k
    return max_r, min_r, mean_r

def test_each_ckpt(ckpt_dir, INTERVALS=8):
    env = DiscreteCartPoleEnv(INTERVALS)
    file_names = [int(f.split('.')[0]) for f in os.listdir(ckpt_dir) if f.endswith('.npy')]
    file_names.sort()
    test_results = []
    for i in tqdm.tqdm(file_names):
        q_table = np.load(os.path.join(ckpt_dir, f'{i}.npy'), allow_pickle=True)
        _, _, mean = test_k_rounds(env, q_table, k=20)
        test_results.append(mean)
    return test_results

# -----------------------
# Main runner
# -----------------------
def train_test_model(n_step, intervals=8, episodes=200):
    env = DiscreteCartPoleEnv(intervals)
    test_env = DiscreteCartPoleEnv(intervals)

    q_shape = tuple([intervals]*len(env.observation_space.nvec)) + (env.action_space.n,)
    q_table = np.zeros(q_shape, dtype=float)

    save_path = f'q_tables_n{n_step}_i{intervals}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    learner = NStepSarsaLearner({
        'env': env,
        'test_env': test_env,
        'q': q_table,
        'n': n_step,
        'start_iter': 0,
        'iter': episodes,
        'gamma': 0.95,
        'epsilon_lower': 0.05,
        'epsilon_upper': 0.4,
        'epsilon_decay_freq': 1000.0,
        'lr_lower': 0.01,
        'lr_upper': 0.5,
        'lr_decay_freq': 2000.0,
        'save_path': save_path,
        'save_freq': max(1, episodes // 10)  # save 10 checkpoints
    })

    learner.train()
    # final test
    max_r, min_r, mean_r = test_k_rounds(test_env, learner.q, k=10)
    cprint(f'Final test (n={n_step}) mean: {mean_r:.2f}, min: {min_r:.2f}, max: {max_r:.2f}', 'green')
    env.close()
    test_env.close()
    return save_path

if __name__ == '__main__':
    # run experiments for multiple n values and plot results
    test_n_steps = [1, 2, 8, 64]  # 512 might be too large for episodes=200
    intervals = 8
    episodes = 200

    ckpt_dirs = {}
    for n in test_n_steps:
        cprint(f'\n========== Training n={n} ==========', 'magenta')
        ckpt_dir = train_test_model(n, intervals=intervals, episodes=episodes)
        ckpt_dirs[n] = ckpt_dir

    # evaluate each checkpoint sequence and plot
    results = {}
    for n, ckpt_dir in ckpt_dirs.items():
        rs = test_each_ckpt(ckpt_dir, INTERVALS=intervals)
        results[n] = rs

    # plotting
    plt.figure(figsize=(8,5))
    for n, rs in results.items():
        plt.plot(range(len(rs)), rs, label=f'n={n}')
    plt.xlabel('Checkpoint index')
    plt.ylabel('Mean reward (20 trials)')
    plt.title('n-step SARSA: mean reward per checkpoint')
    plt.legend()
    plt.grid(True)
    plt.show()
