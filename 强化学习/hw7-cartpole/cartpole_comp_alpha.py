import gym
import numpy as np
import tqdm
import random
import math
from typing import Tuple, Dict, Any
import os
from termcolor import cprint


class DiscreteCartPoleEnv(gym.Env):
    def __init__(self, intervals=16):
        self._env = gym.make('CartPole-v1')
        self.action_space = self._env.action_space
        self.intervals = intervals
        self.observation_space = gym.spaces.MultiDiscrete([intervals]*4)
        self._to_discrete = lambda x, a, b: int(min(max(0, (x-a)*self.intervals/(b-a)), self.intervals))
        
    def render(self):
        self._env.render()
    
    def reset(self):
        return self._discretize(self._env.reset())

    def _discretize(self, state:np.array)->Tuple:
        if type(state) is tuple:
            state,info = state
            state = state.tolist()
        cart_pos, cart_v, pole_angle, pole_v = state
        cart_pos = self._to_discrete(cart_pos, -2.4, 2.4)
        cart_v = self._to_discrete(cart_v, -3.0, 3.0)
        pole_angle = self._to_discrete(pole_angle, -0.5, 0.5)
        pole_v = self._to_discrete(pole_v, -2.0, 2.0)
        return (cart_pos, cart_v, pole_angle, pole_v)
    
    def step(self, action:int)->Tuple[Tuple, float, bool, Any]:
        state, reward, done, truncated, info = self._env.step(action)
        state = self._discretize(state)
        return state, reward, done, info


class nstep_TD_Learner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_upper
        self.current_ckpt = 0
        cprint(f"Save frequency: every {self.save_freq} iteration",'yellow')
         

    def add_to_buffer(self, data):

        # buffer[i] is (s_{i},a_{i}, r_{i+1}, s_{i+1}, a_{i+1})
        self.buffer.append(data)

    def greedy(self, state:Tuple)->int:
        return self.q[state].argmax()

    def epsilon_greedy(self, state:Tuple)->int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy(state)
    
    def epsilon_decay(self, total_step):
        # total_step == 0: epsilon = epsilon_upper
        # total_step -> inf: epsilon = epsilon_lower
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)
    
    def update_q(self, tau, T):
        # batch update for each episode in the buffer

        low = tau + 1
        high = min(tau + self.n, T)
        G = 0.0

        # buffer[i] is (s_{i}, a_{i}, r_{i+1}, s_{i+1}, a_{i+1})
        # cprint(f'Updating Q at time tau={tau}, T={T}, n={self.n}, low={low}, high={high}','yellow')
        for i in range(low, high + 1):
            assert(i-1<len(self.buffer)), f'Buffer length: {len(self.buffer)}, i-1: {i-1}, low: {low}, high: {high}, tau: {tau}, T: {T}'
            r_i = self.buffer[i-1][2]  # reward r_i
            G += (self.gamma ** (i - tau - 1)) * r_i

        # bootstrapped term: only if tau + n < T
        if tau + self.n < T:
            s_tn = self.buffer[tau + self.n-1][3]  # s_{t+n}
            a_tn = self.buffer[tau + self.n-1][4]
            # max over actions at s_{t+n}
            G += (self.gamma ** self.n) * self.q[s_tn][a_tn]

        # the state and action to update correspond to time tau:
        # buffer[tau+1] stores (s_tau, a_tau, r_{tau+1}, s_{tau+1}, a_{tau+1})
        state_tau = self.buffer[tau][0]
        action_tau = self.buffer[tau][1]

        # perform Q update for the single (s_tau, a_tau)
        td_error = G - self.q[state_tau][action_tau]
        try:
            self.q[state_tau][action_tau] += self.lr * td_error
        except:
            print(f'Error updating Q at state {state_tau}, action {action_tau}')
            exit(1)

    def train(self):
        self.env_step_num = 0

        for episode_num in range(self.start_iter, self.iter+1):
            T = 1e6 # maximum time step
            state = self.env.reset()
            done = False
            action = self.epsilon_greedy(state)
            self.buffer = list()
            current_step = -1 # record current step in episode
            tau = 0 # time whose estimate is being updated

            while tau != T - 1:
                current_step += 1
                self.epsilon_decay(self.env_step_num)

                if current_step < T: # not terminal
                    
                    new_state, reward, done, _ = self.env.step(action)
                    self.env_step_num += 1
                    if done:
                        reward = self.end_reward
                        T = current_step + 1 # for last s_{t+1} is terminal
                        # print(f'Episode {i} ended at step {T-1}, last {current_step} steps.')
                        self.add_to_buffer((state, action, reward, state, -1)) # only need state, reward, new_state

                    else:
                        new_action = self.epsilon_greedy(new_state)
                        if self.render:
                            self.env.render()
                        self.add_to_buffer((state, action, reward, new_state, new_action)) # only need state, reward, new_state

                # update use tau
                tau = current_step - self.n + 1
                if tau >= 0:
                    self.update_q(tau,T)
                state = new_state
                action = new_action if not done else -1
            # self.save_model(self.env_step_num,episode_num)

    def save_model(self, step,episode):
        if episode % self.save_freq == 0:
            # print(f'Saving checkpoint at step {step}, episode {episode}, ckpt {self.current_ckpt}')
            np.save(os.path.join(self.save_path, f'{episode}.npy'), self.q)
            rd = test_k_rounds(self.test_env, self.q, k=20)
            cprint(f'Test after saving ckpt at step {step}, episode {episode}, mean reward: {round(rd[-1], 2)}', 'cyan')





def train_test_model(N_STEP=None,alpha = 0.5):
    env_name = 'DiscreteCartPole'
    

    EPISODES = 1000
    INTERVALS = 16
    save_path = f'interval_{INTERVALS}_N_{N_STEP}_alpha_{alpha}'

    env = DiscreteCartPoleEnv(INTERVALS)
    test_env = DiscreteCartPoleEnv(INTERVALS)
    print(" n-step SARSA for estimating Q (without replay buffer)")
    print(f" n = {N_STEP} ")
    q_table = np.zeros(shape=(INTERVALS+1,)*env.observation_space.shape[0]+(env.action_space.n,))
    
            
    trainer = nstep_TD_Learner({
        'env':env,
        'test_env':test_env,
        'env_name':env_name,
        'render':False,
        'end_reward':-1,
        'q':q_table,
        'n':N_STEP,
        'start_iter':0,
        'iter':EPISODES,
        'gamma':0.95,
        'epsilon_lower':0.1,
        'epsilon_upper':0.4,
        'epsilon_decay_freq':1000,
        'lr':alpha,
        'save_path':save_path,
        'save_freq':1e8 # not save during training
    })

    trainer.train()

    print("*"*40)

    max_reward, min_reward, mean_reward = test_k_rounds(test_env, trainer.q, k=40)
    cprint(f'Test mean reward: {round(mean_reward, 2)}', 'green')
    cprint(f'Test min reward: {min_reward}', 'green')
    cprint(f'Test max reward: {max_reward}', 'green')
    print("*"*40)
    env.close()
    test_env.close()
    return mean_reward



def test(env, q_table):
    state = env.reset()
    done = False
    episode_reward = 0
    cnt = 0
    while not done:
        cnt += 1
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        if done:
            reward = -1
        episode_reward += reward
        if cnt > 1000:
            break
    return episode_reward


def test_k_rounds(env, q_table, k=5):
    max_reward = float('-inf')
    min_reward = float('inf')
    mean_reward = 0.0
    episode_reward_list = []

    for _ in range(k):
        episode_reward = test(env, q_table)
        max_reward = max(max_reward, episode_reward)
        min_reward = min(min_reward, episode_reward)
        episode_reward_list.append(episode_reward)
    
    if k >= 4:  # 只有当k>=4时才进行去除操作
        remove_count = k // 4  # 计算要去除的数量
        
        # 排序并去除remove_count个最高分和最低分
        episode_reward_list.sort()
        trimmed_list = episode_reward_list[remove_count:-remove_count] if remove_count > 0 else episode_reward_list
        
        # 重新计算k值
        new_k = len(trimmed_list)
        mean_reward = sum(trimmed_list) / new_k if new_k > 0 else 0.0
    else:
        # 如果k<4，直接计算平均值
        mean_reward = sum(episode_reward_list) / k
    
    return max_reward, min_reward, mean_reward

def test_each_ckpt(ckpt_dir, INTERVALS=8):
    env = DiscreteCartPoleEnv(INTERVALS)
    file_names = [int(f.split('.')[0]) for f in os.listdir(ckpt_dir) if f.endswith('.npy')]
    # Sort the file names
    file_names.sort()
    test_results = []
    for i in tqdm.tqdm(file_names):
        q_table = np.load(os.path.join(ckpt_dir, f'{i}.npy'))
        # Test the loaded Q-table
        rd = test_k_rounds(env,q_table,k=20)
        test_results.append(rd[-1])
    # print("="*40)
    # cprint(f'Test results for each checkpoint in {ckpt_dir}:', 'cyan')

    # # 打印表头
    # cprint('id\tmax\tmin\tmean', 'yellow')

    # 打印每个检查点的结果
    # for file_name, rd in test_results.items():
    #     max_val, min_val, mean_val = rd
    #     cprint(f'{file_name}\t{max_val}\t{min_val}\t{mean_val}', 'magenta')
    return test_results


if __name__ == '__main__':
    N_STEP_list = [1,2,4,8,16,32,64,128]
    alpha_list = [0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]
    # N_STEP_list = [1,8]
    # alpha_list = [0.1,0.3,0.5]
    reluts = np.zeros((len(N_STEP_list),len(alpha_list)))
    for N_STEP in N_STEP_list:
        for alpha in alpha_list:
            reluts[N_STEP_list.index(N_STEP)][alpha_list.index(alpha)] = train_test_model(N_STEP=N_STEP,alpha=alpha)

    # plot the len(N_STEP_list) curves with x axis as alpha and y axis as mean reward
    import matplotlib.pyplot as plt
    for i, N_STEP in enumerate(N_STEP_list):
        plt.plot(alpha_list, reluts[i], label=f'n={N_STEP}')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward vs Alpha for Different n-step Values')
    plt.legend()
    save_path = f'cartpole_comp_alpha.png'
    plt.savefig(save_path)
    plt.show()

            







