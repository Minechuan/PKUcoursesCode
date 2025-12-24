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

class QLearner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_lower
        self.lr = self.lr_upper
        self.buffer = list()
        self.current_ckpt = 0
        cprint(f"Save frequency: every {self.save_freq} iteration",'yellow')
        self.buffer_pointer = 0

    
    def add_to_buffer(self, data):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(data)
        else:
            # overwrite oldest
            self.buffer[self.buffer_pointer] = data
        self.buffer_pointer += 1
        self.buffer_pointer %= self.buffer_size
    
    def sample_batch(self):
        return random.sample(self.buffer, self.batch_size)
    
    def greedy(self, state:Tuple)->int:
        return self.q[state].argmax()

    def epsilon_greedy(self, state:Tuple)->int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy(state)
    
    def epsilon_decay(self, total_step):
        self.epsilon = self.epsilon_lower + (self.epsilon_upper - self.epsilon_lower) * math.exp(-total_step / self.epsilon_decay_freq)
    
    def lr_decay(self, total_step):
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)
    
    def update_q(self, total_step):
        if total_step % self.update_freq != 0 or len(self.buffer) < self.batch_size:
            # update until enough samples
            return
        batch = self.sample_batch()
        for state, action, reward, new_state in batch:
            self.q[state][action] += self.lr * (self.gamma * self.q[new_state].max() + reward - self.q[state][action])
    
    def train(self):
        total_step = 0
        for i in tqdm.trange(self.start_iter, self.iter+1):
            state = self.env.reset()
            done = False
            while not done:
                total_step += 1
                action = self.epsilon_greedy(state)
                self.epsilon_decay(total_step)
                new_state, reward, done, _ = self.env.step(action)
                if self.render:
                    self.env.render()
                if done:
                    reward = self.end_reward
                self.add_to_buffer((state, action, reward, new_state))
                self.update_q(total_step)
                self.lr_decay(total_step)
                state = new_state
            self.save_model(i)
    
    def save_model(self, i):
        self.current_ckpt += 1
        if i % self.save_freq == 0:
            np.save(os.path.join(self.save_path, f'{i}.npy'), self.q)
            rd = test_k_rounds(self.test_env, self.q, k=10)
            cprint(f'Test after saving ckpt {self.current_ckpt} at episode {i}, mean reward: {round(rd[-1], 2)}', 'cyan')

        

class nstep_TD_Learner:
    def __init__(self, config:Dict):
        for k, v in config.items():
            setattr(self, k, v)
        self.epsilon = self.epsilon_upper
        self.lr = self.lr_upper

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
    
    def lr_decay(self, total_step):
        self.lr = self.lr_lower + (self.lr_upper - self.lr_lower) * math.exp(-total_step / self.lr_decay_freq)
    
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
                
                self.epsilon_decay(self.env_step_num)
                current_step += 1
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
                        # if self.render:
                        #     self.env.render()
                        self.add_to_buffer((state, action, reward, new_state, new_action)) # only need state, reward, new_state

                # update use tau
                tau = current_step - self.n + 1
                if tau >= 0:
                    self.update_q(tau,T)
                self.lr_decay(self.env_step_num)
                state = new_state
                action = new_action if not done else -1
            self.save_model(self.env_step_num,episode_num)

    def save_model(self, step,episode):
        if episode % self.save_freq == 0:
            # print(f'Saving checkpoint at step {step}, episode {episode}, ckpt {self.current_ckpt}')
            np.save(os.path.join(self.save_path, f'{episode}.npy'), self.q)
            rd = test_k_rounds(self.test_env, self.q, k=20)
            cprint(f'Test after saving ckpt {self.current_ckpt} at step {step}, episode {episode}, mean reward: {round(rd[-1], 2)}', 'cyan')
            self.current_ckpt += 1





def train_test_model(name:str, N_STEP=None):
    env_name = 'DiscreteCartPole'
    
    if name == 'Q_learning':
        save_path = 'q_tables'
        intervals = 8
        env = DiscreteCartPoleEnv(intervals)
        test_env = DiscreteCartPoleEnv(intervals)
        q_table = np.zeros(shape=(intervals+1,)*env.observation_space.shape[0]+(env.action_space.n,))
        
        latest_checkpoint = 0
        
        if save_path not in os.listdir():
            os.mkdir(save_path)
        elif len(os.listdir(save_path)) != 0:
            latest_checkpoint = max([int(file_name.split('.')[0]) for file_name in os.listdir(save_path)])
            print(f'{latest_checkpoint}.npy loaded')
            q_table = np.load(os.path.join(save_path, f'{latest_checkpoint}.npy'))
        

        EPISODES = 1000
        trainer = QLearner({
            'env':env,
            'env_name':env_name,
            'test_env':test_env,
            'render':True,
            'end_reward':-1,
            'q':q_table,
            'start_iter':latest_checkpoint,
            'iter':latest_checkpoint+EPISODES,
            'batch_size':128,
            'buffer_size':1000,
            'gamma':0.9,
            'update_freq':1,
            'epsilon_lower':0.05,
            'epsilon_upper':0.8,
            'epsilon_decay_freq':200,
            'lr_lower':0.05,
            'lr_upper':0.5,
            'lr_decay_freq':200,
            'save_path':save_path,
            'save_freq':EPISODES//10
        })
    
    elif name == 'TD_learning':
        EPISODES = 200
        INTERVALS = 8
        save_path = f'q_tables_test_{N_STEP}_{INTERVALS}'

        env = DiscreteCartPoleEnv(INTERVALS)
        test_env = DiscreteCartPoleEnv(INTERVALS)
        print(" n-step SARSA for estimating Q (without replay buffer)")
        print(f" n = {N_STEP} ")
        q_table = np.zeros(shape=(INTERVALS+1,)*env.observation_space.shape[0]+(env.action_space.n,))

        latest_checkpoint = 0
        
        if save_path not in os.listdir():
            os.mkdir(save_path)
                
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
            'epsilon_lower':0.05,
            'epsilon_upper':0.4,
            'epsilon_decay_freq':1000,
            'lr_lower':0.1,
            'lr_upper':1.0,
            'lr_decay_freq':1000,
            'save_path':save_path,
            'save_freq':EPISODES//20
        })
    else:
        raise ValueError("Unknown model name")
    trainer.train()

    print("*"*40)

    max_reward, min_reward, mean_reward = test_k_rounds(test_env, trainer.q, k=5)
    cprint(f'Test mean reward: {mean_reward}', 'green')
    cprint(f'Test min reward: {min_reward}', 'green')
    cprint(f'Test max reward: {max_reward}', 'green')
    print("*"*40)
    env.close()
    test_env.close()



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
        if cnt > 500:
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
    test_n_steps = [1,8, 64,512]
    for N_STEP in test_n_steps:
        train_test_model('TD_learning', N_STEP=N_STEP)







    ##############################################
    #   Test
    ###############################################


    root_dir = "./"

    results = {}
    for test_n in test_n_steps:
        ckpt_dir = os.path.join(root_dir, f'q_tables_test_{test_n}_{8}')
        rs = test_each_ckpt(ckpt_dir)
        results[test_n] = rs

    # plot the image
    import matplotlib.pyplot as plt
    for n_step, rs in results.items():
        means = rs
        plt.plot(range(len(means)), means, label=f'n={n_step}')


    plt.xlabel('Checkpoint')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward vs Checkpoint for Different n-step Values')
    plt.legend()
    plt.show()