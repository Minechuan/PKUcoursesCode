# 环境
from gym_env import GymEnv
import torch
from q_network import QNetwork_cartpole as QNetwork



env = GymEnv('CartPole-v0')
state_dim = env.state_dim[0]
action_dim = env.action_dim


from sample import FrameNumpy, SampleBatchNumpy
from collections import deque
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

# 训练流程
buffer_size = 2000
batch_size = 64
episodes = 600
copy_steps = 256

# 超参数'
lr_decay = 0.96
lr_decay_steps = 1000

lr = 1e-3
epsilon = 0.05
gamma = 0.98



# 算法
from dqn_agent import DQNAgent
conf = dict(
    action_dim = action_dim,
    epsilon = epsilon,
    gamma = gamma,
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(conf)
agent = DQNAgent(conf)

# 模型




# current Q网络
model = QNetwork(state_dim, action_dim, lr = lr)
# target Q网络
target_model = QNetwork(state_dim, action_dim, lr = lr)
target_model.to(conf['device'])
# 将模型设置到agent中
agent.set_model(model)




def update_target_model(target_model, model=model):
    target_model.load_state_dict(model.state_dict())
    return



train_returns = []
test_returns = []
step = 0
replay_buffer = deque(maxlen = buffer_size) # 样本池
pbar = tqdm(range(episodes), ncols=100)
for episode in pbar:
    ret = 0
    obs = env.reset()
    done = False
    while not done:
        step += 1
        if step % copy_steps == 0:
            # 同步target网络
            update_target_model(target_model,agent.model)
        if step % lr_decay_steps == 0:
            # 学习率衰减
            agent.model.optimizer.param_groups[0]['lr'] *= lr_decay

        action = agent.predict(obs) # 采样动作

        next_obs, reward, done, _ = env.step(action)
        ret += reward
        sample = FrameNumpy.from_dict({
            'obs': obs,
            'next_obs': next_obs,
            'action': action,
            'reward': reward,
            'done': done
        })
        obs = next_obs
        # 每个step产生的样本加入样本池，并直接采样batch进行单次训练
        replay_buffer.append(sample)
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch = SampleBatchNumpy.stack(batch)
            agent.sample_process(batch)
            loss = agent.learn(batch,target_model)
    train_returns.append((episode, ret))
    if episode % 10 == 0:
        # 每10局测试一局效果
        ret = 0
        obs = env.reset()
        done = False
        while not done:
            action = agent.exploit(obs) # 最优动作
            next_obs, reward, done, _ = env.step(action)
            ret += reward
            obs = next_obs
        test_returns.append((episode, ret))
        pbar.set_postfix(reward=ret,lr=agent.model.optimizer.param_groups[0]['lr'],step=step)




plt.plot([x[0] for x in train_returns], [x[1] for x in train_returns], label = 'train')
plt.plot([x[0] for x in test_returns], [x[1] for x in test_returns], label = 'test')
plt.legend()
plt.title("CartPole")
plt.savefig(f"./cartpole_results/CartPole_lr{lr}_epsilon{epsilon}_gamma{gamma}_buffer{buffer_size}_batch{batch_size}_update{copy_steps}.png")
plt.show()