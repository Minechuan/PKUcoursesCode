# 环境
from gym_env import GymEnv
import torch
from q_network import QNetwork_breakout as QNetwork

from sample import FrameNumpy, SampleBatchNumpy
from collections import deque
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np

# 训练流程


buffer_size = 50000
batch_size = 32
episodes = 5000
initial_lives = 5
copy_steps = 10000
step = 0

# 超参数
stack_size = 4
lr = 5e-5
gamma = 0.98



EPSILON_MAX = 1.0                       # 初始最大 ε
CONST_EPS_FRAMES = 50000                # 保持 EPSILON_MAX 的帧数
EPS_FIRST_END = 0.1                     # 第一段衰减结束值
FRAMES_TO_FIRST_DECAY = 1_000_000       # 第一段衰减时长
EPS_FINAL = 0.01                        # 最终最小 ε
FRAMES_TO_FINAL_DECAY = 1_500_000       # 第二段衰减时长

epsilon = EPSILON_MAX


def get_epsilon(frame):
    """Piecewise linear epsilon schedule (frame is global step count)."""
    if frame < CONST_EPS_FRAMES:
        return EPSILON_MAX
    frame_rel = frame - CONST_EPS_FRAMES

    # first decay phase: EPSILON_MAX -> EPS_FIRST_END
    if frame_rel < FRAMES_TO_FIRST_DECAY:
        frac = frame_rel / FRAMES_TO_FIRST_DECAY
        return EPSILON_MAX + frac * (EPS_FIRST_END - EPSILON_MAX)  # linear interp
    frame_rel -= FRAMES_TO_FIRST_DECAY

    # second decay phase: EPS_FIRST_END -> EPS_FINAL
    if frame_rel < FRAMES_TO_FINAL_DECAY:
        frac = frame_rel / FRAMES_TO_FINAL_DECAY
        return EPS_FIRST_END + frac * (EPS_FINAL - EPS_FIRST_END)
    # after both decays, stick to EPS_FINAL
    return EPS_FINAL






env = GymEnv('BreakoutDeterministic-v4')
state_dim = env.state_dim
action_dim = env.action_dim

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

target_size = 84
state_dim = (stack_size, target_size, target_size)

# current Q网络
model = QNetwork(state_dim, action_dim, lr = lr)
# target Q网络
target_model = QNetwork(state_dim, action_dim, lr = lr)
target_model.to(conf['device'])
# 将模型设置到agent中
agent.set_model(model)


def update_target_model(target_model, model=model):
    target_model.load_state_dict(model.state_dict())
    # print("Target model updated.")
    return


def rgb_obs_process_torch_crop(x, crop_top=34, crop_left=0, crop_height=160, crop_width=160):
    """
    输入: np.uint8 RGB (H,W,3)
    输出: torch.tensor (crop_height, crop_width), float32, [0,1]，已搬到 GPU
    """
    device = conf['device']

    # 灰度 + 归一化
    x = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
    gray = 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]  # (H,W)
    # 裁剪
    gray_cropped = gray[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width]  # (crop_height, crop_width)
    gray_resized = F.interpolate(gray_cropped.unsqueeze(0).unsqueeze(0), size=(84, 84), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    
    return gray_resized

train_returns = []
test_returns = []
replay_buffer = deque(maxlen = buffer_size) # 样本池

from tqdm import tqdm

# 创建进度条实例
pbar = tqdm(range(episodes), ncols=130)


replay_buffer = deque(maxlen=buffer_size)

for episode in pbar:
    ret = 0
    obs = env.reset()
    done = False
    info = None
    current_lives = initial_lives
    start = True
    old_step = step
    # 创建 deque 存最近 stack_size 帧
    frame_queue = deque(maxlen=stack_size)

    # 初始化 frame_queue
    gray_frame = rgb_obs_process_torch_crop(obs).cpu().numpy()    # (84,84)

    for _ in range(stack_size): # init stack
        frame_queue.append(gray_frame)

    obs_stack = np.stack(list(frame_queue), axis=0)  # shape: (C, H, W)
    while not done:
        step += 1
        if step % copy_steps == 0:
            update_target_model(target_model, agent.model)

        epsilon = get_epsilon(step)
        agent.epsilon = epsilon

        if start or (info and 'lives' in info and info['lives'] < current_lives):
            if not start:
                current_lives = info['lives']
            start = False
            with torch.no_grad():
                action = 1
        else:
            # 堆叠最近 stack_size 帧作为输入
            action = agent.predict(obs_stack)

        next_obs, reward, done, info = env.step(action)
        ret += reward
        # 处理 next_obs 并存入队列以形成下一个状态
        gray_frame_next = rgb_obs_process_torch_crop(next_obs).cpu().numpy()  # (84,84)
        # 将下一帧加入队列
        frame_queue.append(gray_frame_next)
        next_obs_stack = np.stack(list(frame_queue), axis=0)  # shape: (C, H, W)

        # print("obs_stack:", obs_stack.shape, "next_obs_stack:", next_obs_stack.shape)
        # 存入 replay buffer
        sample = FrameNumpy.from_dict({
            'obs': obs_stack,
            'next_obs': next_obs_stack,
            'action': action,
            'reward': reward,
            'done': done
        })
        replay_buffer.append(sample)

        obs_stack = next_obs_stack

        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch = SampleBatchNumpy.stack(batch)
            agent.sample_process(batch)
            loss = agent.learn(batch, target_model)

    train_returns.append((episode, ret))
    pbar.set_postfix(reward=ret, loss=loss,epsilon=agent.epsilon,step=step,step_p_epi=step - old_step)
    
    if episode % 100 == 0:
        ret = 0
        obs = env.reset()
        done = False
        info = None
        current_lives = initial_lives
        start = True
        # 创建 deque 存最近 stack_size 帧
        frame_queue = deque(maxlen=stack_size)
        gray_frame = rgb_obs_process_torch_crop(obs).cpu().numpy()    # (84,84)

        for _ in range(stack_size): # init stack
            frame_queue.append(gray_frame)

        obs_stack = np.stack(list(frame_queue), axis=0)  # shape: (C, H, W)
        while not done:
            if start or (info and 'lives' in info and info['lives'] < current_lives):
                if not start:
                    current_lives = info['lives']
                start = False
                with torch.no_grad():
                    action = 1
            else:
                action = agent.exploit(obs_stack) # 最优动作

            next_obs, reward, done, info = env.step(action)
            ret += reward
            # 处理 next_obs 并存入队列以形成下一个状态
            gray_frame_next = rgb_obs_process_torch_crop(next_obs).cpu().numpy()  # (84,84)
            # 将下一帧加入队列
            frame_queue.append(gray_frame_next)
            next_obs_stack = np.stack(list(frame_queue), axis=0)  # shape: (C, H, W)
            obs_stack = next_obs_stack
        assert current_lives == 1, "current_lives: " + str(current_lives)
        test_returns.append((episode, ret))
        print(f"Evaluation at episode {episode}, return: {ret}")




plt.plot([x[0] for x in train_returns], [x[1] for x in train_returns], label = 'train')
plt.plot([x[0] for x in test_returns], [x[1] for x in test_returns], label = 'test')
plt.legend()
plt.title("Breakout")
plt.savefig(f"./breakout_results/nohup_Breakout_lr{lr}_epsilon{epsilon}_gamma{gamma}_buffer{buffer_size}_batch{batch_size}_update{copy_steps}.png")
plt.show()