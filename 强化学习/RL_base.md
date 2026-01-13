# 强化学习期末复习

## 1. 强化学习基本概念

### Bandit问题

1. 乐观初始值法：将初值设得较高，促使算法多尝试不同动作。


2. UCB:
$$
A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

3. Gradient Bandit:
$$
Pr(A_t=a)=\frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}=\pi_t(a); \\
H_{t+1}(A_t) = H_t(A_t) + \alpha (R_t - \bar{R_t})(1 - \pi_t(A_t)) \\
H_{t+1}(a) = H_t(a) - \alpha (R_t - \bar{R_t})\pi_t(a), a \neq A_t
$$
即先计算每个动作选择的可能性，再根据优势调整每个动作的偏好值。

### MDP

在设定中为$(S_0, A_0, R_1, S_1)$

在 Pole-Balancing 中，状态可以是小车位置、速度、杆子角度和角速度的组合。reward 可以是每个时间步的 +1，直到杆子倒下或小车出界。也可以是：每次失败为 -1，其他时间为 0。


$$
v_{\pi}(s) = \mathbb{E}_{\pi} [G_t | S_t = s] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s \right] \\
and \\
q_{\pi}(s, a) = \mathbb{E}_{\pi} [G_t | S_t = s, A_t = a] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s, A_t = a \right]
$$

此外:

$$
v_{\pi}(s) = \sum_a \pi(a|s) q_{\pi}(s, a) \\
q_{\pi}(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')]
$$

**Bellman Equation**
$$
v_{\pi}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')]
$$
Bellman Optimality Equation:
$$
v_{*}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_{*}(s')] \\
q_{*}(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} q_{*}(s', a')]
$$


最优值函数是唯一的，但是可以有多个最优策略。
对于有限 MDP Bellman Optimality Equation 有唯一解。但是大部分问题是无限的。所以需要近似。


### Dynamic Programming

#### Value Update(Evaluation):
通过反复迭代计算一个给定的策略下能达到的总收益
$$
v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]
$$
对所有的 s 依次更新，直到收敛。（更新保证：$\gamma <1$）
**Synchronous Update**：所有的状态同时更新。
**Asynchronous Update**：更新 $v_{k+1}(s_2)$ 使用到 $v_{k+1}(s_1)$

#### Policy Improvement:
$$
\pi'(s) = \arg\max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}(s')]
$$
在状态 s 贪心地选择动作 a，可以保证 $v_{\pi'}(s) \geq v_{\pi}(s)$

#### Policy Iteration:
交替进行 Policy Evaluation 和 Policy Improvement，直到策略不再改变。

**for action values**
$$
q_{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma q_{k}(s', \pi(s'))]
$$

问题：
1. 策略估值需要对所有状态进行评估，计算量大。
2. 在得到最优策略后，可能还在更新值函数

#### Value Iteration:

由于在选择 $\pi$ 的时候只考虑 V 的最大值，可以直接在更新 V 的时候进行贪心选择。

将 Policy Evaluation 和 Policy Improvement 合并：
$$
v_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]
$$
$$
\pi(s) = \arg\max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_{k}(s')]
$$
action value 版本：
$$
q_{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} q_{k}(s', a')]
$$

**异步 DP**

异步地更新部分状态的值函数，可以节省计算资源。一些状态值可能已经回溯多次，而其他的连一次也没有。
收敛条件：1. $0 \leq \gamma<1$; 2. 每个状态都要被更新无限次。

价值迭代和策略迭通常无法直接比较收敛速度，因为两者的迭代方式不同。对初始值依赖都很大、不能区分哪个更好，收敛都比理论快，值迭代的策略优化次数更多。

**广义策略迭代 (Generalized Policy Iteration)**

可以被看做是：贪心选择动作产生新策略；重新计算值函数使得和策略一致。


## 2. 强化学习基本方法


### Monte Carlo Methods

基于采样的策略评估和改进方法。适用于无法获得完整环境模型的情况。
如果没有模型 P ，对于随机环境，最好估计 Q ，这样就可以直接选择策略。
为了更好的，我策略们需要尝试不同的动作。(随机起步)
例如如果使用确定性策略， $s_0$ 和 $a_0$ 需要随机选择。


**epsilon-greedy 和 epsilon-soft**

$\epsilon-soft$: 每个动作都有至少 $\epsilon/|A(s)|$ 的概率被选择。
$\epsilon$-greedy 是 $\epsilon$-soft 的，因为贪心动作的概率为 $1-\epsilon + \epsilon/|A(s)|$。



通过 $G\leftarrow \gamma G + R$ 计算回报。减少开销
更新 Q:
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [G_t - Q(S_t, A_t)], 
$$
$\alpha$ 可以是 $1/N(S_t, A_t)$

如果使用随机性策略，不需要探索性开始


**Off-policy**
往往有更大的方差
目标策略：$\pi$，行为策略：$\mu$
使用重要性采样：
$$v_{\pi}(s) = \mathbb{E}_{\mu} [\rho_{t:T-1} G_t | S_t = s]$$
$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)p(S_{k+1}|S_k, A_k)}{\mu(A_k|S_k)p(S_{k+1}|S_k, A_k)}$$

*Incremental off-policy(every-visit)*

Evaluation：
$$
C_n(S_t, A_t) = C_{n-1}(S_t, A_t) + W \\
Q_n(S_t, A_t) = Q_{n-1}(S_t, A_t) + \frac{W}{C_n(S_t, A_t)} [G_n - Q_{n-1}(S_t, A_t)] \\
W = W \frac{\pi(A_k|S_k)}{\mu(A_k|S_k)}
$$

**Control：这里的目标是确定性的**
Generate episode using $\mu$
Set $G=0$, $W=1$
t=T-1, T-2, ..., 0
$$
C(S_t, A_t) = C(S_t, A_t) + W \\
Q(S_t, A_t) = Q(S_t, A_t) + \frac{W}{C(S_t, A_t)} [G_t - Q(S_t, A_t)] \\
\pi(S_t) = \arg\max_a Q(S_t, a) 
$$
If $A_t \neq \pi(S_t)$, then break; else: $W = W \times \frac{1}{\mu(A_t|S_t)}$ and loop

即如果最优动作和行为动作不一致，就停止更新。

> off-policy 的重要性采样会有较大的方差，同时收敛较慢。

### Temporal-Difference Learning

Monte Carlo 学习时可能会有比较大的抖动，因为每次更新都使用完整的回报 G 。TD 学习使用部分回报进行更新，减少了方差。

TD 比 Monte Carlo 更加高效，因为不需要等到回报 G 计算完成后才能更新。用到了 Bellman 方程在不同采样路径的传递估值。

#### SARSA

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$



#### Q-Learning
是一种 off-policy 方法，使用最大化的动作值进行更新，而不是实际采取的动作值。

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**Expected SARSA**
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]
$$
当 $\pi$ 为贪心策略时，Expected SARSA 等价于 Q-Learning。


估值会偏乐观，因为使用了最大值。如果某一次估计偏高，那么后续的估计也会被拉高。

**Double Q-Learning**
使用两个独立的 Q 函数来减少估值偏差。更新时随机选择一个 Q 函数进行更新。
$$
Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_2(S_{t+1}, \arg\max_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t)] \\
Q_2(S_t, A_t) \leftarrow Q_2(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_1(S_{t+1}, \arg\max_a Q_2(S_{t+1}, a)) - Q_2(S_t, A_t)]
$$
行为策略可以是 $\epsilon$-greedy ，基于 $Q_1 + Q_2$ 。也可以随机选择一个 Q 函数。


#### 对比 TD 和 Monte Carlo

TD 有偏差，但方差较小，收敛更快。Monte Carlo 无偏差，但方差较大，收敛较慢。
动态规划不需要采样，MC 不需要 bootstrap，TD 需要采样和 bootstrap。

### N-step TD

#### Evaluation:

注意：在探索时选择 $A_0$ 需要是随机的，否则可能无法覆盖所有状态。
$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V_{k}(S_{t+n}) \\
V_{k+1}(S_t) = V_{k}(S_t) + \alpha [G_{t:t+n} - V_{k}(S_t)]
$$
如果最后的状态数目不足 n ，则使用实际的回报计算。

**n-step SARSA**
$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n Q_{k}(S_{t+n}, A_{t+n}) \\
Q_{k+1}(S_t, A_t) = Q_{k}(S_t, A_t) + \alpha [G_{t:t+n} - Q_{k}(S_t, A_t)]
$$

**Importance Sampling (n-step version)**
$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^{n} V_{k}(S_{t+n}) \\
\rho_{t:t+n} = \prod_{k=t}^{\min(t+n,T-1)} \frac{\pi(A_k|S_k)}{\mu(A_k|S_k)} \\
V_{k}(S_t) = V_{k-1}(S_t) + \alpha \rho_{t:t+n-1} [G_{t:t+n} - V_{k-1}(S_t)]
$$

#### Tree-backup

$$
G_{t:t+n} = R_{t+1} + \gamma \sum_{a\neq A_{t+1}} \pi(a|S_{t+1})Q(S_{t+1}, a) +\gamma \pi(A_{t+2}|S_{t+2}) G_{t+1:t+n}
$$

#### Q($\sigma$)
将上述的几种算法使用一个参数 $\sigma$ 进行统一表示：
$$
G_{t:h} = R_{t+1} + \gamma \left[ \sigma \rho_{t+1} + (1-\sigma) \pi(A_{t+1}|S_{t+1}) \right] (G_{t+1:h}-Q(S_{t+1}, A_{t+1})) \\
+\gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a)
$$
$\sigma=1$ 代表完全采样，使用样本中的数据更新
$$
G_{t:h} = R_{t+1} + \gamma \rho_{t+1} (G_{t+1:h}-Q(S_{t+1}, A_{t+1})) + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a)
$$
$\sigma=0$时：不采样只使用期望。完全依赖 Tree backup。
$$
G_{t:h} = R_{t+1} + \gamma \pi(A_{t+1}|S_{t+1}) G_{t+1:h} + \gamma \sum_{a\neq A_{t+1}} \pi(a|S_{t+1})Q(S_{t+1}, a)
$$



### Planning and Learning with Tabular Methods
简单的 Model-based RL 方法，使用环境模型进行规划和学习。
#### Dyna-Q
结合了直接学习和规划的思想。通过与环境交互收集数据，更新值函数。同时使用一个环境模型进行规划，模拟更多的经验来更新值函数。
1. 与环境交互，选择动作 $A_t$，观察奖励 $R_{t+1}$ 和下一个状态 $S_{t+1}$。
2. 使用观察到的转移 $(S_t, A_t, R_{t+1}, S_{t+1})$ 更新 Q 值。
3. 更新环境模型，记录转移概率和奖励函数。Model$(S_t, A_t) \leftarrow (R_{t+1}, S_{t+1})$
4. 进行多次规划迭代:
   1. 从模型中**采样虚拟转移**, 使用之前到过的状态和动作。
   2. 更新 Q 值。

根据 Q 值，可以得到我们选择动作的策略。

#### Dyna-Q+
在 Dyna-Q 的基础上，增加了对未被频繁访问的状态-动作对的奖励激励，鼓励探索。
$$
R'(s, a) = R(s, a) + \kappa \sqrt{\tau(s, a)}
$$
其中 $\tau(s, a)$ 是自上次访问以来的时间步数，$\kappa$ 是一个小的正数，控制探索奖励的大小。可以处理变化的环境。

**优先扫描 priority sweeping**
维护一个优先级队列，存储需要更新的状态-动作对。每次从队列中选择优先级最高的对进行更新，并根据更新结果调整相关状态-动作对的优先级。
因为有用的状态-动作对是会使 agent 的估值发生变化的对。
$$
P = |R + \gamma \max_a Q(s', a) - Q(s, a)|
$$
更新时，每次在优先队列中选择优先级最高的状态-动作对(S, A)，更新 Model 和 Q 值。然后对所有可能导致 S 的前驱状态-动作对 $(\bar{s}, \bar{a}) $预测 $\bar{R}$，计算 $\bar{P}=|\bar{R}+\gamma \max_a Q(\bar{s}', a) - Q(\bar{s}, \bar{a})|$，并根据阈值决定是否加入优先级队列。
优先采样可以明显加快收敛速度。

#### MCTS
蒙特卡洛树搜索（MCTS）是一种用于决策过程的启发式搜索算法，特别适用于大型状态空间和不确定性环境。MCTS 通过构建一个搜索树，使用随机采样来评估不同动作的潜在收益，从而指导决策。
MCTS 的主要步骤包括选择、扩展、模拟和回传：
1. 选择（Selection）：从根节点开始，根据某种策略（如 UCT）选择子节点，直到到达一个未完全展开的节点。
2. 扩展（Expansion）：如果选择的节点不是终止状态，则扩展该节点，添加一个或多个子节点。
3. 模拟（Simulation）：从新扩展的节点开始，进行随机模拟（playout），直到达到终止状态，记录结果。
4. 回传（Backpropagation）：将模拟结果回传到树中的所有经过节点，更新它们的统计信息（如访问次数和累计奖励）。

## 3. 基于函数逼近的强化学习方法

### 3.1 随机梯度下降 SGD
对 MSE 损失函数进行最小化:
$$
\theta_{k+1} = \theta_k + \alpha [U_t - \hat{v}(S_t, \theta_k)] \nabla_{\theta} \hat{v}(S_t, \theta_k)
$$
其中 $U_t$ 是目标值，可以是 Monte Carlo 回报，TD 目标等（无偏估计）。
例如 TD(0): $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_k)$。由于只对一个 $\hat{v}$ 求导，所以是**半梯度**方法。

**线性函数**：
$$\hat{v}(s, \theta) = \theta^T \phi(s) = \sum_{i=1}^{n} \theta_i \phi_i(s)$$
其中 $\phi(s)$ 是状态 s 的特征向量表示。
$$
\theta_{k+1} = \theta_k + \alpha [U_t - \theta_k^T \phi(S_t)] \phi(S_t)
$$

#### 特征构造

1. **多项式特征**：使用状态变量的多项式组合来构造特征。
2. **Coarse 特征**：将状态空间划分为多个区域，每个区域对应一个二进制特征。具体来说，如果状态 s 落在某个区域内，则对应的特征值为 1，否则为 0。
3. **Tiling 编码**：将状态空间覆盖多个重叠的网格，每个网格对应一组二进制特征。通过多个网格的组合，可以更细致地表示状态。例如：在二维空间中，可以使用多个不同偏移的网格来覆盖状态空间，每个网格的单元格对应一个特征。如果状态落在某个单元格内，则对应的特征值为 1，否则为 0。

### 3.2 半梯度 SARSA
$$\theta_{k+1} = \theta_k + \alpha [R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \theta_k) - \hat{q}(S_t, A_t, \theta_k)] \nabla_{\theta} \hat{q}(S_t, A_t, \theta_k)$$

**n-step 半梯度 SARSA**
$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \theta_k) \\
\theta_{k+1} = \theta_k + \alpha [G_{t:t+n} - \hat{q}(S_t, A_t, \theta_k)] \nabla_{\theta} \hat{q}(S_t, A_t, \theta_k)
$$

### 3.3 平均回报
对于无终止状态的任务，使用平均回报作为目标。衡量策略在长期稳定后每一步能得到多少回报。
$$r(\pi) = \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E}[R_t | S_0, A_{0:t-1}\sim\pi] \\
= \lim_{t \to \infty} \mathbb{E}[R_t |A_{0:t-1}\sim\pi]
$$

遍历性：到达一个状态只依赖于策略和 MDP 转移概率，保证方程极限存在(与初始状态无关)。
稳态分布 $\mu_{\pi}(s)$：在策略 $\pi$ 下，选择动作后仍然回到该状态的概率分布。
$$
\sum_s\mu_{\pi}(s)\sum_a \pi(a|s)p(s'| s, a)  = \mu_{\pi}(s')
$$

使用平均回报的 Bellman 方程:
$$v_\pi(s) = \sum_{t=0}^{\infty} \mathbb{E} [ R_{t+1} - r(\pi) | S_0=s ]$$
$$q_\pi(s,a) = \sum_{s', r} p(s', r | s, a) (r - r(\pi) + v_\pi(s'))$$

#### 差分回报：
$$
G_t = R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + ... = \sum_{k=0}^{\infty} [R_{t+k+1} - r(\pi)]
$$
差分值可以满足 Bellman 方程; TD 差分：(也可以使用 $q$ 函数)$\bar{R}$ 是平均回报的估计。
$$\delta_t = R_{t+1} - \bar{R} + \hat{v}(S_{t+1}, \theta_k) - \hat{v}(S_t, \theta_k) \\
\theta_{k+1} = \theta_k + \alpha \delta_t \nabla_{\theta} \hat{v}(S_t, \theta_k) $$


#### Deprecating the Discounted Setting
$$
J(\pi) = \sum_s \mu_{\pi}(s) v_{\pi}^{\gamma}(s) \\
= \sum_s \mu_{\pi}(s) \sum_a \pi(a|s)\sum_{s', r} p(s', r | s, a) [r + \gamma v_{\pi}^{\gamma}(s')] \\
= r(\pi) + \gamma\sum_{s'}\mu_{\pi}(s')v_{\pi}^{\gamma}(s')= \dots \\
=\frac{1}{1-\gamma} r(\pi)
$$
**n 步差分半梯度 SARSA**
$$
\delta = \sum_{i=\tau+1}^{\tau+n}(R_{i}- \bar{R}) + \hat{q}(S_{\tau+n}, A_{\tau+n}, \theta_k) - \hat{q}(S_{\tau}, A_{\tau}, \theta_k) \\
\theta_{k+1} = \theta_k + \alpha \delta \nabla_{\theta} \hat{q}(S_{\tau}, A_{\tau}, \theta_k) \\
\bar{R} \leftarrow \bar{R} + \beta \delta
$$
其中 $\alpha$ 是学习率，$\beta$ 是平均回报的步长参数。


### 3.4 TD($\lambda$)
1. 无限时域定义：
$$
G_{t}^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}
$$
2. 对于有限环境:可以检验系数之和为 1 :
$$
G_{t}^{\lambda} = (1-\lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_{t:T}
$$
通过代数变形后可以得到：
$$
G_{t}^{\lambda}-V(S_t) = \sum_{n=0}^{\infty} (\lambda \gamma)^{n} \delta_{t+n}\\
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$
资格迹(在 semi-gradient 的情境下加速计算):
$$z \leftarrow \gamma \lambda z + \nabla_{\theta} \hat{v}(S_t, \theta_k) \\
\delta \leftarrow R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_k) - \hat{v}(S_t, \theta_k) \\
\theta_{k+1} = \theta_k + \alpha \delta z
$$
最后保证对参数更新的总量是不变的。


## 4. 深度强化学习

### 4.1 Deep Q-Network (DQN)

**Fitted Q iteration(offline)**
使用经验回放缓冲区存储过去的转移样本 $(S_t, A_t, R_{t+1}, S_{t+1})$。在每次训练迭代中，从缓冲区中随机采样一个小批量的样本，计算目标 Q 值，并使用均方误差损失函数更新神经网络参数。

**Online Q learning**
使用 epsilon-greedy 策略选择动作，既利用当前的 Q 网络，又进行探索。随着训练的进行，逐渐减少 epsilon 的值，以增加利用的比例。
Q-learning 不是梯度下降方法，而是半梯度方法。

#### 对 Q -learning 的改进

**目标网络**
使用一个独立的目标网络 $ \phi$ 来计算**目标 Q 值**，定期将主网络的参数复制到目标网络中。这有助于稳定训练过程，减少目标值的变化。
$$
\theta \leftarrow \theta + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \phi) - Q(S_t, A_t; \theta)] \nabla_{\theta} Q(S_t, A_t; \theta)
$$

**Double DQN**
使用两个独立的网络来减少过度估计偏差。主网络用于选择动作，目标网络用于计算目标 Q 值。
$$
\theta_1 \leftarrow \theta_1 + \alpha [R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_1); \theta_2) - Q(S_t, A_t; \theta_1)] \nabla_{\theta_1} Q(S_t, A_t; \theta_1)
$$

**Dueling DQN**
*改进点1*： 将 Q 值分解为状态值函数 V 和优势函数 A 的组合：
$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|A|} \sum_{a'} A(s, a'; \theta, \alpha) \right)
$$
$\theta$ 是共享的卷积层参数，$\alpha$ 和 $\beta$ 分别是优势函数和状态值函数的参数。
在优化时可以促使 $\frac{1}{|A|} \sum_{a'} A(s, a'; \theta, \alpha)$ 接近 0.从而保证不会收敛到 A=Q，V=0 的次优解。

改进点2：优先经验回放（Prioritized Experience Replay, PER）
根据 TD 误差的大小对经验进行优先级排序，更频繁地采样那些 TD 误差较大的经验，从而加快学习速度。$p_i \propto |\delta_i| + \epsilon$
但是优先经验回放池会引入样本相关性，调整各个样本的学习率权重：
$$
\alpha_i \leftarrow \alpha (n \cdot p_i)^{-\beta}
$$
具有较高优先级的样本会有较低的学习率权重，防止过拟合。

**Rainbow**
将 DQN 的多个改进方法结合在一起，包括 Double DQN、Dueling network、优先经验回放、n-step TD、Distributional RL、Noisy Nets
进一步提升了性能。

### 4.2 Policy Gradient 方法

#### 策略梯度定理
策略梯度定理(证明略)表明，策略的梯度可以表示为（这里的样本与当前的 policy 相关）：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau\sim\pi_{\theta}} \left[ \sum_{t=0}^\infin\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) r(\tau) \right]
$$
也可以写为：
$$
\nabla J(\theta) = \sum_s \mu(s) \sum_a \nabla_{\theta} \pi_{\theta}(a|s) Q^{\pi_{\theta}}(s, a)
$$

#### REINFORCE

REINFORCE 算法使用蒙特卡洛方法估计 Q 值，直接使用回报 $G_t$ 作为 Q 的无偏估计：$G_t$ 代表 reward-to-go。
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \gamma^t G_t \frac{\nabla_{\theta} \pi_{\theta}(a|s)}{\pi_{\theta}(a|s)} \right] \\
\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_{\theta} \log \pi_{\theta}(a|s)
$$
**With Baseline**
为了减少方差，可以引入一个基线函数 $b(s)$，通常选择为状态值函数 $V^{\pi}(s)$,仍然是无偏估计。
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a|s) (Q^{\pi_{\theta}}(s, a) - b(s)) \right] \\
= \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a|s) Q^{\pi_{\theta}}(s, a) \right]- \sum_s \mu(s) b(s) \nabla_{\theta}\left(\sum_a  \pi_{\theta}(a|s)\right)\\
= \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a|s) Q^{\pi_{\theta}}(s, a) \right]
$$



#### 针对连续性策略近似

**基于平均回报**
$$
J(\theta) = r(\pi) = \sum_s \mu^{\pi}(s) \sum_a \pi(a|s) \sum_{s',r} p(s', r | s, a) r
$$
于是策略梯度为：
$$ \nabla J(\theta) = \nabla r(\pi) = \sum_{s} \mu_{\pi}(s) \sum_{a} q_{\pi}(s, a) \nabla_{\theta} \pi(a|s, \theta) $$

推导过程：
>Q 的估计为：
$$q_\pi(s,a) = \sum_{s', r} p(s', r | s, a) (r - r(\pi) + v_\pi(s'))$$
其中，$v_\pi(s) = \sum_a \pi(a|s) q_\pi(s,a)$。
$$\nabla v_\pi(s) = \sum_a \left[ \nabla \pi(a|s) q_\pi(s,a) + \pi(a|s) \nabla q_\pi(s,a) \right]$$
>计算 $\nabla q_\pi(s,a)$。根据上面 $q$ 的定义：
$$\nabla q_\pi(s,a) = \nabla \left( \sum_{s', r} p(s', r | s, a) (r - r(\pi) + v_\pi(s')) \right)$$
这里的 $p$（环境动态）和 $r$（奖励）与策略参数 $\theta$ 无关，有关的是 $r(\pi)$ 和 $v_\pi(s')$：
$$\nabla q_\pi(s,a) = \sum_{s'} p(s'|s,a) \left( -\nabla r(\pi) + \nabla v_\pi(s') \right)$$
$$= -\nabla r(\pi) + \sum_{s'} p(s'|s,a) \nabla v_\pi(s')$$
带入 $\nabla v_\pi(s)$ 的表达式：
$$ \nabla r(\pi) = \sum_a \nabla \pi(a|s) q_\pi(s,a) + \sum_a \pi(a|s) \sum_{s'} p(s'|s,a) \nabla v_\pi(s') - \nabla v_\pi(s)$$
>为了消除难以计算的 $\nabla v_\pi$，我们对等式两边同时乘以平稳分布 $\mu_\pi(s)$ 并对所有状态 $s$ 求和。
**左边：**
    $$\sum_s \mu_\pi(s) \nabla r(\pi) = \nabla r(\pi) \sum_s \mu_\pi(s) = \nabla r(\pi) \cdot 1 = \nabla r(\pi)$$
**右边：**
    $$ \underbrace{\sum_s \mu_\pi(s) \sum_a \nabla \pi(a|s) q_\pi(s,a)}_{\text{目标项}} + \underbrace{\sum_s \mu_\pi(s) \sum_a \pi(a|s) \sum_{s'} p(s'|s,a) \nabla v_\pi(s')}_{\text{项 A}} - \underbrace{\sum_s \mu_\pi(s) \nabla v_\pi(s)}_{\text{项 B}} $$
**项 A**：我们交换求和顺序，先看括号里的部分：
$$\text{项 A} = \sum_{s'} \left( \sum_s \mu_\pi(s) \sum_a \pi(a|s) p(s'|s,a) \right) \nabla v_\pi(s')$$
括号里的公式 $\sum_s \mu_\pi(s) \sum_a \pi(a|s) p(s'|s,a)$ 正是**平稳分布的定义**（从所有 $s$ 转移到 $s'$ 的概率总和），它就等于 $\mu_\pi(s')$。
所以：
$$\text{项 A} = \sum_{s'} \mu_\pi(s') \nabla v_\pi(s')$$
发现 **项 A** 和 **项 B** 是一模一样的（只是求和的变量符号一个是 $s'$ 一个是 $s$）。因此，**项 A - 项 B = 0**。

**动作建模**

对于连续动作空间，策略可以建模为参数化的概率分布（如高斯分布）：
$$
\pi_{\theta}(a|s) = \mathcal{N}(\mu_{\theta}(s), \sigma_{\theta}^2(s))
$$

### 4.3 Actor-Critic 方法
与 REINFORCE 相同，这也是严格的 on-policy 方法。

相比于 REINFORCE，Actor-Critic 方法使用一个Critic来估计动作价值函数 $Q^{\pi}(s, a)$ 或状态价值函数 $V^{\pi}(s)$，从而减少方差并提高学习效率。
基本的 Actor-Critic 方法使用 TD 作为价值的估计(理解为加上了 baseline)：
$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \delta_t \frac{\nabla_{\theta} \pi_{\theta}(a|s)}{\pi_{\theta}(a|s)} \right] \\
\delta_t = R_{t+1} + \gamma V(s_{t+1}; w) - V(s_t; w)
$$

Actor 更新：
$$
\theta \leftarrow \theta + \alpha \delta_t \nabla_{\theta} \log \pi_{\theta}(a|s)
$$
Critic 更新：
$$
w \leftarrow w + \beta \delta_t \nabla_w V(s; w)
$$

#### Advantage Actor-Critic (A2C)
使用优势函数，相比于REINFORCE，不使用 MC ，使用单步 TD 或 n-step return 估计 Q 值(直观可以看做是 REINFORCE with baseline 对 G 进行估计)：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \delta_t \frac{\nabla_{\theta} \pi_{\theta}(a|s)}{\pi_{\theta}(a|s)} \right]
$$

**解决 off-policy 问题的方法**
如果使用 replay buffer 存储过去的经验样本，采样时需要处理。
对于 Actor-Critic 方法，从回放缓冲区中采样的经验 $(s_i, a_i, r_i, s'_i)$，仅仅使用 $s_i$，使用当前的策略估计 $a^{\pi}_i$。使用当前的 Critic 估计 $Q(s_i, a_i)$。
更新 actor 时:
$$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} Q(s_i, a^{\pi}_i) \frac{\nabla_{\theta} \pi_{\theta}(a^{\pi}_i|s_i)}{\pi_{\theta}(a^{\pi}_i|s_i)}$$
更新 critic 时: 这里的 $a'_i$ 是使用当前策略计算得到的动作。
$$w \leftarrow w + \beta \left( r_i + \gamma Q(s'_i, a'_i; w) - Q(s_i, a_i; w) \right) \nabla_w Q(s_i, a_i; w)$$

### 4.4 PPO, TRPO, DDPG, GAE
（略）

## 5. Distributed RL

### 5.1 分类

1. 数据并行
2. 模型并行
   1. 流水并行
   2. 张量并行（将模型同一层的矩阵拆分到多个不同的 GPU）
3. MoE：每一次前向传播只计算模型中的一部分参数的梯度

### 5.2 流水并行

GPipe：将模型划分为多个阶段，每个阶段放在不同的设备上。将输入数据划分为多个微批次（micro-batch），每个微批次依次通过各个阶段。当所有 micro-batch 都前向传播完成后，从第一个 micro-batch 开始反向传播计算梯度。
**PipeDream**：在 GPipe 的基础上，在第一个 micro-batch 完成前向传播后，开始对第一个 micro-batch 进行反向传播，同时继续前向传播后续的 micro-batch。这样可以进一步提高设备利用率。

### 5.3 分布式 RL

#### A3C：Asynchronous Advantage Actor-Critic
多个 actor 在不同的环境实例中并行采样数据，计算好梯度并将**梯度**发送给一个全局的参数服务器进行更新。每个 actor 有自己的本地网络，定期将本地网络的参数同步到全局网络。

#### IMPALA

Actor 只被设计来采集**数据**，随后将其传给 Learner，由 Learner 来计算梯度。采用异步方式使得 Learner 能够持续更新

#### SEED RL
Actor 给 learner 发送观测，Learner **推理**，确保模型参数和状态不出本地。

#### DD-PPO
所有Worker既是采样节点，也是训练节点，
每一个Worker都会经过下面几个步骤： （1）跟环境交互，采样并收集训练样本；（2）收集到足够样本之后，计算模型梯度；（3）所有的Worker进行一次全规约(All reduce)操作，得到更新后的模型。

## 6. Self-Play and Multi-Agent Reinforcement Learning

### 6.1 基础知识

非完美信息博弈：玩家无法完全观察到环境状态或其他玩家的动作。
无法讨论最优解，而是讨论均衡解，如纳什均衡。
同步决策和异步决策。

### 6.2 自对弈 (Self-Play)

#### Fictitious Play
反复进行对局，记录对手历史平均策略，针对该平均策略进行最优响应。
如果对手策略不变，可以得到最优解。
如果所有玩家都采用该策略，每个玩家可以收敛到纳什均衡。

#### Neural Fictitious Self-Play (NFSP)
结合了神经网络和虚构自对弈的思想。每个智能体维护两个神经网络：一个用于学习最佳响应策略——策略网络或Q网络；另一个用于学习平均策略——一定是策略网络。

#### Double Oracle
初始时只考虑所有可能策略的一个子集。通过反复进行对局，找到当前策略集合中的纳什均衡，然后计算每个玩家的最佳响应策略，并将**最佳响应策略**添加到策略集合中。重复该过程，直到没有新的最佳响应策略可以添加为止。

#### Policy Space Response Oracles (PSRO)
为每个玩家维护一个策略集合。通过反复进行对局，计算当前策略集合中的**元策略**，然后为每个玩家计算**最佳响应策略**，并将其添加到策略集合中。使用深度强化学习方法来近似计算最佳响应策略。

**选择对手的方式**
1. naive self-play：每次都与当前最新的模型对战。
2. heterogeneous self-play：从人为定义选择一个对手进行对战。
3. delta-uniform self-play：随机选择最新的若干模型作为对手
4. prioritized self-play：历史模型中表现越好的选择的概率越大。
5. population-based self-play：维护多个种群，采用不同的训练方式和对手选择策略。

### 6.3

#### CTDE 框架

纯合作问题：算法：VDN, QMIX，COMA，所有agent共享相同的reward
合作-竞争问题：MADDPG
非完美信息问题：PID

**Valuation Decomposition Networks (VDN)**
$$
Q_{tot}(\tau, a) = \sum_{i=1}^{N} Q_i(\tau^i, a^i)
$$
决策时对每个 agent 使用自身的 Q 函数选择动作。

**QMIX**
QMIX 通过一个混合网络将各个 agent 的 Q 值组合成全局的 Q 值，保证单调性：

**Counterfactual Multi-Agent PG(COMA)**
使用 actor-critic 框架，Critic 估计全局 Q 值，Actor 使用**反事实基线**来计算优势函数：
$$A^i(s, a) = Q(s, a) - \sum_{a'^i} \pi^i(a'^i|s) Q(s, (a^{-i}, a'^i))$$
决策时每个 agent 使用自己的 Actor 选择动作。

**Multi-Agent DDPG (MADDPG)**
每个 agent 有自己的 Actor 和 Critic。Critic 使用所有 agent 的动作和状态作为输入，Actor 只使用自身的状态作为输入。相当于 PID 方法。

**Perfect Information Distillation (PID)**
actor 只接受局部观察，critic 接受全局状态。

### 6.4 游戏 AI 案例

#### 围棋

**AlphaGo**
* 第一阶段：人类数据监督学习一个复杂模型一个简单模型
* 第二阶段：使用 Policy Gradient 方法优化策略网络（从历史模型中随机选择对手）
* 第三阶段：使用强化学习阶段产生的对局数据，训练一个价值网络来估计局面胜率。

实战时使用 Monte Carlo Tree Search (MCTS) 结合策略网络和价值网络进行决策。

**AlphaGo Zero**
将策略网络和价值网络进行合并，使用 ResNet 结构。
训练过程使用 MCTS 进行自对弈，生成数据来训练网络。
（对手选择：历史模型中最好的模型）

**AlphaZero**
将 AlphaGo Zero 的方法推广到国际象棋和将棋等其他棋类游戏中。
区别：
1. 没有使用数据增强技术（如旋转、翻转棋盘）。
2. 只选择最新的模型作为对手进行自对弈。

#### 德州扑克

**AlphaHoldem**
使用 RL 得到了和之前的 CFR 方法相当的效果。
多个 Actor 和一个 learner 组成。对手选择：历史模型中最好的 k 个模型。
状态和动作都以图像的形式编码成特征

*Trinal-Clip PPO*
= Dual-Clip PPO + reward clip
Dual-Clip PPO: 对 ratio 额外增加一个**常数**下界限制，防止策略更新过大,$\beta>0$.
$$
L_t^{\text{Dual-Clip}}(\theta) =
\begin{cases} 
\min\big(r_t(\theta) \hat A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t\big), & \hat A_t > 0 \\[2mm]
\max\Big(\min\big(r_t(\theta) \hat A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t\big), \; \beta \hat A_t \Big), & \hat A_t < 0
\end{cases}
$$

在更新 value 时，对 reward 进行裁剪，防止过大波动。

#### 立直麻将

**Suphx**
1. 第一阶段：使用人类对局数据进行监督学习，训练 6 个初始策略网络。
2. 第二阶段：使用自强化学习优化*出牌模型*。对手选择：使用最新的模型进行对战。算法：PG with entropy，动态调整entropy weight。

针对非完美信息的处理方法：oracle guiding；在开始使用不可见信息，在训练过程中逐渐减少对不可见信息的依赖。
额外训练一个 global reward predictor 在当前这一下局下预测打完大局的得分。每一小局的 reward = 结束时预测的总分 - 这一局之前预测的总分。

在实际对局时，使用parametric Monte-Carlo PolicyAdaptation算法：每一小局开始时，**随机猜测**几种对手的手牌，并用网络自对弈跑完对局用这几局数据重新finetune策略网络。

**JueJong**
用 RL 解决双人麻将
对手选择：最新的模型进行对战

#### 斗地主

**DouZero**
使用 Deep Monte Carlo 方法进行自对弈，生成数据训练策略网络和价值网络。
分别使用 3 个网络对应地主和农民两个角色。
对手选择：最新的模型进行对战。

**PerfectDou**
算法：PPO + PID(Perfect Information Distillation)
对手模型选择最新的模型
使用 Reward Shaping 技术，设计了多种奖励函数来引导训练过程。


#### 游戏

**AlphaStar**
* 第一阶段：使用监督数据训练 3 个初始模型
* 使用强化学习+self-play

算法：
1. TD($\lambda$)
2. UPGO (Upgoing policy update)：只使用采用数据中比平均表现更好的样本进行策略更新。
3. V-trace: 处理模型参数不同步的问题。

种群之间自对弈：
* main agent: 从所有种群中采样对手进行对战
* main exploiter: 只以 main agent 作为对手进行训练，不断重启训练（找到 main Agent 的弱点）
* league exploiter: 从所有历史版本的模型中采样对手进行训练，不断重启训练（找到历史模型的弱点）。


**OpenAI Five**
使用 PPO 进行自对弈训练 Dota2 游戏中的英雄角色。
对手选择：80% 最新模型，20% 从历史模型中随机选择。
用到了：continual transfer via surgery: 强制保持接续前后模型的策略一致。

*英雄选择*：英雄数量过多，选择英雄时使用 minimax 搜索

**JueWu**
使用 Dual-Clip PPO
对手选择：80% 最新模型，20% 从历史模型中随机选择。
采用课程学习的思路：先选择人类对局中最常见的若干英雄组合进行训练
每一个组合使用 RL 训练一个独立的模型，基于这些模型进行知识蒸馏，得到一个通用模型。

随机生成任意英雄的组合，微调 student 模型。
随机选择英雄时，仍然不遍历，使用 MCTS 搜索选择英雄组合。叶结点计算胜率采用额外监督学习的胜率预测器



## 7. CFR: Counterfactual Regret Minimization

### 7.1 基础知识
对于 Normal Form Game，可以直接使用线性规划求解纳什均衡。
对于 Extensive Form Game:
1. 转换成 Normal Form Game（状态空间爆炸）:变量数量与信息集变成指数级别关系
2. 直接 SFLP(sequence form LP)：变量数量与信息集数量成线性关系

#### Regret Matching
动作概率与累计**正**遗憾值成正比：
$$
\pi^{t+1}(a) = \frac{(R^t(a))^{+}}{\sum_{a'} (R^t(a'))^{+}}
$$

### 7.2 CFR 算法
CFR 通过反复进行自对弈，计算每个信息集的**反事实遗憾值**，并使用遗憾匹配更新策略。
反事实遗憾值定义为：在信息集 $I$ 选择动作 $a$ 相对于当前策略的平均收益差异，假设玩家在信息集 $I$ 处选择动作 $a$，而在其他信息集处仍然按照当前策略行动。

#### 符号
1. $\pi^{\sigma}(h)$: 根据策略组合 $\sigma$ 到达历史状态 h 的概率
2. $\pi^{\sigma}(I)$: 根据策略组合 $\sigma$ 到达信息集 I 的概率,$\pi^{\sigma}(I) = \sum_{h \in I} \pi^{\sigma}(h)$
3. $\pi_{-i}^{\sigma}(I)$:  根据策略组合 $\sigma$ 到达信息集 I 的概率，但是不乘自己的动作概率。


状态 h 的收益：
$$v_i(\sigma,h) = \sum_{z\in Z, h\prec z} \pi^{\sigma}_{-i}(h) \pi^{\sigma}(h,z) u_i(z)$$
$Z$ 表示所有终局，$h \prec z$ 表示历史状态 $h$ 是终局 $z$ 的前缀。$u_i(z)$ 表示终局 $z$ 时玩家 $i$ 的收益。
状态 h 下采取动作 a 的遗憾：$$r_i(h, a) = v_i(\sigma_{I\rightarrow a},h) - v_i(\sigma,h)$$
信息集下采取动作 a 的反事实遗憾：
$$r_i(I, a) = \sum_{h \in I} r_i(h, a)$$

累积遗憾值：
$$R_i^T(I, a) = \sum_{t=1}^{T} r_i^t(I, a)$$
在 T+1 个回合中，信息集 I 处的策略更新为（正比于累积正遗憾值）：当存在正遗憾值时：
$$\pi_i^{T+1}(I, a) = \frac{R_i^{T,+}(I, a)}{\sum_{a'} R_i^{T,+}(I, a')}$$
否则均匀分布：
$$\pi_i^{T+1}(I, a) = \frac{1}{|A(I)|}$$

#### CFR+
1. 保证所有的 regret 值非负：
$$R_i^{T,+}(I, a) = \max(R_i^{T-1,+}(I, a) + r_i^T(I, a), 0)$$
2. 还是对所有策略进行平均，但使用延迟并增长的权重：
   前 d 次迭代不计入平均策略，之后的迭代使用线性增长的权重：$w^t = \max\{t - d, 0\}$

#### MCCFR：Monte Carlo CFR
使用采样的方法估计反事实遗憾值，减少每次迭代的计算量。

**Vanilla CFR**
每次迭代不完整遍历，而是采样
利用抽象减少信息集的个数，利用 DP 优化算法效率

**Chance-sampling CFR**
只对随机事件的节点进行采样，对各玩家的决策节点都需要遍历其子节点;允许上帝节点采样

**External-sampling CFR**
只对对手和上帝节点进行采样
1. 对*对手玩家和自然随机事件节点*进行采样（只采一条路径）。
2. 对*己方玩家的决策节点*，仍然完整遍历所有子节点动作。

**Outcome-sampling CFR**
可以对上帝节点、对手节点和自己节点进行采样。从根节点到叶结点采样一条轨迹。
1. 对整条从根到叶节点的路径进行采样。
2. 包括己方玩家决策节点也只采样一条动作，不遍历所有动作。



缺点：方差大
**VR-MCCFR**
Variance Reduction MCCFR


#### Abstraction
将相似的信息集合并，减少信息集的数量，从而降低计算复杂度。

### 7.3 CFR 的扩展应用

#### DUDO, Liar Die

#### Double Neural CFR (DNCFR)

传统的 CFR 需要维护两张表： $r(I,a)$ 和 $\sigma(I,a)$，当信息集数量过大时，存储和更新这些表变得不可行。DNCFR 使用神经网络来近似这些表，从而能够处理更大规模的博弈。使用一个值网络输入 I 输出每个 a 的*遗憾值*，使用一个策略网络输入 I 输出每个 a 的策略概率。

#### Deep CFR
使用神经网络近似遗憾值和策略概率。与 DNCFR 不同，Deep CFR 使用多个遗憾网络和一个策略网络
1. 对每个玩家一个值网络，输入该玩家需要做决策的I，**输出每个a的遗憾值**
2. 一个策略网络，输入I，输出每个a的概率.
维护策略样本池和每个玩家的值样本池

#### Single Deep CFR (SD-CFR)

Deep CFR专门训练一个平均策略的网络，会带来较大的采样近似误差
Single Deep CFR 只保存每次迭代时的值网络参数，需要用平均策略时实时计算平均策略

#### DREAM: Deep REgret minimization with Advantege Baseline and Model-free learning
在Single Deep CFR基础上加入baseline网络，拟合v(I,a)

### 7.4 德州扑克

**规则：Kuhu Poker**
算法：Linear Programming 求 Nash均衡

**算法：SFLP**
将策略分离为每个决策点下的策略
变量的个数正比于 信息集*动作数

无法解决：HULHE

CFR 也无法解决 HULHE
但是 CFR+ 可以解决 HULHE


**CFR-D**
对博弈树进行分解；子问题求解

**DeepStack** HUNL
