# Machine Learning Notes

## III. Boosting
### AdaBoost Algorithm
AdaBoost (Adaptive Boosting) 是一种迭代的集成学习算法，旨在通过组合多个弱分类器（Weak Learners）来构建一个强分类器（Strong Learner）。其核心思想是通过调整训练样本的权重，使得每一轮训练都更加关注之前被错误分类的样本，从而提高整体分类性能。

#### Algorithm Steps
1.  **初始化样本权重**：对于训练集中的每个样本，初始化权重为均等值，即 $D_1(i) = \frac{1}{N}$，其中 $N$ 是样本总数。
2.  **迭代训练弱分类器**：对于每一轮 $t = 1, 2, \dots, T$：
    1.  使用当前的样本权重分布 $D_t$ 训练一个弱分类器(base classifier) $h_t(x)$。具体方法可以是决策树桩（Decision Stump）等简单模型。权重分布 $D_t$ 影响弱分类器的训练过程，使其更关注被错误分类的样本。
    2.  计算弱分类器的加权错误率：
        $$
        \epsilon_t = \sum_{i=1}^N D_t(i) \cdot I(h_t(x_i) \neq y_i)
        $$
        其中 $I(\cdot)$ 是指示函数，当括号内条件为真时取值为 1，否则为 0。
    3.  计算弱分类器的权重：
    4.  $\gamma_t = 1-2\epsilon_t \in [-1,1]$
    5.  $\alpha_t = \frac{1}{2} \ln\left(\frac{1 + \gamma_t}{1 - \gamma_t}\right)$
    6.  更新样本权重：
        $$
        D_{t+1}(i) = \frac{D_t(i) \cdot e^{-\alpha_t y_i h_t(x_i)}}{Z_t}
        $$
        其中 $Z_t$ 是归一化因子，确保 $D_{t+1}$ 是一个概率分布。

3.  **构建最终分类器**：最终的强分类器 $H(x)$ 通过加权投票的方式组合所有弱分类器：
    $$
    H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)
    $$

If $\alpha_t >0$, the weak classifier is better than random guessing; if $\alpha_t <0$, the weak classifier is worse than random guessing, but its predictions can be inverted to improve performance. When $\alpha_t =0$, the weak classifier performs no better than random guessing and can be discarded.

#### Exponential Loss Interpretation
AdaBoost can be interpreted as minimizing an exponential loss function. The exponential loss for a single sample $(x_i, y_i)$ is defined as:
$$L(y_i, f(x_i)) = e^{-y_i f(x_i)}$$
where $f(x) = \sum_{t=1}^T \alpha_t h_t(x)$ is the combined classifier output before applying the sign function.
The exponential loss is the upper bound of hinge loss, which is used in SVM.

> If use other loss functions, such as logistic loss. $F(\alpha) = \sum_{i=1}^m L(y_i f(x_i)),~f(x)=\sum_{j=1}^t \alpha_j h_j(x)$. By take the derivative of $F$ on a dimension $\alpha_u$, we can proof $F'(\alpha_u)\propto -(1-2\epsilon_u)$.

### Property of AdaBoost

1. In each iteration, the optimal $\alpha_t$ is given by:
$$\alpha_t = \arg\min_\alpha \sum_{i=1}^N D_t(i) e^{-\alpha y_i h_t(x_i)}$$


2. The exponential empirical error after $T$ rounds is:
$$\prod_{t=1}^T Z_t = \frac{1}{n}\sum_{i=1}^N e^{-y_i f(x_i)}$$

3. Assume in AdaBoost algorithm, $\gamma_t\ge\gamma>0, \quad \forall t\in [T]$. Then
   $$\mathbb{P}_S(y_if(x_i)<0)= \frac{1}{n}\sum_{i=1}^N I[y_i f(x_i)<0] \le \frac{1}{n}\sum_{i=1}^N e^{-y_i f(x_i)} \le (1-\gamma^2)^{T/2}$$.


4. The error of $h_t(\cdot)$ measure by $D_{t+1}$ weight is:
    $$
    \sum_{i=1}^N D_{t+1}(i) I[h_t(x_i) \neq y_i] = \frac{1}{2}
    $$

## IV. Clustering

### K-Means Algorithm
Input: Data points $\{x_1, x_2, \dots, x_N\}$, number of clusters $K$.
Output: Cluster assignments $\{C_1, C_2, \dots, C_K\}$ and cluster centroids $\{\mu_1, \mu_2, \dots, \mu_K\}$.
1.  **Initialization**: Randomly select $K$ initial centroids $\{\mu_1, \mu_2, \dots, \mu_K\}$ from the data points.
2.  **Assignment Step**: For each data point $x_i$, assign it to the nearest centroid:
    $$
    C_k = \{x_i : \|x_i - \mu_k\|^2 \le \|x_i - \mu_j\|^2, \forall j = 1, 2, \dots, K\}
    $$
3.  **Update Step**: For each cluster $C_k$, update the centroid $\mu_k$ as the mean of all points assigned to that cluster:
    $$
    \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
    $$
4.  **Convergence Check**: Repeat steps 2 and 3 until the cluster assignments do not change or the centroids stabilize.
5.  **Output**: Return the final cluster assignments and centroids.

But K-meas has no theoretical guarantee to find the global optimal solution.

### K-Means ++ Algorithm
Select the initial centroids more carefully to improve convergence and solution quality.

K-Means ++ Initialization Steps:
1.  Randomly select the first centroid $\mu_1$ from the data points.
2. For each subsequent centroid $\mu_k$ (for $k = 2, 3, \dots, K$):
    1.  Compute the distance $D(x_i)$ from each data point $x_i$ to the nearest already chosen centroid.
    2.  Select the next centroid $\mu_k$ from the data points with probability proportional to $D(x_i)^2$.

Theorem: The expected approximation ratio of K-Means ++ is $O(\log K)$.

## V. On-line Learning

### Online Learning with Expert Advice

There are $N$ experts. At each round $t=1,2,\dots,T$, each expert $i$ makes a prediction $x_{t,i} \in [0,1]$. The learner combines the predictions of all experts to make its own prediction $x_t \in [0,1]$. After making the prediction, the true outcome $y_t \in [0,1]$ is revealed, and the learner incurs a loss $\ell(x_t, y_t)$.

The total loss of the learner after $T$ rounds is:
$$
L_T = \sum_{t=1}^T \ell(x_t, y_t)
$$
The Regret of the learner compared to the **best expert** is defined as:
$$
R_T = L_T - \min_{i=1,\dots,N} \sum_{t=1}^T \ell(x_{t,i}, y_t)
$$

#### Deterministic Weighted Majority Algorithm
1.  **Initialization**: Assign an initial weight $w_{1,i} = 1$ to each expert $i$. $\beta \in (0,1)$
2.  **Prediction**: At each round $t$, compute the learner's prediction as a weighted average of the experts' predictions:
    $$
    x_t=sgn\left(\sum_{i=1}^N w_{t,i} \tilde{y}_{t,i}\right)
    $$
3.  **Update Weights**: After observing the true outcome $y_t$, if $x_t \neq y_t$, update the weights of the experts based on their performance:
    $$
w_{t+1,i} = \begin{cases} \beta w_{t,i} & \text{if } \tilde{y}_{t,i} \neq y_t \\ w_{t,i} & \text{if } \tilde{y}_{t,i} = y_t \end{cases}
    $$
    where $\beta \in (0,1)$ is the learning rate.
4.  **Repeat**: Repeat steps 2 and 3 for $T$ rounds.
5.  **Output**: Return the final weights of the experts.

**Loss Bound**
After $T$ rounds, the total loss of the learner $L_T$ is bounded by:
$$
L_T \le \frac{\log{1/\beta}}{\log(2/(1+\beta))}m^*_T + \frac{\log N}{\log(2/(1+\beta))}
$$
Where $m^*_T$ is the loss made by the best expert after $T$ rounds. $m_T^*=\min_i \sum_{t=1}^T \ell(\tilde{y}_{t,i}, y_t) = \min_i \sum_{t=1}^T I(\tilde{y}_{t,i}\neq y_t)$.

**Proof**
>Define potential function:
$$
w_t = \sum_{i=1}^N w_{t,i}
$$
Initially, $w_1 = N$ since all weights are 1.
At each round $t$, if the learner makes a mistake, the total weight decreases by at least a factor of $\frac{1+\beta}{2}$. Therefore, we have:
$$
w_{t+1} \le w_t \cdot \frac{1+\beta}{2}
$$
After $T$ rounds, we have:
$$
w_{T+1} \le N \left(\frac{1+\beta}{2}\right)^{L_T}
$$
On the other hand, the weight of the best expert after $T$ rounds is at least $\beta^{m^*_T}$, since it made $m^*_T$ mistakes. Therefore, we have:
$$
w_{T+1} \geq \beta^{m^*_T}
$$

#### Randomized Weighted Majority Algorithm
1.  **Initialization**: Assign an initial weight $w_{1,i} = 1$ to each expert $i$. $\beta \in (\frac{1}{2},1)$
2.  **Prediction**: At each round $t$, compute the probability distribution over experts:
    $$
    p_{t,i} = \frac{w_{t,i}}{\sum_{j=1}^N w_{t,j}}
    $$
    The learner **randomly selects an expert** according to this distribution and uses its prediction as the learner's prediction $x_t$.
3.  **Update Weights**: After observing the true outcome $y_t$, update the weights of the experts based on their performance:
    $$
w_{t+1,i} = \begin{cases} \beta w_{t,i} & \text{if } \tilde{y}_{t,i} \neq y_t \\ w_{t,i} & \text{if } \tilde{y}_{t,i} = y_t \end{cases}
    $$
4.  **Repeat**: Repeat steps 2 and 3 for $T$ rounds.
5.  **Output**: Return the final weights of the experts.

**Expected Regret Bound**
The loss of the learner after $T$ rounds is bounded by:
$$
L_T := \sum_{t=1}^T \mathbb{E}[\ell(x_t, y_t)] = \sum_{t=1}^T \sum_{i=1}^N p_{t,i} I(\tilde{y}_{t,i} \neq y_t)
$$
$$
L_T \le (2 -\beta)m^*_T + \frac{\log N}{1-\beta}
$$
When $\beta = 1-\sqrt{\frac{\log N}{T}}$, the expected regret is bounded by:
$$
L_T - m^*_T \le 2\sqrt{T \log N}
$$

**Proof**:
>Define potential function:
$$
w_t = \sum_{i=1}^N w_{t,i}
$$
Initially, $w_1 = N$ since all weights are 1.
At each round $t$, the total weight decreases by at least a factor of $\beta$ for the experts that made mistakes. Therefore, we have:
$$
w_{t+1} = w_t - (1-\beta) \sum_{i=1}^N w_{t,i} \mathbb{I}[\tilde{y}_{t,i} \neq y_t] = w_t (1 - (1-\beta) L_t)
$$
After $T$ rounds, we have:
$$
w_{T+1} = N \prod_{t=1}^T (1 - (1-\beta) L_t) \leq N e^{-(1-\beta) \sum_{t=1}^T L_t} = N e^{-(1-\beta)L_T}
$$
On the other hand, the weight of the best expert after $T$ rounds is at least $\beta^{m^*_T}$, since it made $m^*_T$ mistakes. Therefore, we have:
$$
w_{T+1} \geq \beta^{m^*_T} = e^{m^*_T \log \beta}
$$
It's easy to prove:
$$
\log \beta \geq -(2-\beta)(1-\beta)
$$
So $\log \beta \geq -(2-\beta)(1-\beta)$, hence $w_{T+1} \geq e^{-(2-\beta)(1-\beta) m^*_T}$.
Combining the two bounds on $w_{T+1}$, we have:
$$
N e^{-(1-\beta)L_T} \geq e^{-(2-\beta)(1-\beta) m^*_T}
$$
Taking logarithms on both sides, we get:
$$
-(1-\beta)L_T + \log N \geq -(2-\beta)(1-\beta) m^*_T; \\
L_T \leq (2-\beta)m^*_T + \dfrac{\log N}{1-\beta}
$$

If loss is not binary, the equation $w_{t+1}= w_t (1 - (1-\beta) L_t)$ still holds, where $L_t = \sum_{i=1}^N p_{t,i} \ell(\tilde{y}_{t,i}, y_t)$.


**The Doubling Trick**
When the time horizon $T$ is unknown, we can use the doubling trick to achieve similar regret bounds. The idea is to run the algorithm in phases, where each phase has a length that doubles the previous phase.
First, we run the algorithm for $T_1 = 1$ round, then for $T_2 = 2$ rounds, then for $T_3 = 4$ rounds, and so on. In each phase, we reset the weights of the experts to their initial values.

### Use Online Learning to Prove Minimax Theorem
**Minimax Theorem**: For a zero-sum game with payoff matrix $M$, the minimax theorem states that:
$$
\min_{p } \max_{q} p^T M q = \max_{q} \min_{p} p^T M q
$$

Here row player chooses a mixed strategy $p$ (a probability distribution over rows), and column player chooses a mixed strategy $q$ (a probability distribution over columns).

**Problem reformulation**
Row player as a online learner and each row of the payoff matrix $M$ as an expert. At each round $t$, the row player selects a mixed strategy $p_t$ (a probability distribution over rows) and the column player selects a mixed strategy $q_t$ (a probability distribution over columns). The row player incurs a loss of $p_t^T M q_t$.
Column player tries to maximize the loss of the row player as adversary.

1. **Initialization**: Assign an initial probability $p_{1,i}=\frac{1}{n}$ to each expert $i$.
2. **Update Weights**: After observing the column player's strategy $q_t$, update the probability of the experts based on their performance:
    $$p_{t+1}(i)=p_t(i)\beta^{M_i q_t}/ \sum_{j=1}^n p_t(j)\beta^{M_j q_t} $$
3.  **Repeat**: Repeat step 2 for $T$ rounds.

#### Proof of Minimax Theorem
$$
\min_{p } \max_{q} p^T M q \ge \max_{q} \min_{p} p^T M q
$$
> This direction is easy to prove by noting that for any fixed strategies $p$ and $q$, we have:
$$
p^T M q \ge \min_{p} p^T M q
$$
Taking the maximum over $q$ on both sides gives:
$$
\max_{q} p^T M q \ge \max_{q} \min_{p} p^T M q
$$
Now we take the minimum over $p$ on the left side to obtain the desired inequality.

Next to prove：
$$
\min_{p } \max_{q} p^T M q \le \max_{q} \min_{p} p^T M q
$$

**Proof**
>Let the row player use the Randomized Weighted Majority Algorithm. Use the potential function $w_t=\sum_{i=1}^n w_{t,i}$ where $w_{t,i}$ is the weight of expert $i$ at round $t$. 
For the **Exponential Weights Algorithm**, after $T$ rounds, the expected regret of the row player is bounded by: $R_T\le \sqrt{(T/2)\log N}$

### Multi-armed Bandit Problem
In the multi-armed bandit problem, a learner is faced with $K$ different options (arms). At each round $t=1,2,\dots,T$, the learner selects an arm $a_t \in \{1,2,\dots,K\}$ and receives a loss $r_t$ drawn from an unknown distribution associated with that arm.
We define regret as the difference between the expected reward of the best arm and the expected loss of the learner:
$$
R_T = \mathbb{E}\left[\sum_{t=1}^T \ell_t\right] - \mu(a^*) T=\sum_{a:\Delta(a)>0} \Delta(a) \mathbb{E}[N_T(a)]
$$
#### Upper Confidence Bound (UCB) Algorithm
Select the arm with the highest upper confidence bound at each round.
$$
a_t = \arg\min_{a \in \{1,2,\dots,K\}} \hat{\mu}_{t-1}(a) - c \sqrt{\frac{\log T}{N_{t-1}(a)}}
$$

Regret Bound of UCB Algorithm:
$$
R_T \le \sum_{a:\Delta(a)>0} \left( \frac{16 \log T}{\Delta(a)} + \Delta(a) \right)
$$
$\Delta a = \mu(a)-\mu(a^*)$
其中置信区间的大小对应为 $O\left(\sqrt{\frac{\log T}{N_{t}(a)}}\right)$

**Proof** for $c=1$:
> The $R_T=\sum_{\Delta a>0}\Delta a\cdot N_T(a)$, so we need to bound $\mathbb{E}[N_T(a)]$ for each suboptimal arm $a$.
> Define bad event for arm $a$:
> 1. The best arm $a^*$ is underestimated:
$$
\hat{\mu}_{t-1}(a^*) - \sqrt{\frac{\log T}{N_{t-1}(a^*)}} \ge \mu(a^*)
$$
> 2. The suboptimal arm $a$ is overestimated:
$$
\hat{\mu}_{t-1}(a) + \sqrt{\frac{\log T}{N_{t-1}(a)}} \le \mu(a)
$$



遇到的问题：当非最优的 arm 与 $a^*$ 非常接近时，$R_T$ 不会很大，但是得到的 bound 会很大。
优化 bound：将 suboptimal 分类：
$$
a_{\Delta a>0}: \begin{cases} 
\Delta a \ge \Omega\left(\sqrt{\frac{\ln T}{T}}\right) \quad (1) \\ \Delta a \le O\left(\sqrt{\frac{\ln T}{T}}\right) \quad (2)
\end{cases}
$$

始终 pull (2) 类直接使用 $O\left(\sqrt{\frac{\ln T}{T}}\right)$ 如果在 (1) 类，只用之前的 bound，于是可以综合起来得到：
$$
R_T\le O(\sqrt{T\ln T})
$$

## VI. EM Algorithm

### Target of EM Algorithm
Given observed data $X = \{x_1, x_2, \dots, x_N\}$ and latent variables $Z = \{z_1, z_2, \dots, z_N\}$, we want to maximize the likelihood function:
$$
\max_{\theta} \quad L(\theta) = p(X | \theta) = \prod_{i=1}^N p(x_i | \theta)
$$
However, directly maximizing $L(\theta)$ can be difficult due to the presence of latent variables. The EM algorithm provides an iterative approach to find the maximum likelihood estimates of the parameters $\theta$.


### EM Algorithm for Gaussian Mixture Model
Input: Data points $\{x_1, x_2, \dots, x_N\}$, number of components $K$.
Output: Estimated parameters $\{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$.
1.  **Initialization**: Initialize the parameters $\{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$ randomly or using K-Means.
2.  **E-Step**: For each data point $x_i$ and each component $k$, compute the responsibility $\gamma_{ik}$:
    $$
    \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
    $$
3.  **M-Step**: Update the parameters using the responsibilities:
    $$
    N_k = \sum_{i=1}^N \gamma_{ik} \\
    \pi_k = \frac{N_k}{N}\\
    \mu_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i\\
    \Sigma_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T
    $$
4.  **Convergence Check**: Repeat steps 2 and 3 until the parameters converge (i.e., changes in parameters are below a threshold).
5.  **Output**: Return the estimated parameters $\{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$.
6.  **Note**: EM algorithm guarantees that the likelihood will not decrease after each iteration, but it may converge to a local maximum.
