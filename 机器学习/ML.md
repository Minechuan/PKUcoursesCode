# Machine Learning

## 预备知识

### Probability Theory

Markov Inequality:$a>0$,$X$ is a non-negative random variable:
$$
\mathbb{P}(X \ge a) \le \frac{\mathbb{E}[X]}{a}
$$
Chebyshev Inequality:
$$
\mathbb{P}(|X-\mathbb{E}[X]| \ge k) \le \frac{\mathrm{Var}(X)}{k^2}
$$
Concetration Inequality:
Hoeffding Inequality:
$$
\mathbb{P}\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mathbb{E}[X_i]\right| \ge \epsilon\right) \le 2e^{-\frac{2n\epsilon^2}{(b-a)^2}}
$$

Concentration Inequality for iid Bernoulli Distribution:(proved by chernoff Bound)
$$
\mathbb{P}\left(\frac{1}{n}\sum_{i=1}^n X_i - p \ge \epsilon\right) \le e^{-2n\epsilon^2}
$$
* If the indenpendent random variable $X_i$ is bounded in $[0,1]$, the above inequality still holds.
* If the random variables are not independent, but drawn without replacement from a finite population, the above inequality still holds (proved by Taylor expension of $\mathbb{E}[e^{t\sum X_i}]$).

**Hoeffding Lemma**:
If random variable $X$ is bounded in $[a,b]$ and $\mathbb{E}[X]=0$, then for any $t>0$,
$$
\mathbb{E}[e^{tX}] \le e^{\frac{t^2(b-a)^2}{8}}
$$

**Khintchine-Kahane Inequality**:
Let $\{\epsilon_i\}_{i=1}^n$ be independent Rademacher random variables, i.e., $\mathbb{P}(\epsilon_i=1)=\mathbb{P}(\epsilon_i=-1)=\frac{1}{2}$. For any vector $\{v_i\}_{i=1}^m$ in a normed space $\mathbb{R}^n$, we have:
$$
\frac{1}{2}\sum_{i=1}^m \|v_i\|^2 \le \left(\mathbb{E}\left\|\sum_{i=1}^m \epsilon_i v_i\right\|\right)^{2} \le \sum_{i=1}^m \|v_i\|^2
$$




---

## I. VC Theory

Vapnik-Chervonenkis (VC) Theory 是统计学习理论的一个重要分支，主要研究学习算法的泛化能力。但是假设的是 worst-case 的情况，因此 VC Theory 给出的泛化界限通常比较松。

### Finite Hypothesis Space
$$
\mathbb{P}\left(\mathbb{P}(Y\neq \hat{f}(X))-\frac{1}{n}\sum_{i=1}^n I(y_i\neq \hat{f}(x_i))\ge\epsilon\right) 
\le \mathbb{P}\left(\exist f\in \mathcal{F}, \mathbb{P}(Y\neq f(X))-\frac{1}{n}\sum_{i=1}^n I(y_i\neq f(x_i))\ge\epsilon \right)
$$
根据 Union Bound，有：
$$
\le \sum_{f\in \mathcal{F}} \mathbb{P}\left(\mathbb{P}(Y\neq f(X))-\frac{1}{n}\sum_{i=1}^n I(y_i\neq f(x_i))\ge\epsilon \right)
$$
根据 Bernoulli 分布的 Hoeffding 不等式，有：
$$
\le |\mathcal{F}| e^{-2n\epsilon^2}
$$

### Infinite Hypothesis Space

$$
\mathbb{P}\left(\mathbb{P}(Y\neq \hat{f}(X))-\frac{1}{n}\sum_{i=1}^n I(y_i\neq \hat{f}(x_i))\ge\epsilon\right) 
\le \mathbb{P}\left(\sup_{f\in \mathcal{F}}\mathbb{P}(Y\neq f(X))-\frac{1}{n}\sum_{i=1}^n I(y_i\neq f(x_i))\ge\epsilon \right)
$$

#### Step 1. Double Sample Trick
>Let $X_1,X_2,\dots,X_n$ and $X_{n+1},X_{n+2},\dots,X_{2n}$ be two independent iid Bernoulli samples. Define $\nu_1=\frac{1}{n}\sum_{i=1}^n X_i$ and $\nu_2=\frac{1}{n}\sum_{i=n+1}^{2n} X_i,\quad \mathcal{E}[X]=p$.
>Assume $n\ge \frac{\ln 2}{\epsilon^2}$. Then we have:
>$$
\frac{1}{2}\mathbb{P}(|\nu_1-p|\ge 2\epsilon) \le \mathbb{P}(|\nu_1-\nu_2|\ge \epsilon)\le 2\mathbb{P}(|\nu_1-p|\ge \epsilon/2)
>$$
>**Proof:**
>> **Left Inequality:**
>> $$
\mathbb{P}(|\nu_1-\nu_2|\ge \epsilon) \ge \mathbb{P}(|\nu_1-p|\ge 2\epsilon, |\nu_2-p|\le \epsilon) \\
= \mathbb{P}(|\nu_1-p|\ge 2\epsilon)\mathbb{P}(|\nu_2-p|\le \epsilon)$$
>> By Hoeffding Inequality, we have:
>> $$
\mathbb{P}(|\nu_2-p|\le \epsilon) \ge 1 - 2e^{-2n\epsilon^2} \ge \frac{1}{2}
>> $$
>> **right Inequality:**
>> $$
\mathbb{P}(|\nu_1-\nu_2|\ge \epsilon) \le \mathbb{P}(|\nu_1-p|\ge \epsilon/2) + \mathbb{P}(|\nu_2-p|\ge \epsilon/2) \\
= 2\mathbb{P}(|\nu_1-p|\ge \epsilon/2)
>> $$



So the original problem can be transformed to:
$$
\le 2\mathbb{P}\left(\sup_{f\in \mathcal{F}}\left[\frac{1}{n}\sum_{i=1}^n I(y_i\neq f(x_i)) - \frac{1}{n}\sum_{i=n+1}^{2n} I(y_i\neq f(x_i))\right] \ge \epsilon/2\right)
$$

#### Step 2. Symmetrization
Define the $Z_i=(X_i,Y_i)$, and $\varphi_f(Z_i)=I(y_i\neq f(x_i))$.
Introduce permutation variables $\sigma_i$:
$$
\le 2\mathbb{P}\left(\sup_{f\in \mathcal{F}}\left[\frac{1}{n}\sum_{i=1}^n\varphi_f(Z_i) -\frac{1}{n} \sum_{i=n+1}^{2n}\varphi_f(Z_{i})\right] \ge \epsilon/2\right)\\
=2\mathbb{E}_{Z_{1:2n}}\mathbb{P}_{\sigma\in S_{2n}}\left(\sup_{f\in \mathcal{F}}\left[\frac{1}{n}\sum_{i=1}^n\varphi_f(Z_{\sigma_i}) -\frac{1}{n} \sum_{i=n+1}^{2n}\varphi_f(Z_{\sigma_i})\right] \ge \epsilon/2\right)\\
=2\mathbb{E}_{Z_{1:2n}}\mathbb{P}_{\sigma\in S_{2n}}\left(\sup_{f\in \mathcal{F}}\left[\frac{2}{n}\sum_{i=1}^n\varphi_f(Z_{\sigma_i}) -\frac{1}{n} \sum_{i=1}^{2n}\varphi_f(Z_{\sigma_i})\right] \ge \epsilon/2\right)
$$

> **Draw Without Replacement**
> Let $Z_1,Z_2,\dots,Z_n$ be samples drawn without replacement from $\{a_i\}_{i=1}^N$. $p=\frac{1}{N}\sum a_i$. Then we have:
> $$
> \mathbb{P}\left(\frac{1}{n}\sum_{i=1}^n Z_i - p\ge \epsilon\right) \le e^{-2n\epsilon^2}
> $$ 

So we have:
$$
\mathbb{P}_{\sigma\in S_{2n}}\left(\frac{1}{n}\sum_{i=1}^n\varphi_f(Z_{\sigma_i}) -\frac{1}{2n} \sum_{i=1}^{2n}\varphi_f(Z_{\sigma_i}) \ge \epsilon/4\right)\le e^{-O(n\epsilon^2)}
$$
Define $N^{\mathcal{F}}(n)=\max_{Z_1,\dots,Z_n}N^{\mathcal{F}}(Z_1,\dots,Z_n)=\max_{Z_1,\dots,Z_n}\left|\{(\varphi_f(Z_1),\dots,\varphi_f(Z_n)):f\in \mathcal{F}\}\right|$
The original problem can be bounded by:
$$
2\mathbb{E}_{Z_{1:2n}}\mathbb{P}_{\sigma\in S_{2n}}\left(\sup_{f\in \mathcal{F}}\left[\frac{2}{n}\sum_{i=1}^n\varphi_f(Z_{\sigma_i}) -\frac{1}{n} \sum_{i=1}^{2n}\varphi_f(Z_{\sigma_i})\right] \ge \epsilon/2\right)\\
\le 2 N^{\mathcal{F}}(2n) e^{-O(n\epsilon^2)}
$$

#### Step 3. Characterize $N^{\mathcal{F}}(n)$ by VC dimension
By Sauer's Lemma, we have:
$$
N^{\mathcal{F}}(n) \le \sum_{i=0}^{d} \binom{n}{i} \le \left(\frac{en}{d}\right)^d
$$
Where $d$ is the VC dimension of hypothesis space $\mathcal{F}$. So we can bound the **Growth Function**.

**Proof**
>**First Inequality:**
> Form special unshatterable pattern $\{0,\dots,0\}_{d+1}$, we can find $\sum_{i=0}^{d} \binom{n}{i}$ different patterns. For normal patterns, change $1,*$ to 0 can increase feasible patterns, so the total number of patterns is at least $\sum_{i=0}^{d} \binom{n}{i}$.
> **Second Inequality:**
> $$\sum_{i=0}^{d} \binom{n}{i}\le \sum_{i=0}^{d} \binom{n}{i}(\frac{n}{d})^{d-i}\le (\frac{n}{d})^d\sum_{i=0}^{n} \binom{n}{i}(\frac{d}{n})^i = (\frac{n}{d})^d(1+\frac{d}{n})^n \le (\frac{en}{d})^d$$


#### Final Result
Combining all the steps, we have:
$$
\mathbb{P}\left(\mathbb{P}(Y\neq \hat{f}(X))-\frac{1}{n}\sum_{i=1}^n I(y_i\neq \hat{f}(x_i))\ge\epsilon\right)
\le 2 \left(\frac{2en}{d}\right)^d e^{-\frac{n\epsilon^2}{8}}
$$


### VC dimension

#### The definition of VC dimension
The VC dimension of a hypothesis space $\mathcal{F}$ is defined as the largest integer $d$ such that **there exists a set of $d$ points that can be shattered** by $\mathcal{F}$. A set of points is said to be shattered by $\mathcal{F}$ if, for every possible binary labeling of the points, there exists a hypothesis in $\mathcal{F}$ that correctly classifies the points according to that labeling.

#### The VC dimension of linear classifiers in $\mathbb{R}^d$

The VC dimension of the hypothesis space of linear classifiers (hyperplanes) in $\mathbb{R}^d$ is $d + 1$.

**Proof:**
>**首先证明存在$d+1$个点，$\mathcal{F}$能 shatter。**
设 $d$ 个点为 $\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_{d}\}$，且线性无关，构成一个极大线性无关组。令 $\mathbf{x}_{d+1}=\mathbf{0}$，使得下面的矩阵满秩：
>$$
\begin{bmatrix}
    \mathbf{x}_1^T & 1 \\
    \mathbf{x}_2^T & 1 \\
    \vdots & \vdots \\
    \mathbf{x}_{d+1} & 1
\end{bmatrix}
$$
>对于任意一个标记集合$\{y_1,y_2,\cdots,y_{d+1}\},y_i\in\{+1,-1\}$，考虑如下线性方程组：
>$$
\begin{bmatrix}
    \mathbf{x}_1^T & 1 \\
    \mathbf{x}_2^T & 1 \\
    \vdots & \vdots \\
    \mathbf{x}_{d+1}^T & 1
\end{bmatrix}
\begin{bmatrix}
    \mathbf{w} \\
    b
\end{bmatrix}
    =
\begin{bmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_{d+1}
\end{bmatrix}
$$
>由于系数矩阵是满秩的，因此该方程组有解$(\mathbf{w}^*,b^*)$。则对于任意$i\in\{1,2,\cdots,d+1\}$，都有$\text{sgn}(\mathbf{w}^{*T}\mathbf{x}_i+b^*)=y_i$，因此$\mathcal{F}$能 shatter 这$d+1$个点。
**另一方面，证明任意的 $d+2$ 个点不能被 $\mathcal{F}$ shatter。**
记 $\tilde{\mathbf{x}}^T=(\mathbf{x}^T,1)$. $\tilde{\mathbf{w}}^T=(\mathbf{w}^T,b)$。设 $d+2$ 个点为 $\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_{d+2}\}$。如果可以 shatter 这 $d+2$ 个点，则对于任意的标记集合 $\{y_1,y_2,\cdots,y_{d+2}\},y_i\in\{+1,-1\}$，$\exists \tilde{\mathbf{w}}$ 使得:
$$
y_i(\tilde{\mathbf{w}}^T\tilde{\mathbf{x}}_i)>0,\quad \forall i=1,2,\cdots,d+2
$$
由于 d+2 个点满足 $\exists \mathbf{a}\in\mathbb{R}^{d+2}\neq \mathbf{0}, \sum\mathbf{a}_i\tilde{\mathbf{x}}_i=0$，根据 $\mathbf{a}$ 将 index 集合 $\{1,2,\cdots,d+2\}$ 分为两部分：
$$I=\{i|\mathbf{a}_i>0\},J=\{j|\mathbf{a}_j<0\}$$
则构造出一个 $\mathbf{y}$，使得 $y_i=1, \forall i\in I$，$y_j=-1, \forall j\in J$。假设可以被 shatter，则存在 $\tilde{\mathbf{w}}$ 使得:
$$
y_i(\tilde{\mathbf{w}}^T\tilde{\mathbf{x}}_i)>0,\quad \forall i=1,2,\cdots,d+2
$$
即：
$$
\mathbf{a}_i(\tilde{\mathbf{w}}^T\tilde{\mathbf{x}}_i)>0,\quad \forall i=1,2,\cdots,d+2 \\
\sum_{i=1}^{d+2}\mathbf{a}_i(\tilde{\mathbf{w}}^T\tilde{\mathbf{x}}_i)>0
$$
同时有：
$$
\tilde{\mathbf{w}}^T\sum_{i=1}^{d+2}\mathbf{a}_i\tilde{\mathbf{x}}_i=0 \\
\sum_{i=1}^{d+2}\mathbf{a}_i(\tilde{\mathbf{w}}^T\tilde{\mathbf{x}}_i)=0
$$
矛盾，即 $\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_{d+2}\}$ 不可以被 shatter。

---

## II. Support Vector Machine

### Optimization Problem of SVM

支持向量机 (Support Vector Machine, SVM) 是一种用于分类和回归分析的监督学习模型。SVM 的核心思想是找到一个最优超平面 (Hyperplane)，将不同类别的数据点分开，并且最大化类别之间的间隔 (Margin)。


SVM 的优化问题可以表示为：
$$
\begin{aligned}
\min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y_i (w^T x_i + b) \ge 1, \quad i = 1, \dots, N
\end{aligned}
$$
其中，$w$ 是超平面的法向量，$b$ 是偏置项，$(x_i, y_i)$ 是训练数据点及其对应的标签。

**Proof the SVM optimization problem**
$$
\begin{aligned}
\max_{w, b,t} t \\
\text{s.t.} \quad & y_i (w^T x_i + b) \ge t, \quad i = 1, \dots, N \\
\quad & \|w\| = 1
\end{aligned}
$$

$$
\begin{aligned}
\max \quad & \frac{1}{\|w'\|^2} \\
\text{s.t.} \quad & y_i (w'^T x_i + b') \ge 1, \quad i = 1, \dots, N
\end{aligned}
$$

We can convert the maximization problem to minimization problem as follows:
$$
\begin{aligned}
\min \quad & \|w'\|^2 \\
\text{s.t.} \quad & y_i (w'^T x_i + b') \ge 1, \quad i = 1, \dots, N
\end{aligned}
$$


---

### Lagrange Duality

考虑标准形式的优化问题（Primal Problem）：

$$
\begin{aligned}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \le 0, \quad i = 1, \dots, m \\
& h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$

其中 $x \in \mathbb{R}^n$ 是优化变量。要求函数 $f(x)$ 和 $g_i(x)$ 是 convex，$h_j(x)$ 是 linear。

我们定义**拉格朗日函数 (Lagrangian)**：
$$L(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$
其中 $\lambda_i$ 和 $\nu_j$ 是拉格朗日乘子（Dual variables）。**注意：对于不等式约束，要求 $\lambda_i \ge 0$。**

---

### Use minimax to derive the dual problem

Use minimax theorem to derive the dual problem from primal problem:
$$
\min_{x} \max_{\lambda, \nu}  L(x, \lambda, \nu) = \max_{\lambda, \nu} \min_{x} L(x, \lambda, \nu)
$$

The dual problem is defined as:
$$
\begin{aligned}
\max_{\lambda, \nu} \quad & g(\lambda, \nu) = \min_{x} L(x, \lambda, \nu) \\
\text{s.t.} \quad & \lambda_i \ge 0, \quad i = 1, \dots, m
\end{aligned}
$$


### KKT Conditions 的具体形式

KKT 条件由四组方程/不等式组成。如果 $x^*$ 是原问题的最优解，且 $(\lambda^*, \nu^*)$ 是对偶问题的最优解，在一定条件下（如强对偶性 Strong Duality 成立），它们必须满足：

1.  **平稳性条件 (Stationarity)**：拉格朗日函数对 $x$ 的梯度为 0。
    $$ \nabla_x L(x^*, \lambda^*, \nu^*) = \nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0 $$

2.  **原问题可行性 (Primal Feasibility)**：解必须满足原约束。
    $$ g_i(x^*) \le 0, \quad h_j(x^*) = 0 $$

3.  **对偶可行性 (Dual Feasibility)**：不等式约束的乘子必须非负。
    $$ \lambda_i^* \ge 0 $$

4.  **互补松弛性 (Complementary Slackness)**：核心条件。
    $$ \lambda_i^* g_i(x^*) = 0, \quad \forall i $$


**Proof**
>**必要性**：如果 $x^*$ 是最优解，且满足强对偶性，那么 KKT 条件成立。
原问题和对偶问题满足**强对偶性 (Strong Duality)**。这意味着原问题的最优值 $p^*$ 等于对偶问题的最优值 $d^*$。
即：$p^* = d^*$。
令 $x^*$ 为原问题最优解，$(\lambda^*, \nu^*)$ 为对偶问题最优解。
$$
\begin{aligned}
f(x^*) &= p^* \quad \text{(原问题最优值)} \\
&= d^* \quad \text{(强对偶性假设)} \\
&= \min_x L(x, \lambda^*, \nu^*) \quad \text{(对偶函数定义)} \\
&\le L(x^*, \lambda^*, \nu^*) \quad \text{(因为 min 值肯定小于等于在具体点 } x^* \text{ 的函数值)} \\
&= f(x^*) + \sum \lambda_i^* g_i(x^*) + \sum \nu_j^* h_j(x^*) \quad \text{(展开 L)} \\
&\le f(x^*)
\end{aligned}
$$
这说明中间所有的 $\le$ 都必须取等号。
>**充分性**：如果问题是凸的（Convex），且满足 KKT 条件，那么该点一定是最优解。目标：证明 $\tilde{x}$ 是全局最优解。
因为 $f$ 和 $g_i$ 是凸函数，且 $\tilde{\lambda}_i \ge 0$，所以拉格朗日函数 $L(x, \tilde{\lambda}, \tilde{\nu})$ 是关于 $x$ 的凸函数（凸函数的非负线性组合仍然是凸函数）。
>利用平稳性：KKT 条件告诉我们 $\nabla_x L(\tilde{x}, \tilde{\lambda}, \tilde{\nu}) = 0$。所以，对于任意可行解 $x$：$$ L(\tilde{x}, \tilde{\lambda}, \tilde{\nu}) \le L(x, \tilde{\lambda}, \tilde{\nu})\\
f(\tilde{x}) + \sum \tilde{\lambda}_i g_i(\tilde{x}) + \sum \tilde{\nu}_j h_j(\tilde{x}) \le f(x) + \sum \tilde{\lambda}_i g_i(x) + \sum \tilde{\nu}_j h_j(x) $$
对于任意可行解 $x$，有 $g_i(x) \le 0$ 且 $h_j(x) = 0$。又因为 $\tilde{\lambda}_i \ge 0$，所以 $\sum \tilde{\lambda}_i g_i(x) \le 0$，所以右边 $\le f(x)$。


### Derive the Dual Problem of SVM

Let the Lagrange function be:
$$
L(w,b,\lambda,\mu) = \frac{1}{2} \|w\|^2 + \sum_{i=1}^N \lambda_i [1- y_i (w^T x_i + b) ] \\
\partial L / \partial w = w - \sum_{i=1}^N \lambda_i y_i x_i = 0 \implies w = \sum_{i=1}^N \lambda_i y_i x_i \\
\partial L / \partial b = - \sum_{i=1}^N \lambda_i y_i = 0 \implies \sum_{i=1}^N \lambda_i y_i = 0
$$
Substituting $w$ back to $L$, we have Dual Problem: 
$$
\begin{aligned}
\min_{\lambda} \quad &  \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \lambda_i \lambda_j y_i y_j x_i^T x_j - \sum_{i=1}^N \lambda_i \\
\text{s.t.} \quad & \sum_{i=1}^N \lambda_i y_i = 0 \\
\quad & \lambda_i \ge 0, \quad i = 1, \dots, N
\end{aligned}
$$

#### Gram Matrix
The Gram matrix is defined as $K$ where $K_{ij} = x_i^T x_j$. It is a symmetric positive semi-definite matrix.
Let $G_{ij}=y_i y_jx_i^T x_j$, then the dual problem can be rewritten as:
$$
\lambda^TG \lambda = \|\sum_{i=1}^N \lambda_i y_i x_i\|^2 \ge 0 \quad \forall \lambda \ge 0
$$

### SVM for unseparable case

We can introduce variable $\epsilon_i \ge 0$, and the goal is to minimize the following objective function:
$$
\begin{aligned}
\min_{w,b,\epsilon} & \quad  \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \epsilon_i \\
& \text{s.t.} \quad  y_i (w^T x_i + b) \ge 1 - \epsilon_i, \quad i = 1, \dots, N \\
& \quad \epsilon_i \ge 0, \quad i = 1, \dots, N
\end{aligned}
$$
So the Langrange function becomes:
$$
L(w,b,\lambda,\mu) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \epsilon_i + \sum_{i=1}^N \lambda_i [1- y_i (w^T x_i + b) - \epsilon_i ] - \sum_{i=1}^N \mu_i \epsilon_i
$$
The only difference between separable case and unseparable case is the constraint:
$$
\lambda_i \ge 0 \rightarrow 0 \le \lambda_i \le C
$$
* Other cases like $p_i\epsilon_i$, $\epsilon_i^p$ but they only change the optimization target function, won't change the constraint on $\lambda_i$.