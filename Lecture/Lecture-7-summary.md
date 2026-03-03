# Lecture 7: Optimization Algorithms for ML/DL

**STAT 426 | George Michailidis**

---

## 1. Basics of Gradient Descent

### Problem Setup

- **Objective**: $\min_{x \in \mathbb{R}^n} f(x)$ — minimize a differentiable loss function
- **Gradient** $\nabla f(x)$: column vector of all partial derivatives; points in the direction of steepest ascent
- **Hessian** $H(x) = \nabla^2 f(x)$: matrix of second partial derivatives; describes local curvature

### Optimality Conditions

- **First order necessary**: $\nabla f(x^*) = 0$ (stationary/critical point)
- **Second order sufficient**: $\nabla f(x^*) = 0$ and $H(x^*) \succ 0$ (positive definite) implies strict local minimizer
- **Critical points**: Local min ($H \succ 0$), local max ($H \prec 0$), saddle point ($H$ has mixed eigenvalues)
- **Convexity**: When $f$ is convex, every local minimum is also a global minimum

### Examples

- **Linear Regression**: $f(\beta) = \frac{1}{2n} \|y - X\beta\|^2$ — convex; closed-form solution $\hat{\beta} = (X^\top X)^{-1} X^\top y$
- **Logistic Regression**: negative log-likelihood — convex; solved via Newton's method / IRLS
- **MLP**: non-convex objective w.r.t. weights and biases

### Gradient Descent Algorithm

- **Update rule**: $x_{k+1} = x_k - \eta_k \nabla f(x_k)$
- Three design factors: descent direction, step size (learning rate), stopping criterion
- **Stopping criteria**:
  1. Gradient norm: $\|\nabla f(x_k)\|_2 \leq \epsilon$
  2. Function value change: $|f(x_{k+1}) - f(x_k)| \leq \epsilon$
  3. Iterate movement: $\|x_{k+1} - x_k\|_2 \leq \epsilon$
- Always include a maximum iteration count

### Step Size Selection

- **Too small** $\eta$: slow convergence (many iterations)
- **Too large** $\eta$: oscillation or divergence
- **Strategies**: exact line search (quadratics only), backtracking line search (too expensive for DL), constant step size, diminishing step size (schedulers)

## 2. Classical Momentum Methods

### The Zigzagging Problem

- GD deteriorates when the objective is highly curved in some directions but flat in others (ill-conditioned, large condition number $\kappa = \lambda_{\max} / \lambda_{\min}$)
- The gradient oscillates across the narrow valley while making slow progress along the valley floor
- Common with **multicollinearity** (highly correlated features)

### Polyak's Heavy Ball Method

- **Two-step view**:
  - Momentum step: $y_k = x_k + \xi(x_k - x_{k-1})$, $\xi \in (0,1)$
  - Gradient step: $x_{k+1} = y_k - \eta_k \nabla f(x_k)$
- **Velocity view** (PyTorch implementation): $v_k = \xi v_{k-1} + \nabla f(x_k)$; $x_{k+1} = x_k - \eta v_k$
- Momentum is an **Exponential Moving Average (EMA)** of past gradients
- Physical intuition: heavy ball with inertia plows through flat regions; $\xi$ acts as friction
- Typical $\xi \in \{0.7, 0.99\}$; $\xi = 0$ recovers standard GD

### Nesterov's Momentum Method

- **Look-ahead strategy**: compute gradient at the extrapolated point, not the current point
  - Extrapolation: $y_k = x_k + \xi_k(x_k - x_{k-1})$
  - Gradient correction: $x_{k+1} = y_k - \eta_k \nabla f(y_k)$
- For convex functions, optimal $\xi_k = \frac{k-1}{k+2}$
- **Key difference from Polyak**: Nesterov applies a "correction" before the step, preventing overshoot

## 3. Adaptive Step Size Methods

### AdaGrad (Adaptive Gradient)

- **Idea**: per-coordinate adaptive step sizes based on gradient history
- Accumulate squared gradients: $s_k = s_{k-1} + g_k \odot g_k$
- Update: $x_{k+1} = x_k - \frac{\eta}{\sqrt{s_k} + \epsilon} \odot g_k$
- **Pro**: excellent for sparse features (rare parameters get larger updates)
- **Con**: $s_k$ monotonically grows, causing the effective step size to shrink to zero prematurely

### RMSprop (Root Mean Square Propagation)

- **Fix for AdaGrad**: use EMA instead of cumulative sum
- $z_k = \beta_2 z_{k-1} + (1 - \beta_2)(g_k \odot g_k)$, with $\beta_2 \approx 0.9$ or $0.99$
- Update: $x_{k+1} = x_k - \frac{\eta}{\sqrt{z_k} + \epsilon} \odot g_k$
- Denominator stays bounded, so the algorithm keeps making progress indefinitely

### ADAM (Adaptive Moment Estimation)

- Combines **Momentum** (acceleration) + **RMSprop** (adaptive step sizes)
- **First moment** (mean of gradients): $m_k = \beta_1 m_{k-1} + (1 - \beta_1) g_k$, $\beta_1 \approx 0.9$
- **Second moment** (uncentered variance): $z_k = \beta_2 z_{k-1} + (1 - \beta_2)(g_k \odot g_k)$, $\beta_2 \approx 0.999$
- **Bias correction**: $\hat{m}_k = m_k / (1 - \beta_1^k)$, $\hat{z}_k = z_k / (1 - \beta_2^k)$
- **Update**: $x_{k+1} = x_k - \eta \frac{\hat{m}_k}{\sqrt{\hat{z}_k} + \epsilon}$
- Currently the **default optimizer** in deep learning (Transformers, CNNs, RNNs)

### AMSGrad

- **Problem**: in ADAM, $\hat{z}_k$ can decrease, causing wild step size increases
- **Fix**: enforce non-decreasing denominator: $\hat{z}_k^{\max} = \max(\hat{z}_{k-1}^{\max}, z_k)$
- Guarantees step size decay and convergence
- PyTorch: `torch.optim.Adam(params, amsgrad=True)`

### AdamW (Decoupled Weight Decay)

- Standard $\ell_2$ regularization is not identical to weight decay in adaptive methods
- **AdamW** applies weight decay directly to parameters, independent of the adaptive step:
  $x_{k+1} = (1 - \eta\lambda) x_k - \eta \frac{\hat{m}_k}{\sqrt{\hat{z}_k} + \epsilon}$
- Typical $\lambda$: 0.01–0.1 for Transformers, $10^{-4}$–$10^{-2}$ for CNNs
- PyTorch: `torch.optim.AdamW(weight_decay=0.01)`

## 4. Stochastic Gradient Descent (SGD)

### Motivation

- Full gradient requires $O(mp)$ operations per update — prohibitive for large datasets
- SGD uses mini-batch gradient: $\nabla f_{I_k}(x_k) = \frac{1}{s} \sum_{i \in I_k} \nabla f_i(x_k)$
- Noise in gradient estimates helps **escape saddle points** and shallow local minima

### Batch Size Terminology

- $s = m$: Batch GD (standard GD) — accurate, stable, expensive
- $s = 1$: Pure SGD — fast, very noisy
- $1 < s < m$: **Mini-batch SGD** — standard in DL (e.g., $s = 32, 64, 256$)

### Convergence Conditions (Robbins-Monro)

- $\sum_{k=1}^{\infty} \eta_k = \infty$ (can travel infinite distance)
- $\sum_{k=1}^{\infty} \eta_k^2 < \infty$ (accumulated variance is finite)
- $\eta_k = 1/k$ and $\eta_k = 1/\sqrt{k}$ both satisfy these conditions
- Fixed step size violates condition #2; SGD oscillates in a "noise ball"

### Batch Size Trade-offs

- **Small batch** ($s \in [16, 256]$): noisy gradients, better generalization, may underutilize GPU
- **Large batch** ($s > 1000$): accurate gradients, approaches full GD; use **linear scaling rule** ($\eta \propto s$)

### Learning Rate Schedules

- Step decay, cosine annealing, warm-up + decay
- Adaptive methods (RMSprop, ADAM) automate decay based on gradient history

## 5. Epochs, Data Splitting, and Autodiff

### Epochs vs. Iterations

- **Epoch**: one complete pass through all $m$ training samples
- **Iteration**: one parameter update using a single batch
- Mini-batch SGD: 1 epoch = $\lceil m/s \rceil$ iterations
- In DL, **number of epochs** is the primary stopping criterion

### Data Splitting

- **Training / Validation / Test** sets
- Traditional ML: 70/15/15 or 60/20/20
- Deep Learning (big data): 98/1/1 — 1% of millions is sufficient for validation

### Automatic Differentiation (Autodiff)

- Evaluates gradients exactly and efficiently via computational graphs
- Core mechanism: decompose functions into elementary operations, apply chain rule via graph traversal
- **Forward mode**: $\partial(\text{all outputs}) / \partial(\text{one input})$ — rarely used in DL
- **Reverse mode (backpropagation)**: $\partial(\text{one output}) / \partial(\text{all inputs})$ — essential for DL (1 loss, millions of parameters)
- Uses **memoization**: caches intermediate values in forward pass to avoid recomputation in backward pass

## 6. PyTorch Optimizer Cheatsheet

| Optimizer | PyTorch Call |
|---|---|
| SGD | `torch.optim.SGD(params, lr=0.01)` |
| SGD + Nesterov | `torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)` |
| Adagrad | `torch.optim.Adagrad(params, lr=0.01)` |
| Adam | `torch.optim.Adam(params, lr=0.001)` |
| NAdam | `torch.optim.NAdam(params, lr=0.002)` |
| AdamW | `torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)` |
| RMSprop | `torch.optim.RMSprop(params, lr=0.01, alpha=0.99)` |
| AMSGrad | `torch.optim.Adam(params, lr=0.001, amsgrad=True)` |
