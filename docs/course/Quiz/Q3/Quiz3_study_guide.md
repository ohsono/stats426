# Quiz 3 — Study Guide with Lecture Concept Links

**STAT 426 | Based on Lectures 5 & 6**

---

## Q1. What is the key limitation of standard MLPs for time series modeling?

| | Answer |
|---|---|
| **Correct** | **(C) They can only see a fixed window of lagged inputs.** |
| (A) | They can not model nonlinear functions. |
| (B) | They require strict stationarity. |
| (D) | They assume Gaussian errors. |

### Lecture Concept Link
> **Lecture 5, Section 4 — MLP for Time Series (Nonlinear Autoregression)**
>
> MLPs operate on a fixed window of lagged inputs $[y_{t-1}, ..., y_{t-p}]$ and cannot access information beyond lag $p$. While they gain universal approximation and learn lag interactions (ruling out A), the **fixed lookback window** is the fundamental structural limitation compared to recurrent architectures. Stationarity and Gaussian errors are not required.

---

## Q2. In a Linear State Space Model (LSSM), which equation describes how observations are generated?

| | Answer |
|---|---|
| **Correct** | **(B) $y_t = C h_t + \eta_t$** |
| (A) | $h_t = A h_{t-1} + \epsilon_t$ — this is the *transition* equation |
| (C) | $y_t = \phi y_{t-1} + \epsilon_t$ — this is an AR(1) model |
| (D) | $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$ — this is an RNN |

### Lecture Concept Link
> **Lecture 5, Section 6 — State Space Models (SSMs)**
>
> An LSSM has two equations:
> - **Transition**: $h_t = A h_{t-1} + B u_t + \epsilon_t$ (how the hidden state evolves)
> - **Observation**: $y_t = C h_t + \eta_t$ (how observations are *generated* from the hidden state)
>
> The observation equation maps latent states to what we actually measure. Option (A) is the transition equation, not the observation.

---

## Q3. Why is tanh preferred over ReLU in vanilla RNNs?

| | Answer |
|---|---|
| **Correct** | **(C) ReLU is unbounded and may cause exploding activations.** |
| (A) | ReLU is computationally expensive. — False, ReLU is cheaper |
| (B) | tanh avoids vanishing gradients. — False, tanh still suffers from vanishing gradients |
| (D) | tanh is linear. — False, tanh is nonlinear |

### Lecture Concept Link
> **Lecture 5, Section 8 — The Vanilla RNN Cell**
>
> In RNNs, the *same* weight matrix $W_{hh}$ is multiplied at every time step. ReLU is unbounded $[0, \infty)$, so if weights are even slightly > 1, values grow exponentially (e.g., $1.1^{100} \approx 13{,}780$), leading to NaN. Tanh is **bounded in $[-1, 1]$**, squashing the hidden state at every step and ensuring forward pass stability. Note: tanh does NOT solve vanishing gradients — that requires LSTM/GRU.

---

## Q4. In a Many-to-One RNN structure, what is used to make the final prediction?

| | Answer |
|---|---|
| **Correct** | **(C) The final hidden state $h_T$** |
| (A) | The first hidden state $h_1$ |
| (B) | The average of all hidden states |
| (D) | The input sequence |

### Lecture Concept Link
> **Lecture 5, Section 9 — RNN Input-Output Architectures (Many-to-One)**
>
> In Many-to-One architectures (e.g., sentiment analysis, time series classification), the RNN processes the entire sequence $x_1, ..., x_T$, accumulating information into the hidden state. Only the **final hidden state $h_T$** is used as a "feature vector" representing the whole sequence: $\hat{y} = g(W_{hy} h_T + b_y)$. Intermediate outputs are ignored.

---

## Q5. Why does Backpropagation Through Time (BPTT) lead to instability for long sequences?

| | Answer |
|---|---|
| **Correct** | **(B) Because gradients depend on repeated multiplication of $W_{hh}^\top$.** |
| (A) | Because hidden states are independent across time. — False, they are dependent |
| (C) | Because the loss is averaged across time. — Averaging doesn't cause instability |
| (D) | Because Softmax saturates. — Not relevant to BPTT |

### Lecture Concept Link
> **Lecture 6, Section 1 — Training RNNs: BPTT / Gradient Computation**
>
> The recursive hidden gradient expands to:
> $$\frac{\partial L}{\partial h_t} = \sum_{i=t}^{T} (W_{hh}^\top)^{i-t} W_{qh}^\top \frac{\partial L}{\partial o_i}$$
>
> The term $(W_{hh}^\top)^{i-t}$ involves raising a matrix to a high power. Eigenvalues $|\lambda| > 1$ cause **exploding** gradients; $|\lambda| < 1$ cause **vanishing** gradients. This repeated multiplication is the mathematical root cause of instability.

---

## Q6. What determines whether gradients vanish or explode in a linearized RNN analysis?

| | Answer |
|---|---|
| **Correct** | **(C) The eigenvalues of the recurrent weight matrix.** |
| (A) | The batch size. |
| (B) | The learning rate. |
| (D) | The activation function only. |

### Lecture Concept Link
> **Lecture 6, Section 1 — Analytical Expansion & Instability / Lecture 5, Section 10**
>
> The gradient is proportional to $(W_{hh})^T$. The behavior is governed by the **eigenvalues $\lambda$** of $W_{hh}$:
> - $|\lambda| > 1$: $\lambda^{100} \to \infty$ — **Exploding** (fix: gradient clipping)
> - $|\lambda| < 1$: $\lambda^{100} \to 0$ — **Vanishing** (fix: LSTM/GRU)
>
> Batch size and learning rate affect optimization but not this fundamental matrix-power instability.

---

## Q7. What is the main idea of Truncated BPTT?

| | Answer |
|---|---|
| **Correct** | **(B) Limit gradient flow to a fixed window $\tau$.** |
| (A) | Remove hidden states entirely. |
| (C) | Freeze recurrent weights. |
| (D) | Train only on small datasets. |

### Lecture Concept Link
> **Lecture 6, Section 1 — Truncated BPTT**
>
> Full BPTT is infeasible for very long sequences. The strategy:
> 1. **Split** the sequence into blocks of length $\tau$ (e.g., $\tau \approx 35$)
> 2. **Forward**: pass $h_\tau$ from block $k$ to block $k+1$ (preserves context)
> 3. **Backward**: stop gradient at block boundaries ($\partial L / \partial h_\tau \leftarrow 0$)
>
> **Trade-off**: limits the gradient's "memory" to $\tau$ steps, sacrificing long-term dependency learning for computational efficiency and stability.

---

## Q8. In a Deep RNN with $L$ hidden layers, what are the two inputs to hidden layer $l$ at time $t$?

| | Answer |
|---|---|
| **Correct** | **(C) $h_t^{(l-1)}$ and $h_{t-1}^{(l)}$** |
| (A) | Only $h_{t-1}^{(l)}$ — missing spatial input |
| (B) | Only $h_t^{(l-1)}$ — missing temporal input |
| (D) | The output layer only |

### Lecture Concept Link
> **Lecture 6, Section 2 — Deep (Stacked) RNNs / Architecture**
>
> Each layer $l$ receives two inputs:
> - **Spatial (bottom-up)**: $h_t^{(l-1)}$ — output of layer below at current time $t$
> - **Temporal (left-right)**: $h_{t-1}^{(l)}$ — same layer's state at previous time $t-1$
>
> Update: $h_t^{(l)} = \phi_l(W_{xh}^{(l)} h_t^{(l-1)} + W_{hh}^{(l)} h_{t-1}^{(l)} + b_h^{(l)})$

---

## Q9. What is the theoretical bottleneck in a standard Encoder-Decoder model without attention?

| | Answer |
|---|---|
| **Correct** | **(C) A fixed-size context vector must encode arbitrarily long sequences.** |
| (A) | The decoder can not generate sequences. — False |
| (B) | The encoder hidden state dimension grows with input length. — False, it's fixed |
| (D) | The model cannot use Softmax. — False |

### Lecture Concept Link
> **Lecture 6, Section 3 — Encoder-Decoder / The Bottleneck Problem**
>
> The context variable $c$ (typically $c = h_T$) is a **fixed-size vector** regardless of input length. It must encode the nuance of 5 words or 100 words equally. As $T$ increases, $h_T$'s ability to retain information from $x_1$ diminishes due to fixed capacity and vanishing gradients. This **information bottleneck** is what motivates the Attention Mechanism.

---

## Q10. During Teacher Forcing in Decoder training, what is fed into the next time step?

| | Answer |
|---|---|
| **Correct** | **(C) The ground truth previous token.** |
| (A) | The predicted output. — This is used during *inference* |
| (B) | A random token. |
| (D) | Only the context vector. |

### Lecture Concept Link
> **Lecture 6, Section 3 — Teacher Forcing / Lecture 5, Section 9**
>
> Two regimes:
> - **Training (Teacher Forcing)**: feed the actual ground truth $y_{t'-1}$ — "breaks" the causal chain so the model's previous errors don't compound
> - **Inference (Auto-regressive)**: feed the predicted $\hat{y}_{t'-1}$ — errors at $t-1$ become inputs at $t$, causing "error drift"

---

## Q11. In scaled dot-product attention, why do we divide by $\sqrt{d}$?

| | Answer |
|---|---|
| **Correct** | **(B) To prevent large variance in dot-product scores.** |
| (A) | To normalize embeddings. |
| (C) | To reduce memory usage. |
| (D) | To enforce orthogonality. |

### Lecture Concept Link
> **Lecture 6, Section 4 — Scoring Functions / Scaled Dot-Product**
>
> If $q$ and $k$ elements are i.i.d. with mean 0 and variance 1, the dot product $\sum_{j=1}^{d} q_j k_j$ has **variance $d$**. For large $d$ (e.g., 512), scores become huge ($\pm 20$), pushing softmax outputs to 0 or 1 and causing **vanishing gradients**. Dividing by $\sqrt{d}$ rescales variance back to 1, stabilizing softmax.

---

## Q12. Given $Q \in \mathbb{R}^{n \times d}$ and $K \in \mathbb{R}^{m \times d}$, what is the dimension of $QK^\top$?

| | Answer |
|---|---|
| **Correct** | **(C) $n \times m$** |
| (A) | $d \times d$ |
| (B) | $n \times d$ |
| (D) | $m \times n$ |

### Lecture Concept Link
> **Lecture 6, Section 4 — Matrix Form (GPU-efficient)**
>
> $QK^\top$: $(n \times d) \times (d \times m) = (n \times m)$ — a matrix of scores where entry $(i, j)$ is the similarity between query $i$ and key $j$. Softmax is applied row-wise (for each query). Then result $\times V$: $(n \times m) \times (m \times d_v) = (n \times d_v)$.

---

## Q13. In a univariate autoregressive RNN without external covariates, what is typically used as the input $x_t$?

| | Answer |
|---|---|
| **Correct** | **(C) The previous observation $y_{t-1}$.** |
| (A) | Random noise. |
| (B) | The full past history $(y_{t-1}, ..., y_1)$. — This is implicit in $h_t$, not explicit input |
| (D) | The hidden state $h_{t-1}$. |

### Lecture Concept Link
> **Lecture 5, Section 8 — The Vanilla RNN Cell / Univariate Autoregressive RNN**
>
> When there are no external covariates, the "external input" slot is empty. We set: $x_t := y_{t-1}$
>
> The equation becomes: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} y_{t-1} + b)$
>
> The RNN implements a nonlinear autoregressive model with implicit, potentially infinite memory encoded in $h_t$. The full history is captured through the recurrent hidden state, not fed explicitly as input.

---

## Q14. In a Many-to-Many (synchronized) RNN setting, how are outputs generated?

| | Answer |
|---|---|
| **Correct** | **(C) At every time step aligned with the input sequence.** |
| (A) | Only after the final time step. — This is Many-to-One |
| (B) | Only at the first time step. |
| (D) | Randomly during training. |

### Lecture Concept Link
> **Lecture 5, Section 9 — RNN Input-Output Architectures (Many-to-Many)**
>
> In synchronized Many-to-Many: $T_{in} = T_{out}$. At each time step $t$, the RNN consumes input and produces output immediately:
> - $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$
> - $\hat{y}_t = g(h_t)$
>
> Examples: frame-by-frame video classification, named entity recognition. When $T_{in} \neq T_{out}$, the Encoder-Decoder (Seq2Seq) architecture is needed instead.

---

## Quick Reference: Topic-to-Lecture Map

| Topic | Lecture | Section |
|---|---|---|
| MLP limitations for time series | Lec 5 | Sec 4: MLP for Time Series |
| LSSM observation equation | Lec 5 | Sec 6: State Space Models |
| tanh vs ReLU in RNNs | Lec 5 | Sec 8: Vanilla RNN Cell |
| Many-to-One architecture | Lec 5 | Sec 9: RNN I/O Architectures |
| Univariate autoregressive RNN input | Lec 5 | Sec 8: Vanilla RNN Cell |
| Many-to-Many synchronized outputs | Lec 5 | Sec 9: RNN I/O Architectures |
| BPTT instability ($W_{hh}$ multiplication) | Lec 6 | Sec 1: Training RNNs (BPTT) |
| Eigenvalues & vanishing/exploding gradients | Lec 6 | Sec 1: Training RNNs (BPTT) |
| Truncated BPTT | Lec 6 | Sec 1: Training RNNs (BPTT) |
| Deep RNN layer inputs | Lec 6 | Sec 2: Deep (Stacked) RNNs |
| Encoder-Decoder bottleneck | Lec 6 | Sec 3: Encoder-Decoder (Seq2Seq) |
| Teacher Forcing | Lec 6 | Sec 3: Encoder-Decoder (Seq2Seq) |
| Scaled dot-product ($\sqrt{d}$ scaling) | Lec 6 | Sec 4: Attention Mechanisms |
| $QK^\top$ dimensions | Lec 6 | Sec 4: Attention Mechanisms |
