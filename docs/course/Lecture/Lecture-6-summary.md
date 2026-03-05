# Lecture 6: RNNs, Sequence-to-Sequence Models and Attention Mechanisms

**STAT 426 | George Michailidis**

---

## 1. Training RNNs: Backpropagation Through Time (BPTT)

### Forward and Backward Pass

- **Forward**: RNN unrolls over T time steps, computing hidden states h_t and outputs o_t sequentially
- **Backward**: error gradients propagate from the future back into the past
- Unlike feedforward networks, the gradient at time t depends on all previous steps 1, ..., t-1

### Simplified RNN (no bias, identity activation)

- Hidden state: $h_t = W_hx * x_t + W_hh * h_{t-1}$
- Output: $o_t = W_qh * h_t$
- Loss: $L = (1/T) * sum of l(o_t, y_t)$ over $t = (1..T)$ (MSE for regression, cross-entropy for classification)
- Parameters W_hx, W_hh, W_qh are **shared across all time steps**

### Gradient Computation

- **Output layer gradient**: $dL/dW_qh$ = sum over t of $(dL/do_t) * h_t^T$ — requires storing $h_t$ for every $t$
- **Recursive hidden gradient** (the key challenge):
  - $dL/dh_t$ = (immediate contribution from $o_t$) + (future contribution from $h_{t+1}$)
  - $dL/dh_t = W_qh^T * dL/do_t + W_hh^T * dL/dh_{t+1}$
  - Base case at $t = T$: future term is 0
- **Expanded form**: $dL/dh_t = \sum_{i=t}^{T} (W_hh^T)^{i-t} * W_qh^T * dL/do_i$
- **Eigenvalue analysis** of $(W_hh)^{i-t}$:
  - $|\lambda| > 1$: **Exploding gradients** (fix: gradient clipping)
  - $|\lambda| < 1$: **Vanishing gradients** (fix: LSTM/GRU)

### Parameter Gradient Accumulation

- $dL/dW_hx = \sum_t (dL/dh_t) * x_t^T$
- $dL/dW_hh = \sum_t (dL/dh_t) * h_{t-1}^T$
- Must cache $x_t, h_t, h_{t-1}$ for the entire sequence; computation of $dL/dh_t$ is the bottleneck (sequential, not parallelizable)

### Truncated BPTT

- Full BPTT is infeasible for very long sequences (memory + instability)
- **Strategy**: split the sequence into blocks of length tau (e.g., tau ~ 35)
- **Forward pass**: final h_tau of block k becomes initial h_0 of block k+1 (preserves context)
- **Backward pass**: gradient is stopped at block boundaries (dL/dh_tau <- 0)
- **Trade-off**: limits gradient memory to tau steps; sacrifices long-term dependency learning for efficiency

### Training Summary

1. BPTT = standard backprop applied to an unrolled time graph
2. Shared weights mean gradients from all steps accumulate into one update
3. Recursive matrix multiplication causes inherent instability
4. Truncation after tau steps saves memory and prevents divergence

## 2. Deep (Stacked) RNNs

### Motivation

- Single-layer RNN has only one transformation per time step — insufficient for hierarchical features (e.g., syntax -> semantics -> sentiment)
- Stacking L layers processes information at different time scales and abstraction levels

### Architecture

- Layer l hidden state depends on:
  - **Spatial input** (bottom-up): $h_{t}^{l-1}$ from the layer below at current time t
  - **Temporal input** (left-right): $h_{t-1}^{l}$ from the same layer at previous time t-1
- Update equation: $h_t^{l} = \phi_l(W_xh^{l} * h_t^{l-1} + W_hh^{l} * h_{t-1}^{l} + b_h^{l})$
- For layer 1, input is the raw data x_t
- Only the top layer $h_t^{L}$ connects to the output: $o_t = W_hq * h_t^{L} + b_q$

### Parameters

- Each layer has its own unique $W_xh^{l}, W_hh^{l}, b_h^{l}$ (not shared between layers)
- Weights are shared across time steps within each layer

### Computational Cost

- Backpropagation along **two axes**: through time (truncated BPTT) and through depth (layer L to 1)
- Costs scale with sequence length T, batch size B, hidden size H, and depth L

### Implementation Challenges

- **Variable sequence lengths**: use `pack_padded_sequence` to skip padding zeros
- **State management**: use `h.detach()` to pass hidden state values between batches while cutting gradient history (prevents OOM errors)

## 3. Encoder-Decoder (Seq2Seq) Architectures

### Motivation

- Standard RNNs require $T_x = T_y$ (or $T_y = 1$), but many tasks have variable-length I/O:
  - Translation: "Hello world" (2 tokens) -> "Bonjour le monde" (3 tokens)
  - Forecasting: 24 hours of input -> 48 hours of predictions

### Components

1. **Encoder**: RNN (LSTM/GRU) that reads variable-length input $x_1, ..., x_T$ and compresses it into a fixed-length representation
   - $h_t = f(x_t, h_{t-1})$
2. **Context variable c**: fixed-size vector summarizing the entire input
   - Common choice: $c = h_T$ (last hidden state) or $c = tanh(W * h_T)$
   - Fixed dimension regardless of input length T
3. **Decoder**: separate RNN initialized with $s_0 = g(c)$ that generates the output sequence $y_1, ..., y_{T'}$ one token at a time
   - $s_{t'} = g(y_{t'-1}, c, s_{t'-1})$
   - Context c may be used only for initialization or concatenated at every step

### Tokens: Discrete vs. Continuous

- **NLP (text)**: token is a word/sub-word; output layer uses Softmax (classification)
- **Time series**: token is a scalar/vector; output layer uses Linear projection (regression)

### Output Probability

- $P(y_{t'} | y_1, ..., y_{t'-1}, c) = softmax(W_o * s_{t'} + b_o)$
- Full sequence probability: $P(y|x)$ = product of $P(y_{t'} | y_1, ..., y_{t'-1}, c)$ over $t'$

### Teacher Forcing

- **Training**: feed ground-truth $y_{t'-1}$ as input (prevents error compounding)
- **Inference**: feed predicted $y_hat_{t'-1}$ (auto-regressive generation; prone to error drift)

### The Bottleneck Problem

- Context vector c is fixed-size regardless of input length
- As T increases, $h_T$'s ability to retain information from $x_1$ diminishes
- This **information bottleneck** motivates the Attention Mechanism

## 4. Attention Mechanisms

### Breaking the Bottleneck

- Instead of a static c, generate a **dynamic context vector $c_{t'}$** for every decoding step $t'$
- Creates direct connections between decoder state and specific encoder states
- Reduces path length from O(T) to O(1)

### Theoretical Origins

- Borrowed from cognitive neuroscience:
  - **Non-volitional cues (Keys)**: saliency — elements that stand out naturally
  - **Volitional cues (Queries)**: intent — conscious focus
- Attention = biasing the selection of Keys based on a Query

### Core Abstraction: Differentiable Lookup

- Database $D = {(k_1, v_1), ..., (k_m, v_m)}$
- **Query (q)**: decoder state $s_{t'-1}$
- **Keys (k)**: encoder hidden states $h_1, ..., h_m$
- **Values (v)**: usually same as keys
- $Attention(q, D) = sum of alpha(q, k_i) * v_i (soft/weighted selection)$

### Step-by-Step Computation

1. **Score** (alignment): $e_i = score(q, k_i)$
2. **Normalize** (softmax): $\alpha_i = \exp(e_i) / \sum_j \exp(e_j)$
3. **Aggregate**: $c = \sum_i \alpha_i * v_i$

### Scoring Functions

1. **Additive attention**: $e_i = v_a^T *tanh(W_q* q + W_k * k_i)$
   - For when q and k have different dimensions; flexible but slower
2. **Scaled dot-product**: $e_i = (q^T * k_i) / sqrt(d)$
   - For when q and k have the same dimension d
   - Division by sqrt(d) rescales variance to prevent softmax saturation for large d

### Matrix Form (GPU-efficient)

- $Attention(Q, K, V) = softmax(Q *K^T / sqrt(d))* V$
- Q in R^{n x d}, K in R^{m x d}, V in R^{m x d_v} -> result in R^{n x d_v}

### Cross-Attention in Encoder-Decoder

- Encoder hidden states $h_1, ..., h_m$ become Keys and Values
- Decoder state $s_{t'-1}$ becomes the Query
- Dynamic context $c_{t'} = sum of alpha_i * h_i$ is computed at each decoder step
- Decoder uses $[s_{t'-1}; c_{t'}]$ to predict the next token

### Self-Attention (Transformer Core)

- Query, Key, and Value all come from the **same sequence**:
  - $q_i = W_q *x_i, k_i = W_k* x_i, v_i = W_v * x_i$
- Every token looks at every other token to determine context
- Example: in "The bank of the river", "bank" attends to "river" for disambiguation
- **Permutation invariant**: self-attention treats input as a bag of words (order-insensitive)

### Positional Encodings

- Fix permutation invariance by adding position information: $x'_pos = x_pos + PE_pos$
- **Sinusoidal encoding** (fixed):
  - $PE(pos, 2i) = sin(pos / 10000^{2i/d})$
  - $PE(pos, 2i+1) = cos(pos / 10000^{2i/d})$
  - High frequencies capture local order; low frequencies capture global position

### Multi-Head Attention

- A single attention head creates one average distribution; cannot simultaneously focus on different relationship types
- **Solution**: run H attention layers in parallel, each specializing in a different pattern
  - $head_h = Attention(Q *W_Q^h, K* W_K^h, V * W_V^h)$
  - $MultiHead(Q, K, V) = Concat(head_1, ..., head_H) * W_O$
- Example with d=512 and H=8: each head operates on dimension 64; cost roughly equals single-head
- $W_O$ (output projection) mixes insights from all heads into a unified representation

### Embedding: Discrete vs. Continuous Inputs

- **NLP**: embedding lookup table (integer index -> row of learned matrix E)
- **Time series**: linear projection $(x_t * W_emb + b_emb)$ maps $R^3 -> R^512$

### Masking

- **Padding masks**: set scores of PAD positions to -inf before softmax (forces $\alpha = 0$)
- **Causal masks** (decoder): triangular mask prevents attending to future tokens during training

## 5. Complexity Comparison: RNN vs. Self-Attention

| Property | RNN | Self-Attention (Transformer) |
|---|---|---|
| Complexity | $O(T * d^2)$ | $O(T^2 * d)$ |
| Parallelism | $O(1)$ — sequential | $O(T)$ — fully parallel |
| Path length | $O(T)$ — long dependencies are hard | $O(1)$ — direct access to everything |

**Key trade-off**: Attention is faster to train (parallelizable) and better at long-range dependencies, but memory grows quadratically with sequence length (T^2).
