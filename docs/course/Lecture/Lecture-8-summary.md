# Lecture 8: The Transformer Architecture

**STAT 426 | George Michailidis**

---

## 1. Motivation: Why Transformers?

- **RNN/LSTM limitations**: strictly sequential computation ($h_t = f(h_{t-1}, x_t)$), cannot parallelize within examples, long-range dependencies decay over time
- **Transformer solution** (Vaswani et al., 2017): eschew recurrence entirely, rely solely on attention mechanisms
- Creates $O(1)$ path length between any two tokens; "Attention Is All You Need"

## 2. Recap: Scaled Dot-Product Attention

### Core Formula

- $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$
- **Query (Q)** $\in \mathbb{R}^{T_q \times d_k}$: what we're looking for
- **Key (K)** $\in \mathbb{R}^{T_k \times d_k}$: identifiers of the data
- **Value (V)** $\in \mathbb{R}^{T_k \times d_v}$: actual data content

### Step-by-Step

1. **MatMul** $QK^\top$: pairwise dot products (similarity scores) between all queries and keys
2. **Scale** $\div \sqrt{d_k}$: stabilizes variance, prevents softmax saturation for large $d_k$
3. **Mask** (optional): set positions to $-\infty$ (padding or causal mask)
4. **Softmax**: converts to probability distribution (row-wise, sums to 1)
5. **MatMul** $\times V$: weighted sum of values — irrelevant tokens filtered out

### Multi-Head Attention (MHA)

- Run $h$ attention heads in parallel with different learned projections
- $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
- $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$
- Allows attending to different representation subspaces simultaneously

## 3. Building the Encoder

### Anatomy of a Single Encoder Block

1. **Input**: embeddings + positional encoding ($X \in \mathbb{R}^{T \times d_{\text{model}}}$)
2. **Multi-Head Self-Attention**: tokens mix information
3. **Add & Norm 1**: $H_1 = \text{LayerNorm}(X + \text{Attention}(X))$
4. **Feed-Forward Network**: point-wise processing per token
5. **Add & Norm 2**: $H_2 = \text{LayerNorm}(H_1 + \text{FFN}(H_1))$

### Positional Encoding

- Self-attention is **permutation invariant** — needs explicit position information
- **Sinusoidal encoding** (fixed):
  - $PE(pos, 2i) = \sin(pos / 10000^{2i/d_{\text{model}}})$
  - $PE(pos, 2i+1) = \cos(pos / 10000^{2i/d_{\text{model}}})$
  - Relative positions can be represented as linear transformations (rotation matrix property)
- **Modern alternatives**:
  - Learned absolute embeddings (BERT, GPT-2/3) — no extrapolation beyond training length
  - **RoPE** (LLaMA, Mistral) — rotates Q/K in complex plane; good extrapolation via scaling
  - **ALiBi** (BLOOM) — subtracts distance-based penalty from scores; excellent zero-shot extrapolation

### Add & Norm Block

- **Residual connection** ("Add"): $x + \text{Sublayer}(x)$ creates gradient highway; requires matching dimensions ($d_{\text{model}}$)
- **Post-LN** (original): $\text{LayerNorm}(x + \text{Sublayer}(x))$
- **Pre-LN** (modern, GPT/LLaMA): $x + \text{Sublayer}(\text{LayerNorm}(x))$ — leaves residual highway uninterrupted
- **LayerNorm** (preferred for NLP): normalizes across feature dim for each token independently; $\text{LN}(h) = \gamma \frac{h - \mu}{\sigma} + \beta$
- **BatchNorm** (CNNs): problematic for NLP due to variable sentence lengths

### Feed-Forward Network (FFN)

- Applied to each token position independently with shared weights
- **Expand-activate-compress**: $\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2$
  - Expand: $d_{\text{model}} \to d_{ff}$ (typically $d_{ff} = 4 \times d_{\text{model}}$)
  - Non-linearity: ReLU or GELU (modern preference — smooth, no dead neurons)
  - Compress: $d_{ff} \to d_{\text{model}}$

### Complexity

- **Time**: $O(T^2 \cdot d)$ — quadratic in sequence length
- **Memory**: $O(T^2)$ for the attention weight matrix
- Doubling context window **quadruples** memory and compute
- Solutions: **FlashAttention** (hardware-aware tiling in SRAM), **Mamba/SSMs** (true linear $O(T)$ scaling)

## 4. The Decoder

### Anatomy of a Single Decoder Block

1. **Masked Multi-Head Self-Attention**: target sequence attends to itself with look-ahead mask ($-\infty$ upper triangle prevents seeing future tokens)
2. **Add & Norm 1**: $H_{d1} = \text{Norm}(Y + \text{MaskedAttn}(Y))$
3. **Multi-Head Cross-Attention**: Decoder (Query) attends to Encoder output (Keys, Values) — the bridge
4. **Add & Norm 2**: $H_{d2} = \text{Norm}(H_{d1} + \text{CrossAttn}(H_{d1}, H_{\text{enc}}))$
5. **FFN + Add & Norm 3**: identical to Encoder FFN

### Teacher Forcing & "Outputs (Shifted Right)"

- **Training**: feed entire true target sequence shifted right (prepend $\langle$SOS$\rangle$), process all timesteps in parallel
- Causal mask ensures token $t$ only sees tokens $0$ to $t-1$
- **Inference**: auto-regressive — feed predicted tokens one at a time (sequential)

### Output Layer

- **NLP**: linear projection to vocabulary size + softmax → probability distribution over tokens
- **Time series**: linear projection to output dimension (regression, no softmax, use MSE loss)

## 5. Training Pipeline

### Forward Pass

- Encoder processes source sequence → $H_{\text{enc}}$
- Decoder receives shifted target + causal mask → logits $\in \mathbb{R}^{M \times V_{\text{vocab}}}$

### Loss & Padding

- **Cross-entropy loss**: $-\log P(\text{correct token})$, averaged over all timesteps
- **Padding mask**: multiply loss at $\langle$PAD$\rangle$ positions by 0

### Optimization Recipe

- **Optimizer**: AdamW (decoupled weight decay)
- **LR schedule**: warmup (linear increase for first 1–5% of steps) + cosine decay
- **Gradient clipping**: cap global norm to 1.0
- **Weight decay**: 0.01–0.1; never applied to biases or LayerNorm parameters

## 6. Inference Pipeline

### Auto-Regressive Generation

1. Encode source (if encoder-decoder model) — run once
2. Start with $\langle$SOS$\rangle$ / prompt tokens
3. Predict next token logits → select token → append → repeat until $\langle$EOS$\rangle$

### Decoding Strategies

| Strategy | Description |
|---|---|
| Greedy | $\arg\max$ at each step; fast but locally optimal |
| Beam Search | Track top $K$ candidates; common in translation |
| Temperature | $p_i = \exp(z_i/T) / \sum_j \exp(z_j/T)$; higher $T$ = more random |
| Top-$k$ | Sample from top $k$ tokens only |
| Top-$p$ (Nucleus) | Sample from smallest set with cumulative probability $\geq p$ |

### KV Cache

- Store Keys and Values from previous steps; only compute K/V for new token
- Avoids redundant recomputation; essential for large-scale LLM deployment

## 7. Modern Foundation Models

### Encoder-Only (BERT)

- Bidirectional (every token attends to every other)
- Pretrained via **Masked Language Modeling** (predict 15% masked tokens)
- Best for: classification, sentiment analysis, reading comprehension

### Decoder-Only (GPT)

- Unidirectional (causal masking, left-to-right)
- Pretrained via **Causal Language Modeling** (next-word prediction)
- Best for: open-ended generation, chatbots, code, reasoning
- Enhanced with RLHF, RAG, RoPE, grouped-query attention

### Encoder-Decoder (T5, BART)

- Bidirectional encoder + auto-regressive decoder
- Pretrained via **corrupted text reconstruction**
- Best for: translation, summarization, sequence-to-sequence tasks

### Beyond Vanilla Transformers

- **Mixture-of-Experts (MoE)**: activate subset of parameters per token; enables trillion-parameter scale
- **Long-context techniques**: FlashAttention, linear attention, 100k+ token windows
- **Speculative decoding**: small draft model proposes tokens, large model verifies
- **Multimodal models**: unified text, vision, and audio architectures
