# STAT 426: Additional Sample Quiz 4 Questions

George Michailidis

---

**15. What is the key modification that AdamW introduces over the standard Adam optimizer?**

A) It replaces the exponential moving average of gradients with a simple cumulative sum.

B) It decouples the weight decay regularization from the adaptive gradient update step, applying decay directly to the parameters.

C) It removes the bias correction terms for the first and second moment estimates entirely.

D) It replaces the Polyak momentum component of Adam with Nesterov momentum.

---

**16. In a Transformer's Encoder-Decoder architecture, what is the role of Cross-Attention in the Decoder?**

A) It allows each Decoder token to attend to every other Decoder token bidirectionally.

B) The Decoder's hidden states serve as Queries, while the Encoder's final output provides the Keys and Values, enabling the Decoder to dynamically retrieve relevant source information.

C) It replaces the Feed-Forward Network sub-layer to reduce the total parameter count.

D) It forces the Encoder and Decoder to share identical weight matrices for all attention projections.

---

**17. Why does the Adam optimizer include a bias correction step for its first and second moment estimates?**

A) To prevent the momentum term from ever exceeding a magnitude of 1.0.

B) Because both moment estimates are initialized to zero and are biased toward zero in the early iterations; dividing by (1 − β^k) corrects this underestimation.

C) To enforce sparsity in the gradient updates, ensuring that only a subset of parameters are updated per step.

D) To normalize the gradient vector to unit length before applying the parameter update.

---

**18. In the context of mini-batch SGD, what is the relationship between an "epoch" and an "iteration"?**

A) One epoch equals one iteration; the terms are interchangeable in all settings.

B) One epoch consists of ceil(m/s) iterations, where m is the dataset size and s is the batch size, meaning each sample is seen exactly once per epoch.

C) One iteration always processes the entire dataset, while an epoch processes only a single batch.

D) The number of epochs is always equal to the number of iterations multiplied by the batch size.

---

**19. What is the primary purpose of the "Add & Norm" (Residual Connection + Layer Normalization) sub-layer that wraps every component in a Transformer block?**

A) To reduce the model dimension by half after each sub-layer, thereby compressing the representation.

B) The residual connection creates a gradient highway that prevents vanishing gradients in deep stacks, while Layer Normalization stabilizes feature magnitudes to prevent unbounded growth.

C) To inject new positional encoding information at every layer of the Transformer.

D) To randomly zero out activations during training as a form of dropout regularization.

---

**20. What is the purpose of the Feed-Forward Network (FFN) sub-layer within each Transformer Encoder block?**

A) It allows tokens to exchange information with each other across the sequence, replacing the need for attention.

B) It applies a shared, position-wise nonlinear transformation (expand to a higher dimension, activate, compress back) that enables the model to learn complex representations beyond what the purely linear attention mechanism provides.

C) It reduces the sequence length by pooling adjacent token representations together.

D) It serves exclusively as a skip connection to preserve the original input embeddings unchanged.

---

**21. Why is "learning rate warmup" considered critical when training Transformer models?**

A) It prevents the tokenizer from producing out-of-vocabulary tokens during the first few training steps.

B) Early in training, attention patterns are essentially random and internal representations are unorganized; large parameter updates during this phase can destabilize learning, so the learning rate is gradually increased from a very small value.

C) Warmup forces the model to memorize the training data before it begins to generalize.

D) It is needed solely to compensate for the bias correction terms in the Adam optimizer during the first few steps.

---

## Answer Key

| Q# | Answer | Explanation | Reference |
|---|---|---|---|
| 15 | **B** | Standard L2 regularization (adding λx to the gradient) behaves differently inside adaptive optimizers like Adam. AdamW decouples weight decay by applying it directly to parameters: x_{k+1} = (1 − ηλ)x_k − η·m̂_k/√(ẑ_k + ε), improving generalization. | Slides 62–63, Lecture 7 |
| 16 | **B** | Cross-Attention is the "bridge" between Encoder and Decoder. The Decoder state (Query) searches the Encoder's memory (Keys and Values) to retrieve the most relevant source context for generating each output token. | Slides 43–50, Lecture 8 |
| 17 | **B** | Adam initializes both m_0 = 0 and z_0 = 0. Because the EMA update blends with zero, the early estimates severely underestimate the true moments. The correction m̂_k = m_k / (1 − β_1^k) and ẑ_k = z_k / (1 − β_2^k) removes this initialization bias, ensuring accurate moment estimates from the very first step. | Slides 56–57, Lecture 7 |
| 18 | **B** | An epoch is one full pass over all m training samples. With mini-batch size s, the dataset is split into ceil(m/s) batches, each corresponding to one parameter update (iteration). For example, m = 1000 and s = 10 yields 100 iterations per epoch; training for 20 epochs means 2000 total parameter updates. | Slides 87–89, Lecture 7 |
| 19 | **B** | Residual connections (x + Sublayer(x)) allow gradients to flow backwards through the network unchanged, enabling the stacking of dozens of layers without vanishing gradients. Layer Normalization stabilizes the hidden state values, preventing them from growing unboundedly as residual additions accumulate. Both are required for the same dimension d_model. | Slides 13–14, Lecture 8 |
| 20 | **B** | Self-Attention is an entirely linear operation (weighted sums of Value vectors). The FFN introduces the critical nonlinearity (via ReLU or GELU) that allows the model to learn complex, nonlinear representations. It expands the token from d_model to d_ff = 4 × d_model, applies the activation, then compresses back to d_model. It is applied identically and independently to each token position. | Slide 15, Lecture 8 |
| 21 | **B** | Transformer training is sensitive to initialization. Early in training, attention weights are near-random and LayerNorm statistics are unstable. Large learning rates at this stage can cause divergent updates. Warmup gradually increases the LR from near-zero over the first 1–5% of steps, allowing representations to stabilize before applying the full learning rate, followed by cosine decay. | Slides 58–59, Lecture 8 |
