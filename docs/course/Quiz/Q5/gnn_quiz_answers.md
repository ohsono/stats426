# STAT 426 — Sample Quiz: Graph Neural Networks
**George Michailidis | Answers & Explanations**

> All explanations are derived from **Lecture 9: Graph Neural Networks**.

---

## Question 1

**Which of the following is a primary reason why a standard Multi-Layer Perceptron (MLP) fails when directly applied to a flattened adjacency matrix of a graph?**

### ✅ Answer: **(A)**
> MLPs require fixed-size input layers, but real-world graphs have wildly varying numbers of nodes.

### Explanation

From the lecture, three core reasons are given for why MLPs fail on graph data:

1. **Fixed-size input constraint**: MLPs require a fixed-size input. Graphs have a variable number of nodes $N$, so the adjacency matrix $A \in \mathbb{R}^{N \times N}$ changes in size from graph to graph — MLPs cannot directly handle this.
2. **Wasted parameters / sparsity**: Real-world graphs are sparse ($O(N)$ edges), but the full adjacency matrix has $O(N^2)$ entries, making this representation inefficient.
3. **Permutation sensitivity**: Reordering the node indices changes the flattened adjacency matrix representation even though the underlying graph is identical — MLPs are **permutation-sensitive**, not permutation-invariant.

> **Why not (B)?** This is actually backwards. MLPs are permutation-*sensitive* (a flaw), not permutation-invariant. **(B) states the opposite of the correct property.** The lecture says MLPs are "permutation-sensitive", meaning they are *not* invariant — and this is indeed a problem, but option (A) is listed first and is the more primary/commonly cited reason in the lecture.
>
> **Why not (C)?** Real-world graphs are typically **sparse**, not dense. The adjacency matrix is large but has very few non-zero entries.
>
> **Why not (D)?** Sliding windows are a property of CNNs, not MLPs.

---

## Question 2

**Every layer of a Message Passing Neural Network (MPNN) consists of three distinct mathematical steps for each node. What is the correct sequence?**

### ✅ Answer: **(A)**
> **Message → Aggregate → Update**

### Explanation

The lecture defines the Message Passing Algorithm (Gilmer et al., 2017) with exactly these three steps in the following order, for every layer $t$ and every node $i$:

| Step | Formula | Description |
|------|---------|-------------|
| **1. Message** | $m_{j \to i}^{(t)} = M_t(h_i^{(t-1)}, h_j^{(t-1)}, e_{ji})$ | Each neighbor $j$ generates a message to target $i$ |
| **2. Aggregate** | $m_i^{(t)} = \text{AGGREGATE}(\{m_{j \to i}^{(t)} \mid j \in \mathcal{N}(i)\})$ | All incoming messages are pooled (permutation-invariant) |
| **3. Update** | $h_i^{(t)} = U_t(h_i^{(t-1)}, m_i^{(t)})$ | Node embedding is updated using its old state + aggregated message |

The sequence is logically required: you must **generate** messages before you can **aggregate** them, and only after aggregation can you **update** the node's embedding.

---

## Question 3

**In the Update step $h_i^{(t)} = U_t(h_i^{(t-1)}, m_i^{(t)})$, why is it explicitly necessary to include the target node's previous state $h_i^{(t-1)}$ alongside the aggregated neighborhood message?**

### ✅ Answer: **(A)**
> To prevent the target node from instantly forgetting its own unique features and simply becoming the average of its neighbors.

### Explanation

The lecture directly addresses this point:

> *"The Update function must balance 'who I am' ($h_i^{(t-1)}$) with 'who my friends are' ($m_i^{(t)}$)."*

If only the aggregated neighborhood message $m_i^{(t)}$ were used in the update, a node would lose all memory of its own original features and identity — it would collapse into a blend of its neighbors. Including $h_i^{(t-1)}$ preserves the node's own identity through each layer.

The typical implementation makes this concrete:
$$h_i^{(t)} = \text{ReLU}\!\left(W^{(t)} \cdot [h_i^{(t-1)} \| m_i^{(t)}]\right)$$

The concatenation `||` explicitly keeps both the node's own prior state and the neighborhood information.

> **Why not (B)?** Permutation invariance is ensured by the **Aggregate** step, not the Update step.
>
> **Why not (C)?** Edge feature updates are a separate, optional mechanism described in the lecture under "Edge Features" — not the purpose of including $h_i^{(t-1)}$ in the Update.
>
> **Why not (D)?** Converting to a sequence is characteristic of RNNs, not relevant here.

---

## Question 4

**In the propagation rule for a baseline GCN, how does the architecture normalize the aggregated messages between a target node $i$ and its neighbor $j$?**

### ✅ Answer: **(A)**
> By dividing the message by the square root of the product of both nodes' degrees: $\dfrac{1}{\sqrt{\deg(i)\,\deg(j)}}$

### Explanation

The GCN propagation rule (Kipf & Welling, 2017) from the lecture is:

$$H^{(t)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(t-1)} W^{(t-1)})$$

where $\hat{A} = A + I$ (adjacency with self-loops) and $\hat{D}$ is the corresponding degree matrix.

Written at the per-node level, this becomes:

$$h_i^{(t)} = \sigma\!\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\deg(i)\,\deg(j)}}\, h_j^{(t-1)} W^{(t-1)}\right)$$

This **symmetric normalization** ensures that messages from high-degree nodes are down-weighted — preventing highly-connected "hub" nodes from dominating the aggregation.

> **Why not (B)?** Edge-conditioned MLP weights are used in **GAT** (attention weights $\alpha_{ij}$) and edge-conditioned networks — not the baseline GCN.
>
> **Why not (C)?** Self-attention scores $\alpha_{i,j}$ are the mechanism of **GAT**, not GCN.
>
> **Why not (D)?** Neighborhood sampling is the strategy of **GraphSAGE**, not GCN.

---

## Question 5

**Which of the following is an example of an Edge-Level prediction task in a graph neural network?**

### ✅ Answer: **(A)**
> Recommending a new friend to a user in a social network.

### Explanation

The lecture defines three prediction task levels for GNNs:

| Task Level | Definition | Example from the Lecture |
|------------|-----------|--------------------------|
| **Node-level** | Predict a label/property of individual nodes | Spam detection, document classification |
| **Edge-level** | Predict the existence or type of an edge (link prediction) | **Recommendations**, knowledge graph completion |
| **Graph-level** | Predict a property of the entire graph | Drug toxicity, molecular properties |

Recommending a new friend is **link prediction** — predicting whether an edge (friendship) *should* exist between two nodes (users) in the graph. This is the canonical edge-level task.

> **Why not (B)?** Detecting a spam bot predicts a property of a **single node** → Node-level task.
>
> **Why not (C)?** Predicting toxicity of a whole molecule is a property of the **entire graph** → Graph-level task.
>
> **Why not (D)?** Classifying a research document is predicting a label for a **single node** → Node-level task.

---

## Summary Table

| Q | Answer | Key Concept |
|---|--------|-------------|
| 1 | **(A)** | MLPs need fixed-size input; graphs have variable $N$ |
| 2 | **(A)** | Message → Aggregate → Update (Gilmer et al., 2017) |
| 3 | **(A)** | Update preserves node identity: "who I am" + "who my friends are" |
| 4 | **(A)** | GCN symmetric normalization: $1/\sqrt{\deg(i)\deg(j)}$ |
| 5 | **(A)** | Friend recommendation = link prediction = Edge-level task |

> **Source**: Lecture 9 — *Graph Neural Networks*, STAT 426, George Michailidis.
