# Lecture 9: Graph Neural Networks

**STAT 426 | George Michailidis**

---

## 1. Motivation: Why Graph Neural Networks?

### Data Types Recap

- **Tabular data** → MLPs (fixed-size input/output)
- **Image data** → CNNs (fixed 2D/3D grid, sliding window convolutions)
- **Sequential data** → RNNs/LSTMs/GRUs (ordered observations)
- **Graph data** → GNNs (irregular topology, variable-size neighborhoods)

### Graph Data Applications

- **Social networks**: modeling relationships, influence propagation
- **Molecules**: predicting chemical properties based on structural layout
- **Citation networks**: knowledge graphs, recommendation systems
- **Traffic systems**: routing and congestion prediction

### Why Standard Architectures Fail on Graphs

- **MLP**: requires fixed-size input (graphs have variable $N$), wastes parameters on sparse adjacency matrices ($O(N^2)$ space for $O(N)$ edges), **permutation-sensitive** (reordering nodes changes the input but not the graph)
- **CNN**: assumes fixed grid topology with uniform neighborhood sizes and spatial ordering — graphs have arbitrary topology with variable-degree nodes
- **RNN**: linearizing a graph via traversal (DFS/random walk) loses structural integrity and introduces sequence bias — same graph produces different embeddings depending on traversal order
- **Attention/Transformers**: actually a natural fit — self-attention is inherently permutation equivariant and treats data as a fully connected graph; can be masked with the adjacency matrix

## 2. Graph Representations

### Mathematical Formulation

- Graph $G = (V, E)$ with vertices $V$ and edges $E$
- **Adjacency Matrix** $A \in \mathbb{R}^{N \times N}$: $A_{ij} = 1$ if edge exists between $i$ and $j$
- **Node Feature Matrix** $X \in \mathbb{R}^{N \times D}$: row $i$ contains the $D$-dimensional feature vector for node $i$
- **Edge Features** $E_f$ (optional): attributes like distance, type, strength

### Prediction Tasks

1. **Node-level** (classification/regression): predict label of individual nodes (e.g., spam detection, document classification)
2. **Edge-level** (link prediction): predict existence or type of edge (e.g., recommendations, knowledge graph completion)
3. **Graph-level** (classification/regression): predict property of entire graph (e.g., drug toxicity, molecular properties)

### Key Property: Permutation

- GNNs must be **permutation equivariant** (node tasks) or **permutation invariant** (graph tasks)
- Swapping node indices changes $A$ but not the underlying graph — architecture must be robust to this

## 3. The Message Passing Framework

### Core Idea: Neighborhood Aggregation

- Compute a context-aware embedding $h_v$ for every node $v$ that captures both its features and graph structure
- Stacking $K$ GNN layers expands the receptive field to $K$ hops (analogous to stacking CNN filters)

### Message Passing Algorithm (Gilmer et al., 2017)

Every layer $t$ consists of three steps for each node $i$:

1. **Message**: generate message from each neighbor $j$ to target $i$
   - $m_{j \to i}^{(t)} = M_t(h_i^{(t-1)}, h_j^{(t-1)}, e_{ji})$
2. **Aggregate**: combine all incoming messages (must be **permutation invariant**)
   - $m_i^{(t)} = \text{AGGREGATE}(\{m_{j \to i}^{(t)} \mid j \in \mathcal{N}(i)\})$
3. **Update**: update target node embedding using old state + aggregated message
   - $h_i^{(t)} = U_t(h_i^{(t-1)}, m_i^{(t)})$

### What Information Flows?

- **Base case** ($t=1$): $h_j^{(0)} = X_j$ — raw features, no topology
- **Recursive case** ($t>1$): $h_j^{(t-1)}$ is a compressed summary of $j$'s entire $(t-1)$-hop neighborhood
- **Update function** must balance "who I am" ($h_i^{(t-1)}$) with "who my friends are" ($m_i^{(t)}$)
- Typical implementation: $h_i^{(t)} = \text{ReLU}(W^{(t)} \cdot [h_i^{(t-1)} \| m_i^{(t)}])$ (concatenation + linear + activation)

### Edge Features

- Edge features $e_{ji}$ encode relationship type (bond type, interaction strength, etc.)
- **Strategy 1 (Concatenation)**: $m_{j \to i}^{(t)} = \text{MLP}(h_i^{(t-1)} \| h_j^{(t-1)} \| e_{ji})$
- **Strategy 2 (Edge-conditioned weights)**: $m_{j \to i}^{(t)} = \text{MLP}_{\text{edge}}(e_{ji}) \cdot h_j^{(t-1)}$
- Advanced GNNs can also **update edge features**: $e_{ji}^{(t)} = U_{\text{edge}}(e_{ji}^{(t-1)}, h_i^{(t-1)}, h_j^{(t-1)})$

## 4. Specific GNN Architectures

### GCN — Graph Convolutional Networks (Kipf & Welling, 2017)

- **Propagation rule**: $H^{(t)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(t-1)} W^{(t-1)})$
  - $\hat{A} = A + I$ (self-loops); $\hat{D}$ = degree matrix of $\hat{A}$
- Per-node: $h_i^{(t)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\deg(i)\deg(j)}} h_j^{(t-1)} W^{(t-1)}\right)$
- **Symmetric normalization**: down-weights messages from high-degree senders and receivers
- **Pros**: simple, efficient ($O(|E|)$ scaling), great baseline
- **Cons**: treats all neighbors equally (degree-based only), **transductive** (requires entire graph in memory), suffers from over-smoothing

### GraphSAGE (Hamilton et al., 2017)

- **Inductive framework** with **neighborhood sampling** — samples fixed number of neighbors (10–25)
- Breaks dependency on full graph; enables standard mini-batch training
- Can generate embeddings for **entirely unseen nodes**
- **Update rule**: $h_i^{(t)} = \sigma(W^{(t)} \cdot \text{CONCAT}(h_i^{(t-1)}, m_{\mathcal{N}(i)}^{(t)}))$
- **Aggregator choices**: Mean, Max-Pooling (MLP + element-wise max), LSTM (random permutation)
- Concatenates old state with message (preserves central node info better than GCN's addition)

### GAT — Graph Attention Networks (Velickovic et al., 2018)

- Introduces **learnable attention coefficients** $\alpha_{ij}$ for edge weighting
- $h_i^{(t)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} W h_j^{(t-1)}\right)$
- Network learns how important each neighbor is (vs. GCN's fixed degree-based weights)
- Effectively a **Transformer with the adjacency matrix as attention mask**
- Multi-head attention can be applied (multiple $\alpha$ sets in parallel)

### GIN — Graph Isomorphism Networks (Xu et al., 2019)

- Designed to **maximize expressive power** — provably as powerful as the 1-WL isomorphism test
- Strictly uses **SUM aggregation** + MLPs everywhere
- $h_i^{(t)} = \text{MLP}^{(t)}\left((1 + \epsilon^{(t)}) h_i^{(t-1)} + \sum_{j \in \mathcal{N}(i)} h_j^{(t-1)}\right)$
- Learnable $\epsilon$ distinguishes node's own features from neighbors'
- **Why SUM over MEAN/MAX**: mean loses count information (2 identical neighbors vs. 4 gives same result); sum preserves it
- Crucial for **graph-level tasks** (molecular chemistry) where structural counts define properties

### Graph Transformers

- Apply full node-to-node attention across the entire graph (not just local neighbors)
- Solves **over-squashing** by creating direct $O(1)$ paths between all nodes
- **Trade-off**: $O(N^2)$ compute/memory — primarily for smaller graphs (molecules)
- **Scaling solutions**: sparse attention, linear attention (Performers), hybrid local-global models
- **Topology injection** (since global attention loses structure):
  - Laplacian positional encodings (eigenvectors of $L = D - A$)
  - Structural encodings: node degree, eigenvector centrality, shortest path distance

## 5. Architecture Comparison

| Architecture | Key Innovation | Aggregation | Best Use Case |
|---|---|---|---|
| GCN | Spectral approximation | Degree-normalized sum | Transductive node classification |
| GraphSAGE | Neighbor sampling | Mean / Max-Pool / LSTM | Massive graphs, inductive learning |
| GAT | Learned edge attention | Attention-weighted sum | Heterogeneous importance neighbors |
| GIN | WL-test expressivity | Sum + MLP | Graph classification (drug discovery) |

## 6. Training Issues

### The Expressivity Problem

- Standard message passing (GCN, GraphSAGE, GAT) is upper-bounded by the **1-Weisfeiler-Lehman test**
- Cannot distinguish certain symmetric structures (e.g., hexagon vs. two triangles)
- **AGGREGATE choice matters**: Mean/Max lose count info; only Sum reaches 1-WL limit
- Graph Transformers: expressivity depends entirely on positional/structural encodings

### Over-smoothing (The Convergence Problem)

- Repeated aggregation causes all node representations to **blur together** across layers
- Analogy: mixing too many paint colors until everything becomes uniform
- Limits practical depth of GNNs

### Over-squashing (The Bottleneck Problem)

- Neighborhood size grows **exponentially** with hops; all information compressed into fixed-size vector
- Distant signals get lost as noise
- Analogy: compressing a 1000-page textbook into a single tweet

## 7. Practical Training Notes

### Deep GNN Regularization

- **Residual (skip) connections**: $H^{(l+1)} = \text{GNN}(H^{(l)}) + H^{(l)}$ — gradient highway, combats over-smoothing
- **DropEdge**: randomly remove edges during training — regularizer that slows over-smoothing

### Normalization

- **BatchNorm** (node-wise): normalizes each feature across all nodes in batch; best for large single-graph tasks
- **LayerNorm** (feature-wise): normalizes across feature dim per node; best for variable graph sizes / Graph Transformers
- Applied after aggregation, before activation

### Scalability & Mini-Batching

- **Neighborhood explosion**: computing $K$-layer embedding requires loading $K$-hop neighborhood — can require entire graph
- **Neighbor sampling** (GraphSAGE): cap neighbors per node per layer (e.g., 10)
- **Subgraph sampling** (Cluster-GCN): cluster graph into dense subgraphs, train as isolated mini-batches
