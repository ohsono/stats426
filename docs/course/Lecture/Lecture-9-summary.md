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

- **MLP**: 
  - Requires fixed-size input (graphs have variable $N$)
  - Wastes parameters on sparse adjacency matrices ($O(N^2)$ space for $O(N)$ edges)
  - **Permutation-sensitive:** If we feed a graph into an MLP, we have to flatten the Adjacency Matrix into a 1D vector. To do this, we arbitrarily number the nodes 1, 2, 3...
  - If we change the arbitrary numbering (e.g., node 1 becomes node 3), the 1D input vector changes completely. The MLP will think it's looking at an entirely new, unrecognizable graph, even though the actual physical structure hasn't changed at all. GNNs don't care what arbitrary label you give a node.
- **CNN**: assumes fixed grid topology with uniform neighborhood sizes and spatial ordering — graphs have arbitrary topology with variable-degree nodes
- **RNN**: linearizing a graph via traversal (DFS/random walk) loses structural integrity and introduces sequence bias — same graph produces different embeddings depending on traversal order
- **Attention/Transformers**: actually a natural fit — self-attention is inherently permutation equivariant and treats data as a fully connected graph; can be masked with the adjacency matrix

## 2. Graph Representations

### Anatomy of a Graph: Features (Attributes/Covariates)

A real-world graph isn't just a skeleton of dots and lines. It is mathematically defined as $G = (V, E, X, E_f)$, which represents four distinct layers of information required for Machine Learning.

#### The Raw Skeleton
- **$V$ (Vertices/Nodes):** The naked list of dots (e.g., `[Node 1, Node 2, Node 3]`). We have no idea *what* these nodes actually represent yet.
- **$E$ (Edges):** The list of connections between the dots (e.g., `[(Node 1, Node 2)]`).

*If you only use V and E, you know the shape of the network, but you don't know anything about the actual data.*

#### The Features (Attributes/Covariates)
This is where we "hang" actual machine-learning data onto the raw skeleton.

- **$X$ (Node Features):** We replace the empty labels with actual $D$-dimensional data vectors.
  - **For a molecule:** The vector might hold `[atomic mass, charge, number of valence electrons]`.
  - **For a social network:** The vector might hold `[age, location coordinate, posts per day]`.

If a graph has $|V|$ nodes (e.g., $N=4$ nodes named A, B, C, D) and each has $D=3$ features, the matrix $X \in \mathbb{R}^{|V| \times D}$ looks like this:
```
xA = [0.1, 1.4, 0]  <-- Node A's features
xB = [0.8, 0.2, 1]  <-- Node B's features
xC = [0.5, 0.9, 0]  <-- Node C's features
xD = [0.3, 0.1, 1]  <-- Node D's features
```

- **$E_f$ (Edge Features):** Just as nodes have identities, the *connections* between them can have identities. These can be simple scalars or full vectors describing the relationship.
  - **For a molecule:** Is this a single bond, a double bond, or a hydrogen bond?
  - **For a traffic map:** What is the speed limit, distance, and current congestion level of this road?

> **Summary:** The variables $V$ and $E$ tell the GNN *how* to build the message-passing highways (who can talk to whom). The matrices $X$ and $E_f$ act as the actual "passengers" driving on those highways.



### Prediction Tasks

1. **Node-level** (classification/regression): predict label of individual nodes (e.g., spam detection, document classification)
2. **Edge-level** (link prediction): predict existence or type of edge (e.g., recommendations, knowledge graph completion)
3. **Graph-level** (classification/regression): predict property of entire graph (e.g., drug toxicity, molecular properties)

### Key Property: Permutation

**The mathematical labels we assign to nodes are completely arbitrary.** If we swap the labels of Node 1 and Node 2, the adjacency matrix $A$ changes mathematically, but the physical graph structure hasn't changed at all.

Therefore, GNNs must obey two golden rules depending on the task:

**1. Permutation Invariant (For Graph tasks)**
- "If I shuffle the input node labels, the final answer stays exactly the same."
- **Example:** Predicting if a molecule is toxic. A molecule doesn't care how you numbered its atoms. The final True/False toxicity answer must remain identical regardless of permutation.

**2. Permutation Equivariant (For Node tasks)**
- "If I shuffle the input node labels, the output predictions shuffle in the exact same way."
- **Example:** Classifying users as bots vs. real humans. If I have User A and User B, and the network predicts [User A = Bot, User B = Human]...
- If I completely swap the input data (feeding User B first, then User A), the network must simply swap its output predicting [User B = Human, User A = Bot].
- It shouldn't give me a completely different set of predictions just because the ordering changed. The output follows (is *equivariant* to) the input permutation.

## 3. The Message Passing Framework

### Core Idea: Neighborhood Aggregation (Message Passing Framework)

Instead of looking at the whole graph at once, the GNN looks at each node and its immediate friends (neighbors). In each "Layer" of a GNN:

1. **Message:** Every neighbor generates a "message" containing its current features.
   - $m_{j \to i}^{(t)} = M_t(h_i^{(t-1)}, h_j^{(t-1)}, e_{ji})$
2. **Aggregate:** The target node collects all incoming messages from its neighbors and combines them (using rules like Sum, Mean, or Max). This step must be **permutation invariant**.
   - $m_i^{(t)} = \text{AGGREGATE}(\{m_{j \to i}^{(t)} \mid j \in \mathcal{N}(i)\})$
3. **Update:** The target node updates its own feature vector by blending its *old* identity with the *aggregated messages* of its friends.
   - $h_i^{(t)} = U_t(h_i^{(t-1)}, m_i^{(t)})$

**The Multi-Layer Effect:**
- **Layer 1:** A node only knows about itself and its 1-hop friends.
- **Layer 2:** Its friends now contain information from *their* friends. So the target node now knows about its 2-hop neighborhood.
- By stacking $K$ layers, every node gets a context-aware embedding covering a $K$-hop radius (analogous to expanding receptive fields in CNNs).

### What Information Flows? (The Evolution of a Node)

As a graph passes through multiple layers of a GNN, the information inside each node's feature vector $h$ evolves. Here is exactly what that vector represents at each stage:

**The Base Case ($t=0$):**
- $h_i^{(0)} = X_i$
- Before the GNN does any message passing, node $i$'s state is just its **raw input features** ($X$). At this exact moment, the node is completely blind to the graph structure. It only knows about itself.

**The Recursive Case ($t>0$):**
- $h_i^{(t-1)}$
- After passing through previous layers, the vector $h$ is no longer just "Node $i$'s features." It has absorbed information from its neighbors. Therefore, $h_i^{(t-1)}$ acts as a **compressed mathematical summary** of Node $i$'s entire $(t-1)$-hop neighborhood.

**The Update Function (The Identity Crisis):**
When calculating the *next* state $h_i^{(t)}$, the GNN has to make a crucial decision: how much should I change my identity based on what my friends are saying?
- It must balance **"who I am currently"** ($h_i^{(t-1)}$) with **"what my friends are telling me"** ($m_i^{(t)}$).
- **Typical implementation (GraphSAGE-style):**
  $$h_i^{(t)} = \text{ReLU}\left(W^{(t)} \cdot [h_i^{(t-1)} \Vert m_i^{(t)}]\right)$$
  - `[ || ]` : Concatenate (glue together) my old state and my friends' aggregated message.
  - `W` : Multiply by a learned weight matrix (so the network can learn *which* pieces of my identity vs my friends' identities are most important).
  - `ReLU` : Apply a non-linear activation to make the network capable of learning complex, non-linear patterns.

### How Do We Use Edge Features?

Sometimes the connections between nodes contain critical data (e.g., in a molecule, a double bond is very different from a single bond). We represent this edge data mathematically as $e_{ji}$ (the feature of the edge pointing from node $j$ to node $i$).

How do we actually inject this edge data into the "message" ($m_{j \to i}^{(t)}$) that neighbor $j$ sends to target $i$? There are two main strategies:

**Strategy 1: Concatenation (The "All-in-One" Approach)**
- **Formula:** $m_{j \to i}^{(t)} = \text{MLP}(h_i^{(t-1)} \Vert h_j^{(t-1)} \Vert e_{ji})$
- **Plain English:** We take the target node's identity, the neighbor's identity, and the edge feature, glue them all together into one massive array (`||` means concatenate), and feed it through a standard Neural Network (MLP). The MLP figures out how to mix them.
- **Detailed Example:** Imagine a social network where we are generating a message from Bob ($j$) to Alice ($i$):
  - Target Alice ($h_i$) has a vector of size 10 (e.g., her age, location).
  - Neighbor Bob ($h_j$) has a vector of size 10 (e.g., his age, location).
  - The Edge ($e_{ji}$) has a vector of size 5 (e.g., how many years they've been friends).
  - **The Glue:** We concatenate (`||`) all three vectors into a single 1D array of size 25 ($10 + 10 + 5$).
  - **The MLP:** We feed this size-25 array into an MLP (a standard feedforward neural network). The MLP is trained to look at this combined 25-number profile and output a final "Message Vector" describing the influence Bob has on Alice.

**Strategy 2: Edge-conditioned weights (The "Filter" Approach)**
- $m_{j \to i}^{(t)} = \text{MLP}_{\text{edge}}(e_{ji}) \cdot h_j^{(t-1)}$
- **Plain English:** First, we feed *only* the edge feature into a small neural network to generate a **weight matrix**. Then, we multiply the neighbor's message by this weight matrix.
- **Why?** It treats the edge like a physical filter. A "double bond" edge literally alters the geometric shape of the message flowing through it differently than a "single bond" edge does.

**Advanced: Updating Edge Features**
- $e_{ji}^{(t)} = U_{\text{edge}}\left(e_{ji}^{(t-1)}, h_i^{(t-1)}, h_j^{(t-1)}\right)$
- **Plain English:** In standard GNNs, edges are static pipes that data flows through. But in advanced GNNs, **the pipes themselves can learn and evolve**. At every layer, the edge updates its own feature ($e_{ji}$) based on its past state, plus the current states of the two nodes it connects. This is crucial for complex physics/chemistry simulations.

## 4. Specific GNN Architectures (The 5 Flavors)

### 1. GCN (Graph Convolutional Networks)
**The Personality:** "The Democratic Averager"

- **How it works:** To figure out what Node A should look like in the next layer, it literally takes an average of all of Node A's friends' features. 
- **The "Symmetric Normalization" trick:** If Node A has a friend who is a massive celebrity with 10 million friends, that celebrity's voice could drown out everyone else. To fix this, GCN divides the message from the celebrity by a large number (specifically, the square root of their degree). It prevents highly connected nodes from overwhelming the network.
- **The Catch:** Because it requires you to mathematically multiply the *entire* Adjacency Matrix by the *entire* Feature Matrix at once, the whole graph has to fit in your computer's memory. This is called being **transductive**. It cannot handle graphs the size of Twitter/X.
- **Update Rule:** $H^{(t)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(t-1)} W^{(t-1)})$

### 2. GraphSAGE (Hamilton et al., 2017)
**The Personality:** "The Scalable Sampler"

- **How it works:** Instead of trying to mathematically average *all* of a node's friends, GraphSAGE randomly samples a fixed number of them (like grabbing 10 random friends). 
- **The Superpower:** Because it only looks at a small, fixed sample, you don't need the whole graph in memory. You can process data in "mini-batches." Even better, if a brand new user joins the network tomorrow, GraphSAGE knows how to generate an embedding for them instantly (this is called being **inductive**).
- **Update rule**: $h_i^{(t)} = \sigma(W^{(t)} \cdot \text{CONCAT}(h_i^{(t-1)}, \text{AGG}(\{h_j^{(t-1)}\})))$

### 3. GAT (Graph Attention Networks)
**The Personality:** "The VIP List"

- **How it works:** GCN treats all friends as equal. GraphSAGE picks friends at random. GAT introduces a learned knob called $\alpha_{ij}$ (attention) to say *"Some friends are more important than others."* 
- **The Superpower:** When Node A looks at its friends, it might decide "B is mostly irrelevant ($\alpha=0.1$), but D has the exact information I need ($\alpha=0.9$)." The network *learns* these attention weights during training based on the actual features of the nodes.
- **Update Rule:** $h_i^{(t)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} W h_j^{(t-1)}\right)$

### 4. GIN (Graph Isomorphism Networks)
**The Personality:** "The Strictly Accurate Mathematician"

- **How it works:** Designed for extreme mathematical power (e.g., predicting drug toxicity). Strictly uses **SUM aggregation** instead of Mean/Max.
- **Why Mean fails:** Imagine Node A is a Carbon atom connected to 2 Oxygen atoms. Node B is a Carbon atom connected to 4 Oxygen atoms. If you take the *average* features of the Oxygen neighbors, both Node A and Node B get the exact same mathematical result. The network becomes blind to the fact that Node B has twice as many bonds!
- **The Superpower:** By forcing the use of SUM, the network explicitly preserves **count** information. Because of this, GIN is mathematically proven to reach the theoretical ceiling of GNN expressivity (the 1-Weisfeiler-Lehman test limit).
- **Update Rule:** $h_i^{(t)} = \text{MLP}^{(t)}\left((1 + \epsilon^{(t)}) h_i^{(t-1)} + \sum_{j \in \mathcal{N}(i)} h_j^{(t-1)}\right)$

### 5. Graph Transformers
**The Personality:** "The Rule Breaker"

- **How it works:** A standard GNN strictly obeys the edges. Node A can only talk to Node B if a line connects them. Graph Transformers break that rule. They let every single node in the graph look at every other node simultaneously using Global Attention.
- **The Superpower:** This completely solves the **"Over-squashing"** problem. Node A doesn't have to wait 10 layers for a message to slowly hop across 10 edges from Node Z. It can look directly at Node Z in step 1.
- **The Catch:** If every node looks at every other node, the math scales as $O(N^2)$ compute/memory. It is incredibly memory intensive and usually only used for small graphs (like small molecules). Because it ignores the edges, you have to use "Positional Encodings" (like Laplacian eigenvectors) to artificially remind the network what the original topology actually looked like.

## 5. Architecture Comparison

| Architecture | Key Innovation | Aggregation | Best Use Case |
|---|---|---|---|
| GCN | Spectral approximation | Degree-normalized sum | Transductive node classification |
| GraphSAGE | Neighbor sampling | Mean / Max-Pool / LSTM | Massive graphs, inductive learning |
| GAT | Learned edge attention | Attention-weighted sum | Heterogeneous importance neighbors |
| GIN | WL-test expressivity | Sum + MLP | Graph classification (drug discovery) |

## 6. Training Issues (The 3 GNN Nightmares)

Deep GNNs are notoriously hard to train due to three unique problems:

### 1. The Expressivity Problem (1-WL Test Limit)

- GNNs are mathematically blind to certain shapes. Because they just collect neighbor features, they often cannot tell the difference between two completely different global graph structures (e.g., a hexagon vs. two separate triangles).
- Standard message passing (GCN, GraphSAGE, GAT) is upper-bounded by the theoretical ceiling of the **1-Weisfeiler-Lehman (1-WL) test**.
- **AGGREGATE choice matters**: Mean/Max lose count info; only Sum aggregation (GIN) reaches the 1-WL limit.

### 2. Over-smoothing (The Blur Problem)

- If you build a 50-layer GNN, a node takes in information from friends, friends-of-friends... all the way out to 50 hops.
- Eventually, everyone is sharing information with everyone else. Like mixing too many paint colors, all the node embeddings blur together into a **uniform grey sludge** where every node looks identical.
- Limits practical depth of GNNs to usually just 2-4 layers.

### 3. Over-squashing (The Bottleneck Problem)

- The number of neighbors grows exponentially with every hop. By hop 5, a single node might be trying to summarize information from 10,000 distant nodes.
- All this information is forced to squeeze into a single tiny, fixed-size feature vector.
- Distant signals get completely crushed and lost as noise (Analogy: compressing a 1,000-page textbook into a single tweet).

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
