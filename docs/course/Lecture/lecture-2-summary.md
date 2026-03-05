# Feedforward Neural Networks (Multilayer Perceptrons)

**George Michailidis**
gmichail@ucla.edu
STAT 426

---

## Table of Contents
1. [Linear Neural Networks](#linear-neural-networks)
2. [Multilayer Perceptrons (MLPs)](#multilayer-perceptrons-mlps)
3. [MLP Examples by Task](#mlp-examples-by-task)
4. [Forward and Backward Propagation](#forward-and-backward-propagation)
5. [End-to-End Learning](#end-to-end-learning)
6. [Illustration: MLP for Binary Classification](#illustration-mlp-for-binary-classification)
7. [Binary Classification Evaluation](#binary-classification-evaluation)
8. [Experimental Setup](#experimental-setup-for-binary-classification)
9. [Regularization Strategies](#regularization-strategies-in-deep-learning)

---

## Linear Neural Networks

### Building and Training a Predictive Model

To build a predictive model `y = f(x) + error` within the Statistical Learning Theory framework, we require:

- **Data**: A training set D = {(xᵢ, yᵢ)}ⁿᵢ₌₁ drawn from spaces X × Y
- **Model**: A hypothesis class H (the set of all possible functions f of interest)
- **Criterion for Success**: A loss function L(y, f(x)) to measure how well the model f predicts labels y

Using the **Empirical Risk Minimization** framework, we find the "best" f ∈ H by minimizing average loss:

```
f̂ = argmin (1/n) Σᵢ L(f(xᵢ), yᵢ)
     f∈H
```

### Recap: Linear Regression

**Data:**
- Input: Feature vector x ∈ ℝᵖ
- Outcome: Continuous scalar y ∈ ℝ

**Model Parameterization:**
```
f(x) = β₀ + Σⱼ βⱼxⱼ = βᵀx
```

*Key Assumption:* The relationship is **globally linear** and **additive**.

**Loss Function: Squared Error Loss**
```
L(y, f(x)) = (y - f(x))² = (y - βᵀx)²
```

**Optimization Objective (Least Squares Criterion):**
```
β̂ = argmin (1/n) Σᵢ (yᵢ - βᵀxᵢ)²
    β∈ℝᵖ
```

**Analytic Solution (Normal Equations):**
```
β̂ = (XᵀX)⁻¹Xᵀy
```

### Feedforward Neural Networks: Key Components

1. **Architecture** - How data flows from input to output (Layers, Weights, Biases)
2. **Loss Function** - Measures prediction error
3. **Optimization** - The algorithm that updates parameters to minimize loss (e.g., Minibatch Stochastic Gradient Descent)

### Linear Regression from a Neural Network Perspective

**Model Architecture:** A single fully connected layer with identity activation:
```
ŷ = wᵀx + b
```

Input x ∈ ℝᵖ → Single Output Neuron → Identity Function → Output ŷ

**Loss Function (MSE):**
```
L(w, b) = (1/n) Σᵢ (1/2)(wᵀxᵢ + b - yᵢ)²
```

**Optimization Goal:** Find parameters ŵ, b̂ that minimize the loss:
```
ŵ, b̂ = argmin L(w, b)
        w,b
```

### The Neural Shift (Linear Regression)

- **Structural Equivalence:** Linear regression corresponds exactly to a linear neural network consisting of a single fully connected layer and an identity activation function.

- **Notation Translation:**
  - β₀ (Intercept) → b (Bias)
  - β (Coefficients) → w (Weights)

- **Algorithmic Shift:** Instead of closed-form analytic solution, we use **iterative optimization** (Stochastic Gradient Descent). Iterative methods scale better to massive datasets where matrix inversion is computationally prohibitive.

### Recap: Logistic Regression Model

**Key Task:** Predict a binary categorical label y ∈ {0, 1} given input features x ∈ ℝᵖ.

**The Probability Model (Sigmoid):**
```
πᵢ(xᵢ, β) = P(yᵢ = 1 | xᵢ, β) = 1 / (1 + exp(-xᵢᵀβ))
```

**Likelihood Function:**
```
L(β) = Πᵢ [πᵢ]^yᵢ [1 - πᵢ]^(1-yᵢ)
```

**Log-Likelihood Function:**
```
ℓ(β) = Σᵢ [yᵢ log πᵢ + (1 - yᵢ) log(1 - πᵢ)]
```

**Loss Function: Negative Log-Likelihood (Cross-Entropy)**
```
L(y, π(x)) = -[y log π(x) + (1 - y) log(1 - π(x))]
```

**Solution (No Closed Form):** Unlike linear regression, taking the gradient and setting it to zero does not yield a closed-form solution. We must use iterative methods such as Newton-Raphson or Gradient Descent.

### Logistic Regression from Neural Network Perspective

**Model Architecture:** A single fully connected layer with sigmoid activation:
```
ŷ = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx+b)))
```

**Loss Function (Binary Cross-Entropy):**
```
L(w, b) = -(1/n) Σᵢ [yᵢ log ŷᵢ + (1 - yᵢ) log(1 - ŷᵢ)]
```

### Recap: Multi-class Classification Model

**Key Task:** Predict a categorical label y ∈ {1, ..., K} given input features x ∈ ℝᵖ.

**The Probability Model (Softmax):**
```
πₖ(x, B) = P(y = k | x) = exp(βₖᵀx) / Σⱼ exp(βⱼᵀx)
```

### One-Hot Encoding for Multiclass Labels

The label y is a vector of length K with:
- Exactly one entry equal to 1
- All other entries equal to 0

Example (K = 4):
| Class | One-hot encoded y |
|-------|-------------------|
| cat   | (1, 0, 0, 0)ᵀ     |
| dog   | (0, 1, 0, 0)ᵀ     |
| bird  | (0, 0, 1, 0)ᵀ     |
| fish  | (0, 0, 0, 1)ᵀ     |

**Loss Function: Categorical Cross-Entropy**
```
L(y, ŷ) = -Σₖ yₖ log ŷₖ
```

---

## Multilayer Perceptrons (MLPs)

### The Limitation of Linear Models

**The Problem:** Single-layer networks (Linear/Logistic/Softmax) are limited in their expressiveness.
- They imply a monotonic relationship: increasing xᵢ must always increase (or always decrease) the output
- They cannot solve the **XOR Problem** or handle complex interactions between features

**To model complex data, we need Hidden Layers and Non-linearities.**

### Incorporating Hidden Layers

**Architecture Change:**
1. **Input Layer:** x ∈ ℝᵈ
2. **Hidden Layer:** h ∈ ℝʰ
   - Fully connected to inputs
   - Output of this layer becomes input to the next
3. **Output Layer:** o ∈ ℝᵍ

This architecture is called a **Multilayer Perceptron (MLP)**.

### Mathematical Formulation

Consider an MLP with single hidden layer with h hidden units (depth=1, width=h).

**Step 1: The Hidden Layer Calculation**
```
h = σ(W⁽¹⁾x + b⁽¹⁾)
```

**Step 2: The Output Layer Calculation**
```
o = W⁽²⁾h + b⁽²⁾
```

**Dimensions:**
- Input x ∈ ℝᵈ, Batch size n → X ∈ ℝⁿˣᵈ
- Weights W⁽¹⁾ ∈ ℝᵈˣʰ, Bias b⁽¹⁾ ∈ ℝ¹ˣʰ
- Weights W⁽²⁾ ∈ ℝʰˣᵍ, Bias b⁽²⁾ ∈ ℝ¹ˣᵍ

**Crucial Element:** σ(·) is a non-linear activation function.

### Why Activation Functions?

**Without non-linearity σ, the hidden layer is mathematically useless:**
```
h = W⁽¹⁾x + b⁽¹⁾
o = W⁽²⁾h + b⁽²⁾

Substituting:
o = W⁽²⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾
  = (W⁽²⁾W⁽¹⁾)x + (W⁽²⁾b⁽¹⁾ + b⁽²⁾)
      ︸────︸       ︸──────────︸
       Wₙₑw           bₙₑw
```

An affine function of an affine function is still an affine function - it collapses into a single linear model.

### Activation Functions

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(x, 0)
```
- **Piecewise linear:** Preserves many optimization properties
- **Computation:** Extremely fast (simple thresholding)
- **Gradient:** 1 if x > 0, 0 if x < 0. Avoids "vanishing gradient" for positive inputs

#### Sigmoid
```
σ(x) = 1 / (1 + e⁻ˣ)
```
- Range: (0, 1)
- Used in: Binary output layers, LSTM gates
- Issue: Vanishing gradients (derivative near 0 for large |x|)

#### Tanh (Hyperbolic Tangent)
```
tanh(x) = (1 - e⁻²ˣ) / (1 + e⁻²ˣ)
```
- Range: (-1, 1)
- Centered at 0 (often converges faster than sigmoid)
- Issue: Still suffers from vanishing gradients

### General Notation for Deep Networks

We can stack multiple hidden layers. Let L be the number of layers.

**Recursive definition:**
```
h⁽⁰⁾ = x (Input)
h⁽ˡ⁾ = σₗ(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)  for l = 1, ..., L
```

### Universal Approximation Theorem

**Statement:** A feedforward neural network with:
- a single hidden layer,
- a finite number of neurons, and
- a nonlinear, non-polynomial activation function (e.g., sigmoid, tanh, ReLU),

can approximate **any continuous function** f : K ⊂ ℝⁿ → ℝᵐ uniformly on compact sets K, to arbitrary precision.

**Formally:** For any ε > 0, there exists a network fθ such that:
```
sup ‖f(x) - fθ(x)‖ < ε
x∈K
```

**Important caveats:**
- This is an **existence** result (not a training guarantee)
- The number of required neurons may be very large
- Approximation does not imply good generalization

---

## MLP Examples by Task

### Example 1: MLP for Regression

**Task:** Predict a continuous value y ∈ ℝ (e.g., House Price)

**Architecture:**
- Hidden Layer: ReLU activation: `h = ReLU(W⁽¹⁾x + b⁽¹⁾)`
- Output Layer: Identity activation: `ŷ = w⁽²⁾ᵀh + b⁽²⁾`

**Loss Function:** Mean Squared Error (MSE)
```
L(y, ŷ) = (1/2)‖y - ŷ‖²
```

### Example 2: MLP for Binary Classification

**Task:** Predict a class y ∈ {0, 1} (e.g., Spam vs Not Spam)

**Architecture:**
- Hidden Layer: ReLU activation: `h = ReLU(W⁽¹⁾x + b⁽¹⁾)`
- Output Layer: Sigmoid activation: `ŷ = σ(w⁽²⁾ᵀh + b⁽²⁾)`

**Loss Function:** Binary Cross-Entropy
```
L(y, ŷ) = -[y log ŷ + (1 - y) log(1 - ŷ)]
```

### Example 3: MLP for Multi-class Classification

**Task:** Predict y ∈ {1, ..., K} (e.g., Digit 0-9)

**Architecture:**
- Hidden Layer: ReLU activation: `h = ReLU(W⁽¹⁾x + b⁽¹⁾)`
- Output Layer: Softmax activation with K nodes: `ŷₖ = exp(oₖ) / Σⱼ exp(oⱼ)`

**Loss Function:** Categorical Cross-Entropy
```
L(y, ŷ) = -Σₖ yₖ log ŷₖ
```

### Summary: The MLP Framework

1. **Linearity is not enough:** To learn complex patterns, we need hidden layers
2. **Non-linearity is Key:** We must insert an activation function after every hidden affine transformation
3. **Architecture defines Task:**
   - Input: Matches data dimension (d)
   - Hidden: Hyperparameter (width and depth)
   - Output: Matches target (1 for regression/binary, K for multi-class)
4. **Optimization:** We use (Stochastic) Gradient Descent with gradients calculated via **Backpropagation** (Chain Rule)

---

## Forward and Backward Propagation

### The Training Pipeline

Training involves a cyclical **Forward-Backward Loop**:

1. **Forward Propagation:**
   - Input data flows through the network layer by layer
   - Produces the prediction ŷ and the scalar loss L
   - Must cache intermediate values (activations) for later use

2. **Backward Propagation (Backprop):**
   - Gradients flow in reverse direction (from Loss to Input)
   - Applies the Chain Rule recursively
   - Computes gradient of loss w.r.t. every parameter: ∇wL

3. **Parameter Update:**
   - Adjust weights using computed gradients (e.g., SGD)

### Step 1: Forward Propagation

Let h⁽⁰⁾ = x. For each layer l = 1, ..., L:

1. **Affine Transformation (Pre-activation):**
   ```
   z⁽ˡ⁾ = W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾
   ```

2. **Non-linear Activation:**
   ```
   h⁽ˡ⁾ = σₗ(z⁽ˡ⁾)
   ```

3. **Compute Loss (at the end):**
   ```
   J = L(h⁽ᴸ⁾, y) + λΩ(W)  (Regularization optional)
   ```

**Memory Cost:** During Forward Prop, we must store z⁽ˡ⁾ and h⁽ˡ⁻¹⁾ for backward pass.

### Step 2: Backward Propagation

We want to find ∂L/∂W⁽ˡ⁾. Apply the **Chain Rule** starting from output layer L.

1. **Output Error Signal:**
   ```
   δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾ = (∂L/∂ŷ) · σ'(z⁽ᴸ⁾)
   ```

2. **Backpropagate to Hidden Layers:** For l = L-1, ..., 1:
   ```
   δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ σ'(z⁽ˡ⁾)
   ```
   (⊙ is the element-wise Hadamard product)

3. **Compute Weight Gradients:**
   ```
   ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(h⁽ˡ⁻¹⁾)ᵀ
   ```

### Computational Considerations

1. **Memory Intensity**
   - Cannot throw away h⁽ˡ⁾ after forward pass
   - GPU memory is often the bottleneck
   - Trade-off: Deeper networks = Linear increase in memory

2. **Vanishing/Exploding Gradients**
   - In δ⁽ˡ⁾, we multiply by Wᵀ and σ' repeatedly
   - If W is small or σ' is near 0 (Sigmoid), gradients vanish → No learning
   - **Solution:** ReLU + Batch Normalization

---

## End-to-End Learning

### The Paradigm Shift

**Traditional Machine Learning Pipeline:**
- Broken into separate, manually tuned stages
- Example: Image → Hand-crafted Features → Classifier (SVM) → Output
- Limitation: Errors accumulate; feature extractor not optimized for task

**End-to-End Deep Learning:**
- Entire system treated as a single, differentiable function
- Map raw inputs directly to outputs
- **Feature Learning:** Network learns its own internal representations

### Implications

1. **Global Optimization:** Backpropagation updates all parameters simultaneously
2. **Data Hunger:** E2E models require significantly more data to generalize well
3. **Interpretability Challenges:** Intermediate representations may not be interpretable

---

## Illustration: MLP for Binary Classification

### Mathematical Formulation

**1. Input Layer:** x = [x₁, ..., xd]ᵀ ∈ ℝᵈ

**2. Hidden Layer:**
- Parameters: W₁ ∈ ℝᵠˣᵈ, b₁ ∈ ℝᵠ
- Pre-activation: a = W₁x + b₁
- Activation: h = ReLU(a) ∈ ℝᵠ

**3. Output Layer:**
- Parameters: W₂ ∈ ℝ¹ˣᵠ, b₂ ∈ ℝ
- Logit: z = W₂h + b₂
- Prediction: ŷ = σ(z) ∈ (0, 1)

### Backward Pass Details

**Binary Cross-Entropy Loss:**
```
L(y, ŷ) = -[y log(ŷ) + (1 - y) log(1 - ŷ)]
```

**Step 1: Output Layer Gradients**
```
∂L/∂z = (ŷ - y)
∂L/∂W₂ = (ŷ - y) · hᵀ  ∈ ℝ¹ˣᵠ
∂L/∂b₂ = (ŷ - y)       ∈ ℝ
```

**Step 2: Propagating to Hidden Layer**
```
∂L/∂h = W₂ᵀ · (ŷ - y)  ∈ ℝᵠˣ¹
∂L/∂a = (∂L/∂h) ⊙ 1ₐ>₀  ∈ ℝᵠˣ¹
```

**Step 3: Input Layer Gradients**
```
∂L/∂W₁ = (∂L/∂a) · xᵀ  ∈ ℝᵠˣᵈ
∂L/∂b₁ = ∂L/∂a         ∈ ℝᵠˣ¹
```

### Parameter Initialization

**Why is it important?**

1. **Symmetry Breaking:** If all weights initialized to same constant, every neuron computes same output and receives same gradient
2. **Gradient Stability:** Too large → exploding gradients; Too small → vanishing gradients

**Initialization Methods:**

- **Xavier (Glorot) Initialization** (for Sigmoid/Tanh):
  ```
  W ~ U(-√(6/(dᵢₙ + dₒᵤₜ)), √(6/(dᵢₙ + dₒᵤₜ)))
  ```

- **He Initialization** (for ReLU):
  ```
  W ~ N(0, 2/dᵢₙ)
  ```

- **Biases:** b₁ = 0, b₂ = 0 (typically initialized to zero)

### Batch vs. Stochastic Gradient Descent

1. **Batch Gradient Descent:** Uses entire dataset - slow and expensive for large n

2. **Stochastic Gradient Descent (SGD) with Mini-batches:**
   - Randomly sample small subset of size s (e.g., 32 or 64)
   - Compute gradient and update weights based only on this batch
   - Benefit: Much faster updates and helps escape bad local minima

### What is an Epoch?

**Definition:** An Epoch represents one complete pass through the entire training dataset.

- If we have n observations and batch size s, we must process n/s batches to complete 1 epoch
- **Iteration:** One single update of model parameters
- **Total Iterations = num_epochs × (n/s)**

### Model Training, Tuning and Evaluating

Split data into three disjoint sets:

1. **Training Set (≈ 60-70%):** Used to compute gradients and update parameters
2. **Validation Set (≈ 15-20%):** Used to tune hyperparameters (hidden dimension, batch size)
3. **Test Set (≈ 15-20%):** Used only once at the end to estimate real-world generalization

**Golden Rule: Never train on the test set!**

---

## Binary Classification Evaluation

### The Confusion Matrix

|           | Predicted Positive | Predicted Negative |
|-----------|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Evaluation Metrics

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
*Limitation: Misleading for imbalanced classes*

**Precision (Positive Predictive Value):**
```
Precision = TP / (TP + FP)
```
*Of all instances predicted positive, how many were actually positive?*

**Recall (Sensitivity / TPR):**
```
Recall = TP / (TP + FN)
```
*Of all actual positive instances, how many did the model correctly identify?*

**F1 Score:**
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```
*Harmonic mean of Precision and Recall*

**False Positive Rate (FPR):**
```
FPR = FP / (FP + TN)
```

**Specificity (True Negative Rate):**
```
Specificity = TN / (TN + FP) = 1 - FPR
```

### Which Metric to Use?

- **Imbalanced Classes:** Avoid Accuracy. Use Precision, Recall, or F1 Score
- **High Cost of False Negatives (e.g., Medical Tests):** Prioritize High Recall
- **High Cost of False Positives (e.g., Spam Detection):** Prioritize High Precision

### ROC Curve and AUC

**ROC Curve:** Plots TPR (y-axis) vs. FPR (x-axis) at various classification thresholds t ∈ (0, 1)
- Visualizes the trade-off between sensitivity and specificity
- Perfect classifier reaches top-left corner (TPR = 1, FPR = 0)

**AUC (Area Under the ROC Curve):**
- Range: 0.5 (Random Guessing) to 1.0 (Perfect Classifier)
- **Interpretation:** The probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance
  ```
  AUC = P(s(x⁺) > s(x⁻))
  ```

---

## Experimental Setup for Binary Classification

### Setup

- **Sample Size:** N = 10,000
  - Training: 6,000
  - Validation: 2,000
  - Test: 2,000
- **Features:** p = 100 covariates with average correlation > 0.4
- **Data Generation:** yᵢ generated via logistic regression model with c% label noise

### Hyperparameters for MLP

- Epochs: 50
- Learning rate
- Hidden Dimension (q)
- Mini-batch Size (s)

---

## Regularization Strategies in Deep Learning

### 1. Dropout: Robustness via Noise Injection

**Key Intuition:**
- A well-trained model should be robust to small perturbations
- Dropout prevents **co-adaptation:** Neurons cannot rely on specific other neurons

**The Mechanism:** Given hidden activation h, dropout replaces it with h':
```
h' = 0           with probability p
h' = h/(1-p)     with probability 1-p
```

**Inverted Dropout:** Divide by (1-p) during training to keep expected value unchanged: E[h'] = h

**Training Phase:** For every mini-batch, a different random subset of neurons is deactivated

**Testing Phase:** Dropout is disabled. All neurons are active.

### 2. Weight Decay (ℓ₂ Regularization)

Adds a penalty term to the loss function:
```
Lᵣₑg = Lₐₐₜₐ + (λ/2)||w||²
```
- Encourages weights to remain small/diffuse
- Small weights mean the function is smoother and less sensitive to input changes

### 3. Early Stopping

- Stop training when **Validation Error starts to rise**, even if Training Error is still falling
- Acts as a proxy for controlling model complexity

### 4. Data Augmentation

- Artificially expanding the dataset (e.g., rotations, flips, noise for images)
- Forces invariance in the model

---

## Summary

This lecture covered the fundamentals of Feedforward Neural Networks (Multilayer Perceptrons):

### Key Takeaways

1. **From Linear to Non-linear:** Single-layer networks (linear/logistic regression) are limited. Adding hidden layers with non-linear activations enables learning complex patterns.

2. **Core Components:** Architecture (layers, weights, biases), Loss Function, and Optimization Algorithm (SGD).

3. **Activation Functions:** ReLU is preferred for hidden layers (avoids vanishing gradients); Sigmoid for binary output; Softmax for multi-class.

4. **Training Process:**
   - **Forward Pass:** Compute predictions and loss
   - **Backward Pass:** Compute gradients via chain rule (backpropagation)
   - **Update:** Adjust parameters using gradient descent

5. **Universal Approximation:** MLPs can theoretically approximate any continuous function, but this is an existence result, not a training guarantee.

6. **Practical Considerations:**
   - Proper initialization (Xavier/He) prevents vanishing/exploding gradients
   - Mini-batch SGD for efficient training
   - Train/Validation/Test splits for proper evaluation
   - Regularization (Dropout, Weight Decay, Early Stopping) prevents overfitting

7. **Evaluation Metrics:** For classification, use appropriate metrics (Precision, Recall, F1, AUC-ROC) based on the problem context, especially for imbalanced datasets.

### Mathematical Framework

| Model Type | Output Activation | Loss Function |
|------------|------------------|---------------|
| Regression | Identity | MSE |
| Binary Classification | Sigmoid | Binary Cross-Entropy |
| Multi-class Classification | Softmax | Categorical Cross-Entropy |
