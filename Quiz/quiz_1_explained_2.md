# STAT 426: Quiz 1 Explained (Questions 18-31)

## Part 2: Neural Networks & Deep Learning

---

### Question 18

**What happens if you stack multiple hidden layers in a neural network but use only linear (identity) activation functions?**

- (a) The network becomes a deep non-linear classifier.
- (b) The network collapses mathematically into a single linear model.
- (c) The network suffers from exploding gradients immediately.
- (d) The training speed increases, but the model overfits.

**Correct Answer: (b)**

**Explanation:**
With identity activations σ(z) = z, each layer computes a linear transformation:

**Layer 1:** h₁ = W₁x + b₁
**Layer 2:** h₂ = W₂h₁ + b₂ = W₂(W₁x + b₁) + b₂ = W₂W₁x + W₂b₁ + b₂
**Layer 3:** h₃ = W₃h₂ + b₃ = W₃W₂W₁x + ...

The composition of linear functions is **still linear**:

$$f(x) = W_L W_{L-1} \cdots W_1 x + \tilde{b} = \tilde{W}x + \tilde{b}$$

Where W̃ = W_L W_{L-1} ⋯ W₁ is just another matrix!

**Key insight:** No matter how many layers you stack, without non-linear activations, the entire network is equivalent to a **single linear transformation**. Depth provides no additional expressive power.

This is why non-linear activations (ReLU, Sigmoid, Tanh) are essential—they allow networks to learn non-linear decision boundaries.

Why other options are wrong:

- (a) Linear activations cannot create non-linear classifiers
- (c) Exploding gradients aren't caused by linear activations specifically
- (d) A linear model can't overfit complex non-linear patterns

---

### Question 19

**Which of the following describes the Rectified Linear Unit (ReLU) activation function?**

- (a) f(x) = max(0, x); it helps avoid the vanishing gradient problem for positive inputs.
- (b) f(x) = 1/(1+e⁻ˣ); it squashes inputs to the range (0, 1).
- (c) f(x) = tanh(x); it is centered at zero.
- (d) f(x) = x²; it is a polynomial activation function.

**Correct Answer: (a)**

**Explanation:**
**ReLU (Rectified Linear Unit):**

$$f(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Gradient:**
$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Why ReLU helps with vanishing gradients:**

- For positive inputs, gradient = 1 (constant, never shrinks)
- Compare to Sigmoid: gradient max ≈ 0.25, shrinks exponentially through layers
- Deep networks with Sigmoid: gradients → 0 (vanishing)
- Deep networks with ReLU: gradients stay at 1 for active neurons

**ReLU advantages:**

- Computationally efficient (just a threshold)
- Sparse activation (many neurons output 0)
- Mitigates vanishing gradient for x > 0

**ReLU disadvantage:**

- "Dying ReLU": neurons with negative inputs have zero gradient, can't recover

Why other options are wrong:

- (b) This describes **Sigmoid**
- (c) This describes **Tanh**
- (d) x² is not standard; would cause exploding activations

---

### Question 20

**What does the Universal Approximation Theorem guarantee about a feedforward network with a single hidden layer and sufficient neurons?**

- (a) It can be trained to zero error using Gradient Descent.
- (b) It can approximate any continuous function to arbitrary precision.
- (c) It will generalize perfectly to unseen test data.
- (d) It requires a polynomial activation function to work.

**Correct Answer: (b)**

**Explanation:**
**Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991):**

> A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of ℝⁿ to arbitrary accuracy, given a suitable non-linear activation function.

**What it guarantees:**

- **Existence**: Such an approximation exists
- **Arbitrary precision**: Can get as close as desired (with enough neurons)

**What it does NOT guarantee:**

- How to find the weights (learnability)
- How many neurons are needed (could be exponentially many)
- Generalization to unseen data
- Computational efficiency

**Practical implications:**

- Single hidden layer is theoretically sufficient
- But deep networks often achieve same accuracy with fewer total parameters
- "Can approximate" ≠ "can learn efficiently"

Why other options are wrong:

- (a) Doesn't guarantee trainability or zero error
- (c) Says nothing about generalization
- (d) Works with various activations (sigmoid, tanh, ReLU), not just polynomials

---

### Question 21

**During the Backward Propagation phase of training, what is the primary mathematical rule used to calculate gradients?**

- (a) The Product Rule.
- (b) Integration by Parts.
- (c) The Chain Rule.
- (d) The Pythagorean Theorem.

**Correct Answer: (c)**

**Explanation:**
**Backpropagation** computes gradients of the loss with respect to all parameters by applying the **Chain Rule** of calculus.

**Chain Rule:**
If y = f(g(x)), then:
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**In neural networks:**

```
Input → Layer 1 → Layer 2 → ... → Layer L → Loss
  x    →   h₁    →   h₂    → ... →   ŷ    →  L
```

To find ∂L/∂W₁ (gradient for first layer weights):

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_{L-1}} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

**Backprop algorithm:**

1. **Forward pass**: Compute all activations, store intermediate values
2. **Backward pass**: Propagate gradients from output to input using chain rule
3. Gradients flow backward through the network, hence "backpropagation"

Why other options are wrong:

- (a) Product rule is used occasionally but chain rule is fundamental
- (b) Integration by parts is for integrals, not relevant here
- (d) Pythagorean theorem is for geometry, not calculus

---

### Question 22

**How does "End-to-End" Deep Learning differ from traditional Machine Learning pipelines?**

- (a) It relies on manually engineered features extracted before training.
- (b) It optimizes the feature extraction and classification steps simultaneously as a single differentiable function.
- (c) It uses separate loss functions for every layer of the network.
- (d) It requires less training data because it uses human prior knowledge.

**Correct Answer: (b)**

**Explanation:**
**Traditional ML Pipeline:**

```
Raw Data → [Hand-crafted Features] → [Classifier] → Prediction
              (SIFT, HOG, etc.)      (SVM, etc.)

- Features designed by domain experts
- Feature extraction and classification are separate
- Each stage optimized independently
```

**End-to-End Deep Learning:**

```
Raw Data → [Deep Neural Network] → Prediction
           (learns features automatically)

- Single differentiable function
- Features learned from data
- All parameters optimized jointly via backpropagation
```

**Key difference:** In end-to-end learning, the **entire pipeline** from raw input to output is differentiable. Gradients flow from the loss all the way back to the first layer, jointly optimizing feature extraction AND classification.

**Example:** Image classification

- Traditional: Design edge detectors, color histograms → feed to SVM
- End-to-end: Raw pixels → CNN → class probabilities (features emerge in early layers)

Why other options are wrong:

- (a) This describes traditional ML, not end-to-end
- (c) End-to-end uses a single loss function at the output
- (d) End-to-end typically requires MORE data (no human priors)

---

### Question 23

**Why is initializing all weights to zero (W = 0) a catastrophic strategy for training a Multi-Layer Perceptron?**

- (a) It causes the gradients to explode immediately.
- (b) It prevents the network from learning non-linear functions because the activation functions become linear.
- (c) It creates perfect symmetry where every neuron in a hidden layer learns the exact same feature, making the layer act like a single neuron.
- (d) It causes the optimizer to get stuck in a local minimum immediately.

**Correct Answer: (c)**

**Explanation:**
**The Symmetry Problem:**

If all weights are initialized to zero (or any identical value):

1. **Forward pass**: All neurons in a layer receive the same input and compute the same output
   - h₁ = σ(0·x + 0) = σ(0) for all neurons

2. **Backward pass**: All neurons receive the same gradient
   - ∂L/∂w is identical for all weights in a layer

3. **Update**: All weights updated by the same amount
   - All neurons remain identical!

**Result:** Every neuron in a layer computes the **exact same function** forever. A layer with 100 neurons effectively acts as 1 neuron.

**Breaking symmetry:**

- Initialize weights **randomly** (different values)
- Each neuron starts differently → learns different features
- Common schemes: Xavier/Glorot, He initialization

Why other options are wrong:

- (a) Zero weights cause vanishing gradients (all zero), not exploding
- (b) Activations don't become linear; the issue is symmetry
- (d) It's not a local minimum issue; neurons simply can't differentiate

---

### Question 24

**What is the primary goal of specialized initialization schemes like Xavier (Glorot) or Kaiming (He) initialization?**

- (a) To ensure the weights are all positive.
- (b) To keep the variance of activations and gradients roughly constant across layers, preventing them from vanishing or exploding.
- (c) To ensure that the initial loss is exactly zero.
- (d) To force the weights to be sparse (mostly zero) from the start.

**Correct Answer: (b)**

**Explanation:**
**The Variance Problem in Deep Networks:**

During forward pass, if variance grows or shrinks per layer:

- **Variance grows**: Activations explode → numerical overflow
- **Variance shrinks**: Activations vanish → information loss

During backward pass, same issue with gradients:

- **Gradients explode**: Unstable updates
- **Gradients vanish**: No learning in early layers

**Xavier/Glorot Initialization (for tanh/sigmoid):**
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

**He/Kaiming Initialization (for ReLU):**
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

**Goal:** Choose variance so that:
$$\text{Var}(h_l) \approx \text{Var}(h_{l-1})$$

This keeps signal strength roughly constant through the network, enabling stable training of deep networks.

Why other options are wrong:

- (a) Weights should be both positive and negative (centered at 0)
- (c) Initial loss doesn't matter; we're optimizing it anyway
- (d) Sparsity in weights is a different goal (regularization)

---

### Question 25

**If you use ReLU activations, which initialization method is theoretically preferred to maintain variance through the network?**

- (a) Xavier (Glorot) Initialization, which assumes linear or symmetric activations.
- (b) Kaiming (He) Initialization, which explicitly accounts for the fact that ReLU zeroes out half the inputs.
- (c) Zero Initialization.
- (d) Random Normal initialization with standard deviation σ = 0.01.

**Correct Answer: (b)**

**Explanation:**
**Why ReLU needs different initialization:**

ReLU(x) = max(0, x) sets ~50% of activations to zero (for symmetric inputs around 0).

This **halves the variance** of the output compared to a linear activation!

**Derivation:**

- For linear activation: Var(output) = Var(input) × Var(weights) × n_in
- For ReLU: Var(output) = 0.5 × Var(input) × Var(weights) × n_in

To compensate, **He initialization** doubles the variance:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

Compare to Xavier:
$$W \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right) \text{ or } \mathcal{N}\left(0, \frac{2}{n_{in}+n_{out}}\right)$$

**Summary:**

| Activation | Recommended Init |
|------------|------------------|
| Tanh, Sigmoid | Xavier/Glorot |
| ReLU, Leaky ReLU | He/Kaiming |

Why other options are wrong:

- (a) Xavier assumes activations preserve variance (not true for ReLU)
- (c) Zero initialization causes symmetry problem (Q23)
- (d) σ = 0.01 is arbitrary and likely causes vanishing activations in deep networks

---

### Question 26

**Why is the "Validation Set" distinct from the "Test Set" in a proper machine learning pipeline?**

- (a) The Validation Set is used to calculate the gradients, while the Test Set is used to update the weights.
- (b) The Validation Set is used to tune hyperparameters and select the best model, while the Test Set provides an unbiased estimate of final performance.
- (c) The Validation Set is always smaller than the Test Set.
- (d) There is no difference; the terms are interchangeable.

**Correct Answer: (b)**

**Explanation:**
**Three-way data split:**

| Dataset | Purpose | Used During Training? |
|---------|---------|----------------------|
| **Training Set** | Fit model parameters (weights) | Yes |
| **Validation Set** | Tune hyperparameters, model selection, early stopping | Indirectly (guides decisions) |
| **Test Set** | Final unbiased performance estimate | No (only at the very end) |

**Why validation ≠ test:**

When you use validation data to:

- Choose learning rate
- Select number of layers
- Decide when to stop training
- Pick between model architectures

You're **optimizing** (indirectly) on the validation set. The model's validation performance becomes biased—it's no longer a true estimate of generalization.

**The test set must remain untouched** until final evaluation. It provides the only unbiased estimate of how your model performs on truly unseen data.

**Analogy:**

- Validation = Practice exam (you learn from it)
- Test = Final exam (reveals true knowledge)

Why other options are wrong:

- (a) Neither is used for gradients or weight updates directly
- (c) Relative size is not the defining difference
- (d) They serve fundamentally different purposes

---

### Question 27

**What happens if you use your Test Set to tune your hyperparameters (e.g., selecting the number of layers)?**

- (a) You effectively increase the size of your training data.
- (b) You introduce "Data Leakage," resulting in an overly optimistic estimate of the model's generalization error.
- (c) You guarantee that the model will not overfit.
- (d) The model training time decreases significantly.

**Correct Answer: (b)**

**Explanation:**
**Data Leakage** occurs when information from outside the training data influences model development.

**If you tune hyperparameters on test data:**

1. You try hyperparameter set A → test accuracy 85%
2. You try hyperparameter set B → test accuracy 87%
3. You select B because it performed better on test data
4. You report 87% as your model's performance

**The problem:** You've **fit your choices** to the test set!

- The 87% is now biased (optimistic)
- True performance on new data might be 82%
- You've effectively "trained" on the test set through your decisions

**This is overfitting at the hyperparameter level:**

- Just like models can overfit to training data
- Your model selection process can overfit to test data

**Proper procedure:**

1. Tune hyperparameters on **validation set**
2. Select best model based on validation performance
3. Evaluate **once** on test set for final unbiased estimate

Why other options are wrong:

- (a) Doesn't increase data; corrupts the evaluation process
- (c) Actually increases risk of overfitting (to test set)
- (d) Training time is unaffected

---

### Question 28

**Which of the following is considered a "Hyperparameter" rather than a learnable parameter?**

- (a) The weights matrix W connecting the input to the hidden layer.
- (b) The bias vector b.
- (c) The learning rate η used by the optimizer.
- (d) The final output probabilities.

**Correct Answer: (c)**

**Explanation:**
**Parameters vs Hyperparameters:**

| Type | Learned from data? | Examples |
|------|-------------------|----------|
| **Parameters** | Yes (via gradient descent) | Weights W, biases b |
| **Hyperparameters** | No (set before training) | Learning rate, batch size, # layers, # neurons, regularization λ |

**Learning rate ($\eta$)** is a classic hyperparameter:

- Set before training begins
- Not updated by gradient descent
- Affects how parameters are updated: W ← W - η∇L
- Must be tuned (grid search, random search, etc.)

**Other common hyperparameters:**

- Network architecture (depth, width)
- Activation functions
- Optimizer choice (SGD, Adam)
- Batch size
- Number of epochs
- Dropout rate
- Weight decay coefficient

Why other options are wrong:

- (a) Weights W are learned via backpropagation
- (b) Biases b are learned via backpropagation
- (d) Output probabilities are computed from learned parameters

---

### Question 29

**What does the ROC (Receiver Operating Characteristic) curve visualize?**

- (a) The trade-off between Precision and Recall at different thresholds.
- (b) The trade-off between the True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity) at various classification thresholds.
- (c) The change in Training Loss vs. Validation Loss over time.
- (d) The accuracy of the model across different classes.

**Correct Answer: (b)**

**Explanation:**
**ROC Curve axes:**

- **X-axis**: False Positive Rate (FPR) = FP / (FP + TN) = 1 - Specificity
- **Y-axis**: True Positive Rate (TPR) = TP / (TP + FN) = Sensitivity = Recall

**How it's constructed:**

1. Model outputs probability scores for each sample
2. Vary the classification threshold from 0 to 1
3. At each threshold, calculate (FPR, TPR)
4. Plot all points

**Interpreting ROC:**

```
TPR
 1 |     ____------
   |   /
   | /    Good classifier
   |/     (curves toward top-left)
 0 +--------------- FPR
   0              1
```

- **Perfect classifier**: Goes straight up then right (0,0)→(0,1)→(1,1)
- **Random classifier**: Diagonal line from (0,0) to (1,1)
- **Good classifier**: Curves toward upper-left corner

**ROC vs PR curve:**

- ROC: TPR vs FPR (answer b)
- PR curve: Precision vs Recall (answer a)—different visualization!

Why other options are wrong:

- (a) This describes the **Precision-Recall curve**, not ROC
- (c) This describes a **learning curve**
- (d) ROC is for binary classification trade-offs, not multi-class accuracy

---

### Question 30

**What does an AUC (Area Under the Curve) of 0.5 indicate for a binary classifier?**

- (a) The model is a perfect classifier.
- (b) The model performs no better than random guessing.
- (c) The model predicts the opposite class perfectly (it is perfectly wrong).
- (d) The model has zero variance.

**Correct Answer: (b)**

**Explanation:**
**AUC (Area Under ROC Curve)** summarizes the ROC curve as a single number:

| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect classifier |
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.7 - 0.8 | Fair |
| 0.5 - 0.7 | Poor |
| **0.5** | **Random guessing (no discrimination)** |
| < 0.5 | Worse than random (predictions inverted) |

**Why AUC = 0.5 means random:**

- Random classifier has ROC = diagonal line
- Area under diagonal = 0.5 × 1 × 1 = 0.5
- Model cannot distinguish between classes better than chance

**Probabilistic interpretation:**
AUC = P(model ranks a random positive higher than a random negative)

- AUC = 0.5: 50% chance = coin flip
- AUC = 1.0: 100% chance = perfect ranking

Why other options are wrong:

- (a) Perfect classifier has AUC = 1.0
- (c) Perfectly wrong classifier has AUC = 0.0 (can be fixed by inverting predictions)
- (d) Zero variance is unrelated to AUC

---

### Question 31

**When training a Deep Neural Network, what phenomenon is indicated if the Training Loss decreases, but the Validation Loss starts to increase?**

- (a) Underfitting.
- (b) Overfitting.
- (c) Vanishing Gradient.
- (d) Proper convergence.

**Correct Answer: (b)**

**Explanation:**
**Classic overfitting signature:**

```
Loss
  |
  |  \
  |   \____  Training Loss (keeps decreasing)
  |         \_____
  |
  |      ______
  |     /      \____  Validation Loss (decreases then increases!)
  |    /
  +------------------------→ Epochs
       ↑
    Overfitting starts here
```

**What's happening:**

1. Early training: Model learns general patterns → both losses decrease
2. Later training: Model starts memorizing training data specifics
3. Training loss continues to decrease (memorization improves)
4. Validation loss increases (memorized patterns don't generalize)

**Gap = Generalization Error:**

- The growing gap between training and validation loss indicates overfitting
- Model is fitting noise/idiosyncrasies in training data

**Solutions:**

- **Early stopping**: Stop when validation loss starts increasing
- **Regularization**: Dropout, weight decay, data augmentation
- **More data**: Harder to memorize larger datasets
- **Simpler model**: Reduce capacity

Why other options are wrong:

- (a) Underfitting: Both training AND validation loss are high (model too simple)
- (c) Vanishing gradient: Training loss stops decreasing (gradients → 0)
- (d) Proper convergence: Both losses decrease and stabilize

---

## Summary Table

| Q# | Topic | Correct | Key Concept |
|----|-------|---------|-------------|
| 18 | Linear activations | (b) | Network collapses to single linear model |
| 19 | ReLU | (a) | max(0,x), avoids vanishing gradient |
| 20 | Universal Approximation | (b) | Can approximate any continuous function |
| 21 | Backpropagation | (c) | Chain Rule for gradients |
| 22 | End-to-End Learning | (b) | Joint optimization of features + classifier |
| 23 | Zero initialization | (c) | Symmetry problem—neurons learn same thing |
| 24 | Xavier/He init | (b) | Maintain variance across layers |
| 25 | ReLU initialization | (b) | He init (accounts for ReLU zeroing half) |
| 26 | Validation vs Test | (b) | Validation tunes; Test gives unbiased estimate |
| 27 | Test set leakage | (b) | Data leakage → optimistic estimates |
| 28 | Hyperparameters | (c) | Learning rate (not learned from data) |
| 29 | ROC curve | (b) | TPR vs FPR trade-off |
| 30 | AUC = 0.5 | (b) | Random guessing performance |
| 31 | Train↓ Val↑ | (b) | Overfitting |
