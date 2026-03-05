# STAT 426: Quiz 1 Explained (Questions 1-17)

## Part 1: Foundations of Statistical Learning

---

### Question 1

**How do Neural Networks (specifically MLPs) fundamentally differ from traditional basis expansion methods like Splines or Fourier Regression regarding feature representation?**

- (a) Neural Networks rely on fixed basis functions determined before training begins.
- (b) Neural Networks use adaptive basis functions where internal parameters are learned from the data.
- (c) Neural Networks guarantee a convex optimization landscape for the loss function.
- (d) Neural Networks can not model non-linear relationships without explicit feature engineering.

**Correct Answer: (b)**

**Explanation:**
The key distinction between neural networks and traditional methods lies in how they create features:

- **Traditional methods (Splines, Fourier)**: Use **fixed basis functions** that are predetermined before seeing any data. For example, Fourier regression uses sine and cosine functions at fixed frequencies, and splines use polynomial pieces at fixed knot locations.

- **Neural Networks**: Use **adaptive basis functions** where the hidden layers learn representations from the data itself. The weights connecting layers are learned during training, allowing the network to discover useful features automatically.

Why other options are wrong:

- (a) This describes traditional methods, not neural networks
- (c) Neural networks typically have **non-convex** loss landscapes with many local minima
- (d) Neural networks excel at learning non-linear relationships **without** manual feature engineering—that's their main advantage

---

### Question 2

**In the context of Statistical Learning Theory, why is the Empirical Risk Minimization (ERM) principle insufficient on its own when the hypothesis class H is very complex?**

- (a) It leads to underfitting because the model can not capture the data patterns.
- (b) It is computationally impossible to calculate the empirical risk for complex models.
- (c) It can lead to overfitting, where the model memorizes training data but fails to generalize to the true distribution.
- (d) It requires knowledge of the true distribution P(X, Y), which is unknown.

**Correct Answer: (c)**

**Explanation:**
Empirical Risk Minimization (ERM) minimizes the average loss on the training data:

$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^{n} L(f(x_i), y_i)$$

When the hypothesis class H is very complex (high capacity), ERM can find a function that perfectly fits the training data, including noise. This leads to **overfitting**:

- Low training error (memorizes training data)
- High test error (fails to generalize)

The **bias-variance tradeoff** explains this:

- Complex H → Low approximation error (bias)
- Complex H → High estimation error (variance)

Why other options are wrong:

- (a) Underfitting occurs with **simple** hypothesis classes, not complex ones
- (b) Computing empirical risk is straightforward—just average the losses
- (d) ERM specifically avoids needing P(X,Y) by using training data instead

---

### Question 3

**Why do classification algorithms typically optimize "surrogate" loss functions (like Hinge or Logistic loss) instead of the 0/1 Loss?**

- (a) Surrogate losses are always bounded between 0 and 1, unlike the 0/1 loss.
- (b) The 0/1 Loss leads to NP-hard combinatorial optimization problems because it is non-convex and non-differentiable.
- (c) Surrogate losses effectively transform the classification problem into a clustering problem.
- (d) The 0/1 Loss biases the model towards the majority class.

**Correct Answer: (b)**

**Explanation:**
The 0/1 Loss is defined as:

$$L_{0/1}(f(x), y) = \mathbf{1}[f(x) \neq y]$$

This loss has two critical problems:

1. **Non-differentiable**: It's a step function, so gradients don't exist (or are zero everywhere except at discontinuities)
2. **Non-convex**: The optimization landscape has many local minima

These properties make gradient-based optimization impossible. Finding the optimal classifier under 0/1 loss is an **NP-hard combinatorial problem**.

**Surrogate losses** (Hinge, Logistic, Cross-Entropy) are:

- **Convex**: Single global minimum
- **Differentiable**: Gradients exist for optimization
- **Upper bounds** on 0/1 loss: Minimizing them also reduces misclassification

Why other options are wrong:

- (a) The 0/1 loss IS bounded between 0 and 1; surrogate losses can be unbounded
- (c) Surrogates don't convert classification to clustering
- (d) Class imbalance is a separate issue, not related to 0/1 loss properties

---

### Question 4

**Which of the following best describes the "Bayes Optimal Predictor" f*?**

- (a) It is the model that achieves zero error on the training dataset.
- (b) It represents the theoretical lower bound of risk, limited only by the irreducible error inherent in the data distribution.
- (c) It is a specific algorithm (like Naive Bayes) used for text classification.
- (d) It is the predictor obtained by averaging all models in the hypothesis class.

**Correct Answer: (b)**

**Explanation:**
The **Bayes Optimal Predictor** f* is defined as:

$$f^* = \arg\min_f R(f) = \arg\min_f \mathbb{E}_{(X,Y) \sim P}[L(f(X), Y)]$$

Key properties:

- It minimizes the **true risk** over ALL possible functions (not just those in H)
- Its risk R(f*) is called the **Bayes Risk** or **irreducible error**
- This error comes from inherent noise/overlap in the data distribution—no model can do better

For classification with 0/1 loss:
$$f^*(x) = \arg\max_y P(Y=y|X=x)$$

For regression with squared loss:
$$f^*(x) = \mathbb{E}[Y|X=x]$$

Why other options are wrong:

- (a) Zero training error describes overfitting, not optimality
- (c) "Naive Bayes" is a specific algorithm; "Bayes Optimal" is a theoretical concept
- (d) Model averaging is ensemble learning, not the Bayes optimal predictor

---

### Question 5

**What is "Inductive Bias" in the context of selecting a Hypothesis Class H?**

- (a) The error introduced when the training data is not representative of the population.
- (b) The systematic error that occurs when a model is too simple to capture the underlying trend.
- (c) The set of assumptions a learning algorithm makes about the function to predict outputs for unseen inputs.
- (d) The intercept term β₀ in a regression equation.

**Correct Answer: (c)**

**Explanation:**
**Inductive Bias** refers to the assumptions a learning algorithm makes that allow it to generalize from seen training examples to unseen test examples.

Examples of inductive biases:

- **Linear models**: Assume the relationship is linear
- **Decision trees**: Assume axis-aligned decision boundaries
- **Neural networks**: Assume hierarchical feature composition
- **k-NN**: Assume similar inputs have similar outputs (smoothness)

Without inductive bias, a learner cannot generalize—it could only memorize training data. The "No Free Lunch Theorem" states that no single algorithm works best for all problems; the right inductive bias depends on the problem.

Why other options are wrong:

- (a) This describes **sampling bias**, not inductive bias
- (b) This describes **approximation error** or underfitting
- (d) The intercept β₀ is just a parameter, not a bias concept

---

### Question 6

**Why might a practitioner choose the ℓ₁ Loss (Absolute Error) over the ℓ₂ Loss (Squared Error) for a regression task?**

- (a) The ℓ₁ loss is differentiable everywhere, making optimization easier.
- (b) The ℓ₁ loss penalizes small errors more heavily than large errors.
- (c) The ℓ₁ loss is more robust to outliers because it estimates the median rather than the mean.
- (d) The ℓ₁ loss guarantees a unique solution.

**Correct Answer: (c)**

**Explanation:**
The two losses behave differently:

| Property | ℓ₁ (Absolute) | ℓ₂ (Squared) |
|----------|---------------|--------------|
| Formula | \|y - ŷ\| | (y - ŷ)² |
| Optimal estimate | **Median** | **Mean** |
| Outlier sensitivity | **Robust** | Sensitive |
| Large error penalty | Linear | Quadratic |

**Why ℓ₁ is robust to outliers:**

- Squaring large errors (ℓ₂) makes them dominate the loss
- An outlier with error 100 contributes 100 to ℓ₁ but 10,000 to ℓ₂
- The median (ℓ₁ solution) is a robust statistic; the mean (ℓ₂) is not

Why other options are wrong:

- (a) FALSE—ℓ₁ is **not** differentiable at zero (has a "kink")
- (b) FALSE—ℓ₁ penalizes large errors LESS than ℓ₂ (linear vs quadratic)
- (d) FALSE—ℓ₁ can have multiple solutions (any value between certain data points)

---

### Question 7

**What is the primary obstacle to calculating the True Risk R(f) in a real-world scenario?**

- (a) The loss function L is usually unknown.
- (b) The joint probability distribution P(X, Y) is unknown.
- (c) The hypothesis class H has infinite members.
- (d) Computer precision is insufficient for the integration.

**Correct Answer: (b)**

**Explanation:**
The True Risk is defined as:

$$R(f) = \mathbb{E}_{(X,Y) \sim P}[L(f(X), Y)] = \int L(f(x), y) \, dP(x, y)$$

To compute this, we need to integrate over the **true joint distribution P(X, Y)**—but we never know this distribution! We only have samples from it (our training data).

This is why we use **Empirical Risk** as a proxy:

$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^{n} L(f(x_i), y_i)$$

The Law of Large Numbers guarantees that as n → ∞, the empirical risk converges to the true risk.

Why other options are wrong:

- (a) We choose the loss function ourselves (MSE, Cross-Entropy, etc.)
- (c) Infinite hypothesis class doesn't prevent risk calculation for a specific f
- (d) Numerical precision is not the fundamental barrier

---

### Question 8

**Consider the decomposition of Excess Risk: R(f̂) − R(f*) = Approx. Error + Est. Error. If you have an infinite amount of training data (n → ∞) but your hypothesis class H is a simple linear model trying to fit a complex sine wave, which term remains non-zero?**

- (a) The Estimation Error (Variance).
- (b) The Approximation Error (Bias).
- (c) Both terms converge to zero.
- (d) The Generalization Gap.

**Correct Answer: (b)**

**Explanation:**
The **Excess Risk Decomposition**:

$$R(\hat{f}) - R(f^*) = \underbrace{[R(f_H^*) - R(f^*)]}_{\text{Approximation Error (Bias)}} + \underbrace{[R(\hat{f}) - R(f_H^*)]}_{\text{Estimation Error (Variance)}}$$

Where:

- $f^*$ = Bayes optimal predictor (over all functions)
- $f_H^*$ = Best function in hypothesis class H
- $\hat f$ = Learned function from finite data

**With infinite data (n → ∞):**

- **Estimation Error → 0**: With infinite samples, we can perfectly identify $f_H^*$ within our hypothesis class
- **Approximation Error remains**: If H (linear models) cannot represent the true function (sine wave), there's an irreducible gap

A linear model cannot approximate a sine wave well, no matter how much data you have. This is the **bias** from choosing too simple a model.

Why other options are wrong:

- (a) Estimation error vanishes with infinite data
- (c) Approximation error doesn't depend on sample size
- (d) Generalization gap is related to estimation error, which vanishes

---

### Question 9

**From a geometric perspective, why does Lasso Regression (ℓ₁ penalty) often result in sparse solutions (coefficients exactly zero) while Ridge Regression (ℓ₂ penalty) does not?**

- (a) The ℓ₁ optimization problem is non-convex, leading to local minima at zero.
- (b) The level sets of the ℓ₁ norm are "diamond-shaped" and tend to intersect the loss contours at the coordinate axes (corners).
- (c) Lasso ignores the data and focuses purely on the penalty term.
- (d) Ridge regression uses a squared term which explodes for small coefficients, forcing them away from zero.

**Correct Answer: (b)**

**Explanation:**
The geometric intuition is best understood with an analogy:

**The "Beach Ball" and the "Fence":**
Imagine expanding a balloon (the loss function/error contours) until it touches a fence (the regularization constraint). The solution is the exact point where they touch.

**1. ℓ₂ (Ridge) = The Circular Fence**

- **Shape:** A smooth circle or sphere.
- **Contact:** The balloon touches the smooth curve.
- **Result:** The touch point is rarely exactly at the "poles" (axes).
- **Implication:** Coefficients ($w$) constitute small numbers, but almost never exactly zero. It shrinks variables but keeps them all.

**2. ℓ₁ (Lasso) = The Diamond Fence**

- **Shape:** A diamond (square rotated 45°) with sharp corners on the axes.
- **Contact:** The pointy corners stick out the furthest. The balloon is highly likely to hit a corner first.
- **Result:** A corner lies exactly on an axis, meaning other variables are **zero**.
- **Implication:** It performs **Feature Selection** by killing off useless variables completely.

```
     ℓ₁ (Lasso)           ℓ₂ (Ridge)
         /\                   ___
        /  \                 /   \
       /    \               |     |
      <      >              |     |
       \    /                \___/
        \  /
         \/
    Corners → Sparsity    No corners → No sparsity
```

**Practical Summary:**

| Result in Math | "Real World" Meaning | Practical Benefit |
| :--- | :--- | :--- |
| **Variable = 0** (Lasso) | **"Delete this factor"** | **Simplicity**: Identifies the few factors that matter; ignores the rest. |
| **Variable ≠ 0** (Ridge) | **"Shrink this factor"** | **Accuracy**: Uses all information, handling correlated features effectively. |

Why other options are wrong:

- (a) FALSE—Both ℓ₁ and ℓ₂ problems are convex
- (c) Lasso balances data fit and penalty, doesn't ignore data
- (d) FALSE—Squared term (ℓ₂) shrinks small coefficients less aggressively, not more

---

### Question 10

**Why is the classical generalization bound P(gap > ε) ≤ 2|H| exp(−2nε²) considered "vacuous" or unhelpful for modern Deep Neural Networks?**

- (a) Neural networks do not use i.i.d. data.
- (b) The number of parameters p is often much larger than the sample size n, making |H| so large that the bound predicts probability > 1 (certain overfitting).
- (c) The bound only applies to linear models, not non-linear neural networks.
- (d) Deep networks have zero Approximation Error, making the bound irrelevant.

**Correct Answer: (b)**

**Explanation:**
The classical bound involves |H|, the size of the hypothesis class. For neural networks:

**The problem:**

- A network with p parameters using 32-bit floats has |H| ≈ 2^(32p) possible weight configurations
- Modern networks have millions/billions of parameters (p >> n)
- The bound 2|H|exp(-2nε²) becomes >> 1, which is meaningless as a probability

**The "Deep Learning Paradox":**

- Classical theory predicts massive overfitting for over-parameterized models
- In practice, deep networks generalize well despite p >> n
- This suggests classical bounds miss something (implicit regularization, optimization geometry, etc.)

Why other options are wrong:

- (a) Neural networks typically do use i.i.d. data assumptions
- (c) The bound applies to any finite hypothesis class, not just linear
- (d) Deep networks have non-zero approximation error

---

### Question 11

**What is meant by "Implicit Regularization" in the context of training over-parameterized Neural Networks?**

- (a) We explicitly add a λ‖W‖² term to the loss function.
- (b) We use a validation set to tune hyperparameters.
- (c) The optimization algorithm itself (e.g., SGD) has a bias toward selecting "simple" (minimum norm) solutions among the many global minima, even without an explicit penalty.
- (d) The network architecture limits the number of neurons to be smaller than the dataset size.

**Correct Answer: (c)**

**Explanation:**
**Implicit Regularization** is a fascinating phenomenon:

Over-parameterized networks have many global minima (infinitely many weight configurations achieve zero training loss). **Which one does SGD find?**

Key insight: **SGD implicitly prefers "simpler" solutions**

- For linear models: SGD finds the **minimum norm** solution
- For neural networks: SGD tends toward solutions with certain structural properties (flat minima, low effective rank)

This happens **without adding explicit regularization terms** to the loss!

**Why it matters:**

- Explains why deep networks generalize despite classical theory
- The optimization trajectory matters, not just the loss landscape
- Early stopping, learning rate, batch size all affect implicit regularization

Why other options are wrong:

- (a) This describes **explicit** regularization (weight decay)
- (b) Hyperparameter tuning is a separate concept
- (d) Architecture constraints are different from implicit regularization

---

### Question 12

**In the regularized objective R̂(f) + λΩ(f), what is the effect of increasing the hyperparameter λ to a very large value?**

- (a) It decreases the Approximation Error (Bias) and increases the Estimation Error (Variance).
- (b) It increases the Approximation Error (Bias) and decreases the Estimation Error (Variance) by forcing the model to be very simple.
- (c) It forces the training error to zero.
- (d) It causes the model to interpolate the data perfectly.

**Correct Answer: (b)**

**Explanation:**
The regularized objective is:

$$\min_f \hat{R}(f) + \lambda \Omega(f)$$

Where:

- R̂(f) = Empirical risk (data fit)
- Ω(f) = Regularization penalty (model complexity)
- λ = Regularization strength

**Effect of increasing λ:**

| λ | Model Behavior | Bias | Variance |
|---|---------------|------|----------|
| λ → 0 | Fit data perfectly | Low | High |
| λ → ∞ | Ignore data, minimize penalty | High | Low |

With very large λ:

- The penalty term dominates
- Model is forced to be "simple" (small weights, sparse, etc.)
- Can't fit complex patterns → **High Bias (underfitting)**
- Stable across different samples → **Low Variance**

This is the **bias-variance tradeoff** controlled by λ.

Why other options are wrong:

- (a) Gets the direction backwards
- (c) Large λ INCREASES training error (prioritizes simplicity over fit)
- (d) Interpolation (zero training error) happens with λ → 0, not λ → ∞

---

### Question 13

**In the Statistical Learning Theory framework, the goal of training a predictive model f is to minimize the empirical risk. What mathematical quantity represents this objective?**

- (a) The sum of the squared weights in the network.
- (b) The average loss over the training set D: (1/n)Σᵢ L(f(xᵢ), yᵢ).
- (c) The maximum likelihood estimation of the test data.
- (d) The gradient of the loss function with respect to the input features.

**Correct Answer: (b)**

**Explanation:**
**Empirical Risk** is the cornerstone of statistical learning:

$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^{n} L(f(x_i), y_i)$$

Components:

- n = number of training samples
- (xᵢ, yᵢ) = individual training examples
- f(xᵢ) = model prediction
- L(·,·) = loss function measuring prediction error
- Average over all training examples

**ERM Principle**: Since we can't compute true risk (unknown P(X,Y)), we minimize empirical risk as a proxy.

Why other options are wrong:

- (a) Sum of squared weights is a regularization term, not empirical risk
- (c) MLE on test data doesn't make sense; we don't have test labels during training
- (d) Gradient is used for optimization, not as the objective itself

---

### Question 14

**Which neural network architecture is mathematically equivalent to classical Linear Regression?**

- (a) A network with one hidden layer and ReLU activation.
- (b) A single fully connected layer with a Sigmoid activation function.
- (c) A single fully connected layer with an identity activation function.
- (d) A multi-layer perceptron with no bias terms.

**Correct Answer: (c)**

**Explanation:**
**Linear Regression:**

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

**Neural Network with identity activation:**

```
Input (d features) → Fully Connected Layer → Identity Activation → Output
        x          →      Wx + b          →        Wx + b       →   ŷ
```

With identity activation σ(z) = z:
$$\hat{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b}) = \mathbf{W}\mathbf{x} + \mathbf{b}$$

This is exactly linear regression!

Why other options are wrong:

- (a) ReLU introduces non-linearity → not linear regression
- (b) Sigmoid squashes output to (0,1) → logistic regression, not linear
- (d) MLP with multiple layers is fundamentally different from linear regression

---

### Question 15

**Why do we typically use iterative methods like Stochastic Gradient Descent (SGD) for Logistic Regression instead of finding a closed-form analytic solution?**

- (a) Because the dataset size n is always too small for analytic solutions.
- (b) Because the non-linearity of the sigmoid function results in a transcendental equation with no algebraic solution.
- (c) Because analytic solutions cannot handle binary output labels.
- (d) Because SGD guarantees finding the global minimum in non-convex landscapes.

**Correct Answer: (b)**

**Explanation:**
**Linear Regression** has a closed-form solution because:

- Loss: MSE = (y - Wx)²
- Setting gradient to zero: ∂L/∂W = 0
- Solution: W = (X^T X)^(-1) X^T y (Normal Equations)

**Logistic Regression** has NO closed-form solution because:

- Model: p = σ(Wx) = 1/(1 + e^(-Wx))
- Loss: Binary Cross-Entropy = -[y log(p) + (1-y) log(1-p)]
- Setting gradient to zero leads to **transcendental equations** involving e^(-Wx)
- These equations cannot be solved algebraically

The sigmoid's non-linearity makes the gradient equation:
$$\sum_i x_i (y_i - \sigma(w^T x_i)) = 0$$

This has no closed-form solution—we must use iterative optimization (SGD, Newton's method, etc.).

Why other options are wrong:

- (a) Dataset size is unrelated to existence of closed-form solutions
- (c) Binary labels aren't the issue; the sigmoid function is
- (d) FALSE—Logistic regression IS convex; SGD finds the global minimum but doesn't "guarantee" it in non-convex problems

---

### Question 16

**When mapping Multi-class Classification (with K classes) to a neural network, what is the standard activation function used in the final output layer?**

- (a) K independent Sigmoid functions.
- (b) The Softmax function.
- (c) The ReLU function.
- (d) The Identity function.

**Correct Answer: (b)**

**Explanation:**
**Softmax** converts K raw scores (logits) into a probability distribution:

$$\text{Softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Properties:

- Output values in (0, 1)
- All outputs sum to 1 (valid probability distribution)
- Larger logits → higher probabilities
- Differentiable for backpropagation

**Why not K independent Sigmoids?**

- Each sigmoid outputs independently in (0,1)
- Outputs don't sum to 1
- Can't be interpreted as mutually exclusive class probabilities

**Example:**

```
Logits: [2.0, 1.0, 0.1]
Softmax: [0.659, 0.242, 0.099]  (sums to 1.0)
Sigmoids: [0.881, 0.731, 0.525]  (sums to 2.137, not valid probabilities)
```

Why other options are wrong:

- (a) Independent sigmoids don't create a proper probability distribution
- (c) ReLU outputs unbounded positive values, not probabilities
- (d) Identity outputs raw scores, not probabilities

---

### Question 17

**Why are single-layer networks (like Linear or Logistic Regression) considered limited in expressiveness?**

- (a) They cannot handle continuous input variables.
- (b) They are computationally too expensive to train.
- (c) They imply a monotonic relationship and cannot solve non-linear problems like XOR.
- (d) They require more data than multi-layer networks to converge.

**Correct Answer: (c)**

**Explanation:**
Single-layer networks compute:
$$f(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

This creates a **linear decision boundary** (hyperplane) in feature space.

**The XOR Problem:**

| x₁ | x₂ | XOR |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

Plotting these points: No single straight line can separate the classes!

```
    x₂
    1 |  ●(0,1)     ○(1,1)
      |
    0 |  ○(0,0)     ●(1,0)
      +----------------x₁
         0          1

● = class 1, ○ = class 0
```

Single-layer networks can only solve **linearly separable** problems. XOR requires a non-linear boundary, which needs **hidden layers** with non-linear activations.

Why other options are wrong:

- (a) FALSE—They handle continuous inputs fine
- (b) FALSE—Single layers are computationally cheaper than deep networks
- (d) FALSE—Single layers typically need less data (fewer parameters)
