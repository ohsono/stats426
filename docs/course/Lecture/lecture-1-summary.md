# Conceptual Foundations of Statistical Learning

**STAT 426 - George Michailidis (UCLA)**

---

## Table of Contents
1. [Motivation: Regression/Classification Framework](#1-motivation-regressionclassification-framework)
2. [Statistical Learning Theory](#2-statistical-learning-theory)
3. [Loss Functions](#3-loss-functions)
4. [Risk Minimization](#4-risk-minimization)
5. [Regularization](#5-regularization)
6. [Optimization](#6-optimization)

---

## 1. Motivation: Regression/Classification Framework

### The General Regression Framework

**Goal:** Predict a continuous outcome `y` based on input vector `x`.

**The Model:**
```
y = f(x) + ε
```

**Components:**
- `f(x)`: Unknown deterministic function linking inputs `x` to outcome `y`
- `ε`: Error with E(ε) = 0, Var(ε) = σ²

**Strategy:** Change how we parameterize `f(x)` to capture different patterns in data.

---

### Historical Evolution of Regression Methods

#### 1. Linear Regression (1900s)
- **Input:** Feature vector x ∈ ℝᵖ
- **Outcome:** Continuous scalar y ∈ ℝ
- **Assumption:** Globally linear and additive relationship

**Parameterization:**
```
f(x) = β₀ + Σⱼ βⱼxⱼ = βᵀx
```

| Pros | Cons |
|------|------|
| Highly interpretable | High bias if true f is non-linear |
| Low variance | |

---

#### 2. Polynomial Regression (1970-80s)
- Extends linear regression by adding powers of features
- **Assumption:** f(x) is smooth and global

**Parameterization (scalar):**
```
f(x) = β₀ + β₁x + β₂x² + ... + βₐxᵈ
```

This is a **Global Basis Expansion**.

---

#### 3. Spline Regression (1970-1990s)
- Fits low-degree polynomials in separate regions defined by **knots**
- **Assumption:** f(x) is piecewise continuous with continuous derivatives
- **Locality:** Changing data in one region only affects the fit locally

**Parameterization (Truncated Power Basis):**
```
f(x) = Σⱼ₌₀³ βⱼxʲ + Σₖ₌₁ᴷ θₖ(x - ξₖ)³₊
```
Where ξₖ are the knots.

---

#### 4. Fourier Regression (1960-70s)
- Approximates f as a sum of sines and cosines
- **Input:** Typically time-series or periodic domain
- **Assumption:** f(x) is periodic or defined on a bounded interval

**Parameterization:**
```
f(x) = a₀ + Σₖ₌₁ᴷ [aₖcos(2πkx/T) + bₖsin(2πkx/T)]
```

✓ Good for global frequency analysis
✗ Bad for local spikes

---

#### 5. Wavelet Regression (1990-2000s)
- Basis functions localized in both **time and frequency**
- **Input:** Signals, Images, non-stationary Time-series
- **Assumption:** Function may contain discontinuities or sharp spikes

**Parameterization:**
```
f(x) = Σₖ cⱼ₀,ₖφⱼ₀,ₖ(x) + Σⱼ₌ⱼ₀ᴶ Σₖ dⱼ,ₖψⱼ,ₖ(x)
```
- φ: Scaling function (Coarse approximation)
- ψ: Mother wavelet (Detail coefficients)

---

#### 6. Kernel Regression (1990-2000s) - The Kernel Trick

**Concept:** Map inputs implicitly to an infinite-dimensional feature space H.

**The "Trick":** Avoid computing φ(x) explicitly. Only need dot products:
```
⟨φ(x), φ(x')⟩ = K(x, x')
```

**Parameterization (Dual Representation):**
```
f(x) = Σᵢ₌₁ᴺ αᵢK(x, xᵢ)
```
- Parameters: αᵢ (Dual coefficients), solved via (K + λI)α = y
- **Advantage:** Captures complex non-linearities without defining specific basis features

---

#### 7. Neural Networks (MLP Regression) (1960s → 2010+)

**Concept:** A "Universal Approximator" composed of layers of linear combinations and non-linear activations.

**Parameterization (Single Hidden Layer):**
```
f(x) = β₀ + Σₘ₌₁ᴹ βₘσ(αₘᵀx + bₘ)
```
- σ(·): Non-linear activation (ReLU, Tanh)
- zₘ = σ(αₘᵀx + bₘ) acts as a **Derived Feature**

**Key Distinction (Adaptive vs. Fixed):**

| Traditional Methods | Neural Networks |
|---------------------|-----------------|
| Basis functions are **fixed** beforehand | Basis functions are **learned** |
| Feature engineering | Parameters α inside activation optimized alongside β |

---

### Transition to Classification

**Change:** y is categorical (discrete)
**Goal:** Model the **Probability** of class membership

**The Link Function:**
```
P(y = k|x) = g(f(x))
```
- f(x) remains one of the regression models
- g(·) is the activation (Sigmoid, Softmax)

#### Binary Classification (Logistic)
- **Outcome:** y ∈ {0, 1}
- **Assumption:** Log-odds is linear

```
p(x) = 1/(1 + e⁻ᶠ⁽ˣ⁾) = σ(βᵀx)
```

**Decision Boundary:** Defined where f(x) = 0

#### Multi-Class Classification
- **Outcome:** y ∈ {1, 2, ..., K}
- **Assumption:** Classes are mutually exclusive

**Parameterization (Softmax):**
```
P(y = k|x) = eᶠᵏ⁽ˣ⁾ / Σⱼ₌₁ᴷ eᶠʲ⁽ˣ⁾
```

---

## 2. Statistical Learning Theory

### The Unifying Framework

All models follow the same fundamental rules:

1. **Data Generation:** Data come from unknown distribution P(X,Y)
2. **Evaluation:** Measure "success" using a **Loss Function**
3. **Objective:** Minimize expected error (**Risk**) using observed data (**Empirical Risk Minimization**)

---

### The Statistical Setup

**Environment:**
- True, unknown joint distribution P(X, Y)
- Inputs: X ∈ 𝒳 (images, vectors)
- Labels: Y ∈ 𝒴 (classes, real numbers)

**Data:**
- Training set D = {(xᵢ, yᵢ)}ⁿᵢ₌₁
- **Assumption:** Data points drawn i.i.d. from P(X, Y)

**Goal:**
- Find function f : 𝒳 → 𝒴 that predicts Y given X
- f(x) should work well on **unseen data** (Generalization)

---

### Hypothesis Classes

> **Definition:** A set of functions we are willing to consider.
> ```
> H = {fθ : θ ∈ Θ}
> ```

**Examples:**
| Model | Hypothesis Class |
|-------|------------------|
| Linear Regression | H_lin = {f(x) = βᵀx \| β ∈ ℝᵖ} |
| Splines | H_spline = {Piecewise polynomials with knots ξ} |
| Neural Networks | H_MLP = {Compositions of linear + non-linear maps} |

**Key Concept:** Selecting the model ≡ selecting the Hypothesis Class H

---

### Inductive Bias

> Without assumptions, learning is impossible (**No Free Lunch Theorem**)

**Inductive Bias:** The set of assumptions we make about the relationship between X and Y by choosing a specific H.

**Examples:**
| Model | Assumption |
|-------|------------|
| Linear | Gradients are constant globally |
| Kernel/RBF | y values are similar if x values are close (smoothness) |
| CNNs | **Translation invariance** (spatial invariance) |

#### Translation Invariance — Explained

**Translation invariance** means the model produces the **same output regardless of where in the input a feature appears**.

```
Cat top-left:     Cat center:       Cat bottom-right:
🐱 . . . . .     . . . . . .       . . . . . .
. . . . . .       . . 🐱 . . .     . . . . . .
. . . . . .       . . . . . .       . . . . . 🐱

All three → same output: "cat"
```

**Why MLPs fail at this:**
MLPs assign fixed weights to each pixel position. The same cat shifted by one pixel activates completely different weights — the model must relearn the cat for every possible position.

**How CNNs achieve it:**

| Mechanism | How it helps |
|---|---|
| **Shared weights (convolution)** | The same filter scans the whole image — detects the same feature anywhere |
| **Pooling (max/avg)** | Summarizes a region → small shifts don't change the output |

**Translation Invariance vs. Equivariance:**

| | Invariance | Equivariance |
|---|---|---|
| **Meaning** | Output unchanged when input shifts | Output shifts the same way as input |
| **Example** | "Is there a cat?" → always yes | "Where is the cat?" → location shifts too |
| **Used in** | Classification (CNNs + pooling) | Detection, segmentation |

> **Bottom line:** Translation invariance = "the same pattern anywhere in the input gives the same answer." It's why CNNs are powerful for images — a cat is a cat no matter where it appears.

---

## 3. Loss Functions

### Definition

> A function L : 𝒴 × 𝒴 → ℝ₊ that measures the cost of predicting ŷ = f(x) when the true label is y.

**Properties:**
- L(y, y) = 0 (No cost for perfect prediction)
- Penalizes deviations from truth
- Choice depends on task

---

### Regression Losses

#### 1. Squared Error (ℓ₂ Loss)
```
L(y, f(x)) = (y - f(x))²
```
- ✓ Differentiable everywhere, mathematically convenient
- ✓ Leads to **mean** estimation
- ✗ Sensitive to outliers

#### 2. Absolute Error (ℓ₁ Loss)
```
L(y, f(x)) = |y - f(x)|
```
- ✓ Robust to outliers
- ✓ Leads to **median** estimation
- ✗ Not differentiable at 0

#### 3. Huber Loss (Hybrid)
```
Lδ(a) = {
  ½a²           for |a| ≤ δ
  δ(|a| - ½δ)   otherwise
}
```
Behaves like ℓ₂ near zero (differentiable) and ℓ₁ far away (robust).

---

### Classification Losses

#### The 0/1 Loss (Gold Standard)
```
L₀/₁(y, f(x)) = 𝕀(y ≠ sign(f(x)))
```
**Problem:** Non-convex, non-differentiable → NP-hard optimization

**Solution:** Use **Convex Surrogates**

#### Surrogate Loss Functions

| Loss | Formula | Used In |
|------|---------|---------|
| **Hinge** | max(0, 1 - yf(x)) | SVM |
| **Logistic** | log(1 + exp(-yf(x))) | Logistic Regression |
| **Exponential** | exp(-yf(x)) | Boosting |

**Key Takeaway:** Optimize surrogate loss to approximate the ideal 0/1 loss.

---

## 4. Risk Minimization

### True Risk (The Ideal Objective)

> **Definition:** Expected loss over the data distribution
> ```
> R(f) = 𝔼_X,Y[L(Y, f(X))] = ∫ L(y, f(x))dP(x, y)
> ```

**Intuition:** The true risk is the **actual average loss** your model makes over the **entire data distribution** — including all possible future data you'll never see. It answers: *"How wrong is my model in the real world?"*

**The Fundamental Problem:**
- We do **not know** P(X, Y)
- Therefore, we **cannot compute** R(f) directly
- It is the "ground truth" of how good your model really is

---

### Empirical Risk (The Proxy)

> **Definition:**
> ```
> R̂(f) = (1/n) Σᵢ₌₁ⁿ L(yᵢ, f(xᵢ))
> ```

**Intuition:** Empirical risk is your best **approximation** of the true risk using only the finite training set you actually have. It is what you directly minimize during training (your training loss).

| | True Risk R(f) | Empirical Risk R̂(f) |
|---|---|---|
| **Based on** | Full data distribution (unknown) | Training set only (observed) |
| **Computable?** | ❌ No | ✅ Yes |
| **What it measures** | Real-world generalization | Training performance |

**Key tension in ML:**
- Empirical Risk ↓ → model fits training data well
- True Risk ↓ → model generalizes to new data
- **Overfitting** = empirical risk is very low but true risk is high

**Assumptions:**
- Data are i.i.d.
- Loss is bounded: L(·,·) ∈ [0, 1]

By **Law of Large Numbers**: R̂(f) → R(f) as n → ∞ for a fixed f.

---

### Empirical Risk Minimization (ERM)

> **The ERM Principle:**
> ```
> f̂ = argmin_{f∈H} R̂(f) = argmin_{f∈H} (1/n) Σᵢ₌₁ⁿ L(yᵢ, f(xᵢ))
> ```

**Key Subtlety:** f̂ depends on the same data used to compute it.

This data-dependence is the source of **overfitting risk**.

---

### The Bayes Optimal Predictor (f*)

> **Definition:** The function that achieves minimal possible risk over all functions.
> ```
> f*(x) = argmin_{all f} R(f)
> ```

**What f* looks like:**

| Task | Optimal Predictor |
|------|-------------------|
| Regression (ℓ₂) | Conditional Mean: f*(x) = 𝔼[Y \| X = x] |
| Classification (0/1) | Bayes Classifier: f*(x) = argmax_k P(Y = k \| X = x) |

**Irreducible Error (Bayes Risk):** Even f* is not perfect. R(f*) > 0 due to noise.

---

### Decomposing Excess Risk

```
R(f̂) - R(f*) = [R(f*_H) - R(f*)] + [R(f̂) - R(f*_H)]
                 └─────────────┘   └──────────────┘
                 Approximation      Estimation
                 Error (Bias)       Error (Variance)
```

| Error Type | Description | Cause |
|------------|-------------|-------|
| **Approximation Error** | Penalty for restricting search to H | Model family doesn't contain truth |
| **Estimation Error** | Penalty for finite training data | Selected f̂ based on noisy data |

---

### The Bias-Variance Trade-off

#### In Plain English

**Bias — "Consistently Wrong"**
Bias is how far off your model is **on average**, even if you retrained it on many different datasets.

> 🎯 Like a broken clock that always shows 3:00 — it's **consistently wrong** regardless of when you check.

- High bias = model makes the **same type of mistake** no matter what data it sees
- Caused by the model being **too simple** to capture the real pattern (underfitting)

**Variance — "Inconsistently Right"**
Variance is how much your model's predictions **change** depending on which training data it happened to see.

> 🎯 Like a jittery person who gives a different answer every time you ask the same question — high variance means you can't rely on the model.

- High variance = model is **very sensitive** to the exact training data it saw
- Caused by the model being **too complex**, memorizing noise instead of patterns (overfitting)

**The Shooting Target**

```
                 Hit the bullseye?
                   YES          NO
                ┌──────────┬──────────┐
Shots    YES    │ Low Bias │ High     │
clustered?      │ Low Var  │ Bias     │
                │ ✅ BEST  │ Low Var  │
                ├──────────┼──────────┤
          NO    │ Low Bias │ High     │
                │ High Var │ Bias     │
                │ Scattered│ High Var │
                │          │ ❌ WORST │
                └──────────┴──────────┘
```

#### Mathematical Decomposition

For any prediction point x, the expected MSE decomposes as:

```
Total Error = Bias²[f̂(x)]  +  Var[f̂(x)]  +  σ²
               (systematic)    (sensitivity)   (noise)
```

#### The Three Terms Explained

| Term | Meaning | Cause |
|---|---|---|
| **Bias²** | How wrong *on average* | Model too simple — wrong assumptions |
| **Variance** | How sensitive to training set | Model too complex — memorizes noise |
| **σ² (irreducible)** | Noise inherent in data | Cannot be reduced regardless of model |

- **High Bias → Underfitting:** A linear model on curved data is always wrong regardless of the data
- **High Variance → Overfitting:** A degree-15 polynomial fits training data perfectly but goes wild on new data


#### Visual of the Tradeoff

```
Model Complexity →→→→→→→→→→→→→→

Bias     ████████░░░░░░░░░░░░  (decreases)
Variance ░░░░░░░░████████████  (increases)
Total    ████░░░░░░░░░░░░████  (U-shaped)
                    ↑
               Sweet spot  ← selected by cross-validation
```

| Regime | H Size | Bias | Variance | Risk |
|--------|--------|------|----------|------|
| Underfitting | Small | High | Low | High |
| Optimal | Medium | Balanced | Balanced | **Lowest** |
| Overfitting | Large | Low | High | High |

#### Effect of Regularization on Bias-Variance

| Regularization | Effect on Bias | Effect on Variance |
|---|---|---|
| Increase λ (more penalty) | ⬆️ Bias increases | ⬇️ Variance decreases |
| Decrease λ (less penalty) | ⬇️ Bias decreases | ⬆️ Variance increases |

The optimal λ found by cross-validation is the sweet spot that minimizes **total error** (Bias² + Variance).

#### Archer Analogy

Think of 10 archers shooting at a target across different training datasets:

| | Shots clustered? | Centred on target? | Result |
|---|---|---|---|
| Low Bias, Low Variance | ✅ Yes | ✅ Yes | **Ideal** |
| High Bias, Low Variance | ✅ Yes | ❌ No | Consistently off |
| Low Bias, High Variance | ❌ No | ✅ Yes (on avg) | Scattered |
| High Bias, High Variance | ❌ No | ❌ No | Worst case |

---

### The Generalization Gap

> **Definition:**
> ```
> GenGap = R(f̂) - R̂(f̂)
> ```

The generalization gap is the difference between how well your model performs on **new, unseen data** (true risk) vs. on **training data** (empirical risk).

#### Why Does It Exist?

- **Optimization Bias:** $\hat{f}$ was chosen *specifically* to minimize $\hat{R}$ on training data, so $\hat{R}(\hat{f})$ is an **optimistically biased** estimate of true performance.
- **Overfitting:** Model adapted to random noise in training data rather than the true underlying pattern.

#### Visual Intuition

```
Training data performance  →  R̂(f̂)  →  too optimistic
New data performance       →  R(f̂)   →  the real score
                                ↑
                          GenGap lives here
```

> **Student analogy:** A student who memorizes past exam answers scores perfectly on those but fails the actual exam. That gap between their practice score and real exam score is the generalization gap.

#### How to Reduce It

| Strategy | Mechanism |
|---|---|
| **More data** ($n$ ↑) | Bounds tighten; empirical risk better approximates true risk |
| **Regularization** (L1/L2) | Restricts effective model complexity |
| **Cross-validation** | Uses held-out data to estimate true risk more honestly |
| **Early stopping** | Stops training before model overfits |
| **Dropout / Data augmentation** | Reduces variance in deep learning |

---

### Generalization Bounds

#### Hoeffding's Inequality (for single fixed f)
```
P(|R̂(f) - R(f)| > ε) ≤ 2exp(-2nε²)
```

#### Uniform Convergence Bound (finite H)
```
P(Δₙ(H) > ε) ≤ 2|H|exp(-2nε²)
```

**The Fundamental Trade-off:**
> To maintain low Estimation Error, if you increase complexity (|H| ↑), you **must** increase data (n ↑).

---

## 5. Regularization

### Structural Risk Minimization

> **Regularized Objective:**
> ```
> f̂ = argmin_{f∈H} R̂(f) + λΩ(f)
> ```

| Component | Purpose |
|-----------|---------|
| R̂(f) | Fit the data (minimize bias) |
| Ω(f) | Keep model simple (minimize variance) |
| λ | Hyperparameter controlling trade-off |

---

### Parametric Regularization

#### Ridge Regression (ℓ₂ Penalty)
```
Ω(f) = ‖β‖₂² = Σ βⱼ²
```
- Shrinks coefficients toward zero
- Constraint region: **circular/spherical**

#### Lasso Regression (ℓ₁ Penalty)
```
Ω(f) = ‖β‖₁ = Σ |βⱼ|
```
- Induces **Sparsity** (feature selection)
- Constraint region: **diamond-shaped**

**Geometric Interpretation:**
- ℓ₁: Diamond corners lie on axes → coefficients hit exactly zero
- ℓ₂: Circular → coefficients shrink but rarely hit zero exactly

---

### Function Space Regularization (Splines/Kernels)

**Sobolev Norm Regularization:**
```
Σᵢ(yᵢ - f(xᵢ))² + λ∫[f''(x)]²dx
```
- ∫[f''(x)]²dx: Measures "wiggliness" or curvature
- High λ: Forces f to be linear
- Low λ: Allows wiggly interpolation

---

### The Deep Learning Paradox

**Classical Bound:**
```
P(Excess Risk > ε) ≤ 2|H|exp(-2nε²)
```

**The Conflict:**
- Modern networks: millions of parameters (p ≫ n)
- Bound becomes **vacuous** (→ ∞)
- Classical theory predicts complete failure

**The Reality:** Deep Networks generalize surprisingly well!

---

### Resolving the Paradox

#### 1. It's about the Norm, not the Count
- Complexity depends on **magnitude of weights** (‖W‖_F), not just number
- Network with 1M small weights ≈ simpler linear model

#### 2. Implicit Regularization (Double Descent)
- SGD prefers "simple" (minimum norm) solutions
- Among infinite zero-error solutions, optimization finds simplest

---

### Regularization Techniques for Deep NN

| Technique | Mechanism | Theory |
|-----------|-----------|--------|
| **Weight Decay** (ℓ₂) | Minimize ‖W‖² | Restricts to small ball in parameter space |
| **Dropout** | Randomly drop neurons | Ensemble averaging, reduces variance |
| **Early Stopping** | Stop when validation rises | Prevents reaching "wiggly" overfitting regions |

---

## 6. Optimization

### Solving the ERM Problem

**Analytical Solution** (when possible):
```
β̂ = (XᵀX + λI)⁻¹Xᵀy
```
Works for: OLS, Ridge, Kernel Ridge

**Iterative Optimization** (required for most):
- Lasso, Logistic, Neural Networks

---

### Gradient Descent (GD) — "Full Batch"

At every step, compute the gradient using **all $n$ training samples**:

```
w ← w - η · (1/n) Σᵢ₌₁ⁿ ∇_w ℓ(yᵢ, f(xᵢ))
```

```
Algorithm: Gradient Descent
1: Initialize parameters θ (e.g., random)
2: while not converged do
3:     Compute gradient: g = ∇_θ R̂(f_θ)   ← uses ALL n samples
4:     Update: θ ← θ - η·g
5: end while
```

- η: Learning rate
- **Accurate** — true gradient direction
- **Slow** — must scan entire dataset before taking one step
- **Smooth convergence** — loss decreases steadily

---

### Stochastic Gradient Descent (SGD) — "One Sample"

Approximate gradient using a **single randomly picked sample**:

```
w ← w - η · ∇_w ℓ(yᵢ, f_θ(xᵢ))    ← uses 1 sample
```

- **Fast** — one update per sample, many updates per epoch
- **Noisy** — single-sample gradient is a rough estimate of the true gradient
- **Noisy convergence** — loss bounces around but trends downward

---

### Mini-Batch SGD — The Practical Standard

Use a small batch of $B$ samples (typically 32–256) per update:

```
w ← w - η · (1/B) Σᵢ∈batch ∇_w ℓ(yᵢ, f(xᵢ))
```

This is what's **actually used** in deep learning — balances the accuracy of GD with the speed of SGD.

---

### GD vs. SGD vs. Mini-Batch Comparison

| Property | GD (Full Batch) | SGD (1 Sample) | Mini-Batch SGD |
|---|---|---|---|
| **Gradient quality** | Exact | Very noisy | Approximate |
| **Updates per epoch** | 1 | $n$ | $n/B$ |
| **Memory use** | All data | 1 sample | $B$ samples |
| **Convergence** | Smooth, stable | Noisy, bouncy | In between |
| **Escapes local minima?** | ❌ Harder | ✅ Noise helps | ✅ Somewhat |
| **GPU parallelism** | ✅ Good | ❌ Poor | ✅ Best |
| **Used in practice** | Rarely (small data) | Rarely | ✅ Standard |

---

### Why SGD Noise Can Be a Feature, Not a Bug

The randomness in SGD acts as **implicit regularization**:
- Prevents the optimizer from settling in **sharp, narrow minima** (which tend to overfit)
- Tends to find **flat minima** — which generalize better to new data
- Helps explain why deep learning with SGD generalizes better than classical theory predicts

**Fix for noisy convergence:** Adaptive momentum methods like **Adam** (Adaptive Moment Estimation) smooth out noisy gradient updates automatically.

| Pros of SGD | Cons of SGD |
|---|---|
| Faster updates | Noisy convergence |
| Better GPU scaling | Requires tuning learning rate |
| Noise helps escape local minima | May overshoot minimum |
| Implicit regularization effect | |

---

### Convexity

| Convex Problems | Non-Convex Problems |
|-----------------|---------------------|
| Linear/Ridge/Lasso/SVM | Neural Networks |
| One global minimum | Many local minima |
| Convergence guaranteed | Initialization matters |

---

## Summary: The Big Picture

1. **The Goal:** Minimize True Risk R(f)

2. **The Reality:** We only have Data D, so minimize Empirical Risk R̂(f)

3. **The Tools (Hypothesis Classes):**
   - Linear/Spline/Kernel (Interpretable, specific assumptions)
   - Neural Networks (Universal approximators, data hungry)

4. **The Safety Valve (Regularization):** Prevents Overfitting by penalizing complexity (ℓ₁, ℓ₂)

5. **The Engine (Optimization):** Gradient Descent / SGD

---

## Key Formulas Reference

| Concept | Formula |
|---------|---------|
| True Risk | R(f) = 𝔼[L(Y, f(X))] |
| Empirical Risk | R̂(f) = (1/n)Σ L(yᵢ, f(xᵢ)) |
| Regularized Objective | R̂(f) + λΩ(f) |
| Excess Risk | R(f̂) - R(f*) = Approx. Error + Est. Error |
| Generalization Gap | R(f̂) - R̂(f̂) |
| Hoeffding Bound | P(\|R̂-R\| > ε) ≤ 2exp(-2nε²) |
| Uniform Bound | P(gap > ε) ≤ 2\|H\|exp(-2nε²) |
