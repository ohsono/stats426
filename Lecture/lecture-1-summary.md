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
| CNNs | Spatial invariance |

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

**The Fundamental Problem:**
- We do **not know** P(X, Y)
- Therefore, we **cannot compute** R(f) directly

---

### Empirical Risk (The Proxy)

> **Definition:**
> ```
> R̂(f) = (1/n) Σᵢ₌₁ⁿ L(yᵢ, f(xᵢ))
> ```

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

```
           High
             │
     Error   │    ╭──── True Risk R(f)
             │   ╱
             │  ╱    Overfitting
             │ ╱      Zone
             │╱
     Low     │───────── Training Error R̂(f)
             └────────────────────────────────
             Low    Optimal    High
                  Complexity
```

| Regime | H Size | Bias | Variance | Risk |
|--------|--------|------|----------|------|
| Underfitting | Small | High | Low | High |
| Optimal | Medium | Balanced | Balanced | Lowest |
| Overfitting | Large | Low | High | High |

---

### The Generalization Gap

> **Definition:**
> ```
> GenGap = R(f̂) - R̂(f̂)
> ```

**Why does it exist?**
- **Optimization Bias:** f̂ was chosen specifically to minimize R̂
- **Overfitting:** Model adapted to random noise in training data

**Result:** R̂(f̂) is an **optimistically biased** estimate of R(f̂).

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

### Gradient Descent (GD)

```
Algorithm: Gradient Descent
1: Initialize parameters θ (e.g., random)
2: while not converged do
3:     Compute gradient: g = ∇_θ R̂(f_θ)
4:     Update: θ ← θ - η·g
5: end while
```

- η: Learning rate
- **Issue:** Computing ∇ over all N points is slow

---

### Stochastic Gradient Descent (SGD)

Approximate gradient using single point (or mini-batch):
```
θ ← θ - η∇_θL(yᵢ, f_θ(xᵢ))
```

| Pros | Cons |
|------|------|
| Faster updates | Noisy convergence |
| Better scaling | |
| Noise helps escape local minima | |

**Fix:** Adaptive momentum methods (ADAM, etc.)

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
