# L1 Regularization (Lasso)

## Overview

**L1 regularization**, also known as **Lasso** (Least Absolute Shrinkage and Selection Operator), is a technique used to prevent overfitting in machine learning models by adding a penalty term to the loss function based on the **absolute values** of model coefficients.

---

## The Objective Function

For a linear model, the L1-regularized loss function is:

$$\mathcal{L}(\mathbf{w}) = \underbrace{\text{Loss}(\mathbf{w})}_{\text{data fit}} + \underbrace{\lambda \sum_{j=1}^{p} |w_j|}_{\text{L1 penalty}}$$

Where:
- $\text{Loss}(\mathbf{w})$ — original loss (e.g., MSE for regression, cross-entropy for classification)
- $w_j$ — model coefficients/weights
- $\lambda \geq 0$ — regularization strength (hyperparameter)
- $p$ — number of features

---

## Key Properties

### 1. Sparsity (Feature Selection)
The most distinctive property of L1 regularization is that it **drives many coefficients exactly to zero**, producing a **sparse model**. This effectively performs automatic **feature selection** — only the most informative features retain non-zero weights.

**Example — sparse L1 output:**
```
Coefficients: [2.3, 0.0, 0.0, -1.1, 0.0, 0.0, 0.8, 0.0]
               keep  ✗    ✗   keep   ✗    ✗   keep  ✗
```
Only 3 out of 8 features are kept; the rest are eliminated entirely.

> **Why zero?** The L1 penalty has a non-differentiable "kink" at $w = 0$. The subgradient at zero allows the optimizer to set weights exactly to zero, unlike L2 which only shrinks weights toward (but never to) zero.

### 2. Coefficient Shrinkage
All non-zero coefficients are shrunk toward zero, reducing model complexity and variance.

### 3. Robustness to Irrelevant Features
Because irrelevant features get zeroed out, L1 is especially useful in **high-dimensional settings** where $p \gg n$.

---

## L1 vs. L2 Regularization

| Property | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Penalty term | $\lambda \sum \|w_j\|$ | $\lambda \sum w_j^2$ |
| Solution sparsity | ✅ Sparse (exact zeros) | ❌ Dense (shrinks, never zeros) |
| Feature selection | ✅ Built-in | ❌ No |
| Geometry | Diamond-shaped constraint | Circular constraint |
| Differentiability | ❌ Not at $w=0$ | ✅ Everywhere |
| Best for | High-dim, sparse signals | Correlated features, all features matter |
| Handles multicollinearity | Picks one, zeros others | Shrinks all equally |

### Geometric Intuition

The L1 constraint region is a **diamond (rhombus)** in 2D. The loss function contours tend to hit the **corners** of the diamond, which correspond to axes (i.e., one or more weights = 0). This is why L1 naturally produces sparse solutions.

```
         w2
          |
          ◆  ← L1 diamond
         /|\
        / | \
──────/──+──\──── w1
      \   |  /
       \  | /
        \ |/
          ◆
```

---

## Sparsity vs. Dense — In Depth

### What "Sparse" means (L1)

**Sparse** = most coefficients are **exactly zero**. Only a handful of features survive with non-zero weights. L1 acts as both a regularizer *and* a feature selector.

```
L1 weights: [2.3,  0.0,  0.0, -1.1,  0.0,  0.0,  0.8,  0.0]
                   ^^^   ^^^          ^^^   ^^^          ^^^  ← zeroed out
```

### What "Dense" means (L2)

**Dense** = **all** coefficients remain non-zero — they are only *shrunk* smaller. No feature is ever fully eliminated.

```
L2 weights: [1.1, 0.03, 0.07, -0.9, 0.01, 0.02,  0.4, 0.05]
                  ^^^^  ^^^^        ^^^^  ^^^^         ^^^^  ← tiny but non-zero
```

### Why does L1 zero out but L2 doesn't?

The answer is geometric:

| | L1 | L2 |
|---|---|---|
| **Constraint shape** | Diamond (sharp corners) | Circle (smooth) |
| **Where loss hits constraint** | At a corner → on an axis → $w = 0$ | On the curve → rarely on an axis |

The **sharp corners of the L1 diamond lie exactly on the axes** (where one or more weights = 0). Loss contours are very likely to first touch a corner, producing exact zeros. The **L2 circle has no corners**, so the optimal point almost never lands on an axis.

### When to use each

| Situation | Choice |
|---|---|
| Many features, only a few are expected to matter | **L1** (sparse) |
| All features are meaningful / features are correlated | **L2** (dense) |
| Want sparsity but also handle correlated features | **Elastic Net** (L1 + L2) |

---

## Effect of the Hyperparameter $\lambda$

| $\lambda$ value | Effect |
|---|---|
| $\lambda = 0$ | No regularization — standard loss minimization |
| Small $\lambda$ | Mild shrinkage, most features retained |
| Large $\lambda$ | Heavy shrinkage, most features zeroed out |
| $\lambda \to \infty$ | All coefficients → 0 (null model) |

Use **cross-validation** to select the optimal $\lambda$.

---

## Optimization

Because L1 is **non-differentiable at zero**, standard gradient descent cannot be applied directly. Common solvers include:

- **Coordinate Descent** — updates one weight at a time with a soft-thresholding step (most common for Lasso)
- **Subgradient Methods** — generalization of gradient to non-differentiable functions
- **LARS (Least Angle Regression)** — efficient path algorithm for the full regularization path
- **Proximal Gradient Descent** — decomposes the problem into smooth + non-smooth parts

### Why L1 Has No Closed-Form Solution

Unlike Ridge (L2), Lasso **cannot** be solved with a single formula. The reason is the **sharp kink** in the $|w|$ penalty at $w = 0$:

- A closed-form solution requires setting the gradient to zero and solving algebraically.
- The $|w|$ penalty is **not differentiable at $w = 0$** — its gradient doesn't exist there.
- Therefore, no algebraic shortcut exists; the optimizer must **iterate step by step** toward the answer.

> Compare: L2's $w^2$ penalty is smooth everywhere → gradient always exists → can be set to zero and solved in one step.


### OLS (Ordinary Least Squares)

**OLS** stands for **Ordinary Least Squares** — the standard unconstrained method for fitting a linear model by minimizing the sum of squared residuals:

$$\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^\top \mathbf{x}_i)^2$$

In coordinate descent, $\tilde{w}_j$ denotes the **OLS estimate for coefficient $j$** with all other coefficients held fixed — i.e., the ideal unconstrained weight for that feature before the L1 penalty is applied.

### Soft-Thresholding (Coordinate Descent Update)

The closed-form update for each coefficient in coordinate descent is:

$$w_j \leftarrow \text{sign}(\tilde{w}_j) \cdot \max(|\tilde{w}_j| - \lambda, 0)$$

Where $\tilde{w}_j$ is the OLS estimate for $w_j$ with all other variables fixed. This operation "soft-thresholds" small coefficients to exactly zero:
- If $|\tilde{w}_j| \leq \lambda$ → coefficient is set to **exactly 0**
- If $|\tilde{w}_j| > \lambda$ → coefficient is shrunk by $\lambda$ toward zero

---

## Practical Considerations

1. **Standardize features** before applying L1 — the penalty treats all coefficients equally, so scale matters.
2. **Tune $\lambda$ via cross-validation** (e.g., `LassoCV` in scikit-learn).
3. L1 is **not ideal for correlated features** — it tends to arbitrarily pick one from a group of correlated predictors.
4. For correlated features, consider **Elastic Net** (combines L1 + L2).

---

## Elastic Net: Best of Both Worlds

$$\mathcal{L}(\mathbf{w}) = \underbrace{\text{Loss}(\mathbf{w})}_{\text{data fit}} + \underbrace{\lambda_1 \sum |w_j|}_{\text{L1 term}} + \underbrace{\lambda_2 \sum w_j^2}_{\text{L2 term}}$$

Elastic Net combines L1 and L2 penalties into a single objective. Each term plays a distinct role:

| Term | Type | Effect |
|---|---|---|
| $\text{Loss}(\mathbf{w})$ | Data fit | Minimizes prediction error (e.g., MSE) |
| $\lambda_1 \sum \|w_j\|$ | L1 penalty | Drives coefficients to **exactly zero** → sparsity / feature selection |
| $\lambda_2 \sum w_j^2$ | L2 penalty | Shrinks all coefficients smoothly → stability with correlated features |

### Why combine them?

Each method alone has a weakness:

| Problem | L1 alone | L2 alone | Elastic Net |
|---|---|---|---|
| Correlated features | Picks one arbitrarily, zeros others | Shrinks all equally | ✅ Groups them, shrinks together |
| Too many features | ✅ Zeros out irrelevant ones | ❌ Keeps all | ✅ Still zeros out irrelevant ones |
| Multicollinearity | ❌ Unstable selection | ✅ Stable | ✅ Stable + sparse |

### Special Cases

| Hyperparameters | Reduces to |
|---|---|
| $\lambda_1 = 0,\ \lambda_2 > 0$ | Pure **Ridge** (L2 only) |
| $\lambda_1 > 0,\ \lambda_2 = 0$ | Pure **Lasso** (L1 only) |
| $\lambda_1 > 0,\ \lambda_2 > 0$ | **Elastic Net** (both) |

Elastic Net inherits:
- **Sparsity** from L1 — irrelevant features are zeroed out
- **Grouping effect** from L2 — correlated features shrink together rather than one being arbitrarily eliminated

---

## Python Example (scikit-learn)

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Best practice: scale features first
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))  # alpha = lambda
])
pipe.fit(X_train, y_train)

# Sparse coefficients — many will be exactly 0
print(pipe.named_steps['lasso'].coef_)

# Cross-validate to find optimal alpha
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")
```

---

## Summary

| Aspect | Detail |
|---|---|
| **Penalty** | $\lambda \sum \|w_j\|$ (L1 norm) |
| **Key property** | Sparsity — exact zeros for uninformative features |
| **Use case** | High-dimensional data, automatic feature selection |
| **Hyperparameter** | $\lambda$ (larger = more regularization) |
| **Solver** | Coordinate descent with soft-thresholding |
| **Limitation** | Struggle with multicollinearity; use Elastic Net if needed |

> **Bottom line:** L1 regularization is both a regularizer (reduces overfitting) and a feature selector (zeros out irrelevant weights). It is the go-to choice when you believe only a subset of features truly matter.
