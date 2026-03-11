# L2 Regularization (Ridge)

## Overview

**L2 regularization**, also known as **Ridge regression**, is a technique to prevent overfitting by adding a penalty term to the loss function based on the **squared values** of model coefficients. Unlike L1, it does **not** zero out coefficients — instead, it smoothly shrinks all of them toward zero while keeping every feature in the model.

---

## The Objective Function

$$\mathcal{L}(\mathbf{w}) = \underbrace{\text{Loss}(\mathbf{w})}_{\text{data fit}} + \underbrace{\lambda \sum_{j=1}^{p} w_j^2}_{\text{L2 penalty}}$$

Where:
- $\text{Loss}(\mathbf{w})$ — original loss (e.g., MSE for regression, cross-entropy for classification)
- $w_j$ — model coefficients/weights
- $\lambda \geq 0$ — regularization strength (hyperparameter)
- $p$ — number of features

For linear regression with MSE loss, this becomes:

$$\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n}(y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_2^2$$

---

## Key Properties

### 1. Dense Solutions (No Exact Zeros)
L2 regularization **shrinks all coefficients toward zero** but never sets any exactly to zero. Every feature always contributes to the model (albeit with a smaller weight).

**Example — dense L2 output:**
```
Coefficients: [1.1, 0.03, 0.07, -0.9, 0.01, 0.02, 0.4, 0.05]
               keep  ↓     ↓    keep   ↓     ↓    keep  ↓
               (all kept, small ones shrunk close to zero)
```

### 2. Handles Multicollinearity Well
When features are correlated, L2 **distributes the weight evenly among them**, rather than arbitrarily picking one and zeroing the others (as L1 does). This makes Ridge more stable and interpretable in the presence of correlated predictors.

### 3. Unique Closed-Form Solution
Unlike Lasso (L1), Ridge regression has a **closed-form analytical solution** — meaning you can compute the exact answer **directly with one formula**, with no iteration or approximation needed.

$$\hat{\mathbf{w}}_{\text{Ridge}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

Adding $\lambda \mathbf{I}$ to $\mathbf{X}^\top \mathbf{X}$ ensures the matrix is always invertible, even when features are highly correlated or when $p > n$ (more features than samples).

#### Closed-Form vs. Iterative Solvers

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty shape | $w^2$ — smooth, differentiable everywhere | $\|w\|$ — **not differentiable at $w=0$** |
| Can you set gradient to 0? | ✅ Yes → solve algebraically | ❌ No (kink at zero) |
| Solution method | **Closed-form** matrix equation | **Iterative** coordinate descent |

**Why does L2 have one but L1 doesn't?**
- The $w^2$ penalty is smooth — its gradient always exists, so you can set $\nabla \mathcal{L} = 0$ and solve algebraically in one step.
- The $|w|$ penalty has a **sharp kink at $w=0$** — calculus can't handle it cleanly, so you must use an iterative algorithm that updates weights step by step until convergence.

> **Simple analogy:**
> - Closed-form = solving $2x + 3 = 7$ → $x = 2$ (one step, exact answer)
> - Iterative = using trial and error to approach the answer step by step until "close enough"

### 4. Bias-Variance Tradeoff
L2 introduces **bias** (predictions are slightly pulled toward zero) in exchange for **reduced variance** (less sensitivity to noise in training data). This tradeoff reduces overfitting.

---

## Geometric Intuition

The L2 constraint region is a **sphere (circle in 2D)**. The loss function contours (typically ellipses) touch the sphere at a point that is almost never exactly on a coordinate axis. This is why L2 rarely produces exact zeros.

```
         w2
          |
     _____|_____
    /     |     \
   /      |      \   ← L2 circle (sphere)
  |       |       |
──|───────+───────|──── w1
  |       |       |
   \      |      /
    \_____|_____/
          |
```

The loss contour "slides" along the circle and settles at a point where **both** $w_1$ and $w_2$ are non-zero.

Compare to L1's diamond, whose corners sit exactly on the axes — that's why L1 produces zeros and L2 does not.

---

## L2 vs. L1 Regularization

| Property | L2 (Ridge) | L1 (Lasso) |
|---|---|---|
| Penalty term | $\lambda \sum w_j^2$ | $\lambda \sum \|w_j\|$ |
| Solution sparsity | ❌ Dense (shrinks, never zeros) | ✅ Sparse (exact zeros) |
| Feature selection | ❌ No | ✅ Built-in |
| Geometry | Circular constraint | Diamond-shaped constraint |
| Differentiability | ✅ Everywhere | ❌ Not at $w=0$ |
| Closed-form solution | ✅ Yes | ❌ No (requires iterative solver) |
| Best for | Correlated features, all features matter | High-dim, sparse signals |
| Handles multicollinearity | ✅ Shrinks correlated features together | ❌ Picks one, zeros others |

---

## Effect of the Hyperparameter $\lambda$

| $\lambda$ value | Effect |
|---|---|
| $\lambda = 0$ | No regularization — standard OLS solution |
| Small $\lambda$ | Mild shrinkage, coefficients close to OLS estimates |
| Large $\lambda$ | Heavy shrinkage, all coefficients → 0 |
| $\lambda \to \infty$ | All coefficients → 0 (null model) |

> As $\lambda$ increases, the **bias increases** but **variance decreases**. Use **cross-validation** to find the optimal balance.

---

## Optimization

Because the L2 penalty is **smooth and differentiable everywhere**, standard gradient-based optimizers work directly.

### Gradient of the L2-Regularized Loss

$$\nabla_\mathbf{w} \mathcal{L} = \nabla_\mathbf{w} \text{Loss} + 2\lambda \mathbf{w}$$

The gradient descent update becomes:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \left(\nabla_\mathbf{w} \text{Loss} + 2\lambda \mathbf{w}\right)$$

Which can be rewritten as:

$$\mathbf{w} \leftarrow (1 - 2\eta\lambda)\mathbf{w} - \eta \nabla_\mathbf{w} \text{Loss}$$

The factor $(1 - 2\eta\lambda)$ **decays the weight at every step** — this is why L2 regularization is also called **weight decay** in deep learning.

### OLS Connection

The standard OLS solution $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ breaks down when $\mathbf{X}^\top\mathbf{X}$ is singular (non-invertible), which happens with multicollinearity or when $p > n$.

Ridge adds $\lambda\mathbf{I}$ to regularize the matrix:

$$\hat{\mathbf{w}}_\text{Ridge} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

This is always invertible for any $\lambda > 0$, making Ridge numerically stable.

---

## Weight Decay in Deep Learning

In neural networks, L2 regularization appears as **weight decay**. The update rule becomes:

$$w \leftarrow w(1 - \eta\lambda) - \eta \frac{\partial \text{Loss}}{\partial w}$$

The term $(1 - \eta\lambda)$ slightly shrinks every weight at each gradient step — hence "decay". This is equivalent to L2 regularization for SGD but **not exactly equivalent** for adaptive optimizers (e.g., Adam), where true weight decay (called **AdamW**) differs from L2 regularization.

---

## Probabilistic Interpretation

L2 regularization corresponds to placing a **Gaussian prior** on the weights:

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{0},\ \sigma^2 \mathbf{I})$$

Maximizing the **MAP (Maximum A Posteriori)** estimate under this prior is equivalent to minimizing the L2-regularized loss. The regularization strength $\lambda$ is inversely proportional to the prior variance $\sigma^2$.

> Compare: L1 regularization corresponds to a **Laplace prior** on the weights — which has heavier tails and a sharper peak at zero, explaining why L1 encourages sparse solutions.

---

## Practical Considerations

1. **Standardize features** before applying Ridge — the penalty is scale-sensitive; larger-scale features will be penalized disproportionately otherwise.
2. **Tune $\lambda$ via cross-validation** (e.g., `RidgeCV` in scikit-learn).
3. Ridge is preferred when **most features are expected to contribute** to the outcome.
4. Ridge is preferred over OLS when features are **highly correlated** (multicollinearity).
5. Ridge does **not** perform feature selection — use L1 (Lasso) or Elastic Net if you need sparsity.

---

## Python Example (scikit-learn)

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Best practice: scale features first
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))  # alpha = lambda
])
pipe.fit(X_train, y_train)

# Dense coefficients — all non-zero but shrunk
print(pipe.named_steps['ridge'].coef_)

# Cross-validate to find optimal alpha
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Optimal alpha: {ridge_cv.alpha_:.4f}")
```

---

## Summary

| Aspect | Detail |
|---|---|
| **Penalty** | $\lambda \sum w_j^2$ (L2 / squared norm) |
| **Key property** | Dense — all coefficients shrunk but never zeroed |
| **Use case** | Correlated features, stable estimates, $p > n$ settings |
| **Hyperparameter** | $\lambda$ (larger = more shrinkage) |
| **Solver** | Gradient descent or closed-form $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| **DL name** | Weight decay |
| **Bayesian view** | Gaussian prior on weights (vs. Laplace prior for L1) |
| **Limitation** | No feature selection; use Lasso or Elastic Net if sparsity needed |

> **Bottom line:** L2 regularization keeps all features in the model but reduces their influence, making it ideal when you believe every feature contributes and when features may be correlated. It is numerically stable and has a closed-form solution, making it computationally efficient.
