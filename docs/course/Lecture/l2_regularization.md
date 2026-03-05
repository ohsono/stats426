
# $L_2$ regularization:

Regularization Formula (L2 / Weight Decay)
The "regularization issue" typically refers to overfitting or multicollinearity, where the model learns large, unstable weights to fit noise in the training data.

The formula addresses this by adding a penalty term to the original Loss function (e.g., Mean Squared Error or Cross-Entropy). Here is the standard formula for L2 Regularization:

$$ J(\theta) = L(\theta) + \lambda \sum_{i} w_i^2 $$

Where:

$J(\theta)$ is the total regularized loss.
$L(\theta)$ is the original loss (e.g., prediction error).
$\lambda$ (lambda) is the regularization strength (hyperparameter).
$\sum w_i^2$ is the penalty term (sum of squared weights), forcing weights to be small.
How it affects the update (Gradient Descent):
When we take the derivative (gradient) to update weights: $$ w_{new} = w_{old} - \eta \cdot \nabla J(\theta) $$ $$ w_{new} = w_{old} - \eta \cdot (\nabla L(\theta) + 2\lambda w_{old}) $$ $$ w_{new} = (1 - 2\eta\lambda)w_{old} - \eta \cdot \nabla L(\theta) $$

Notice the term $(1 - 2\eta\lambda)w_{old}$. At every step, the weight is multiplied by a factor slightly less than 1 (decayed) before subtracting the gradient of the loss. This is why L2 regularization is called Weight Decay. By constantly shrinking weights towards zero, it prevents any single feature (in a correlated group) from dominating, thus stabilizing the model against multicollinearity.


## How does it work?

increase $\lambda$ -> decrease $|\beta||_2^2$ -> decrease variance -> increase bias

$f(\beta) = 1/2n|| y-X\beta||_2^2 + \lambda ||\beta||_2^2$

$f(\beta) = 1/2n (B^TX^T - 2B^TX^Ty + y^Ty) + \lambda B^TB$
$= 1/2n (B^T(X^TX + 2n\lambda I)B - 2B^TX^Ty + y^Ty)$
