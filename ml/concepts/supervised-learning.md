# Supervised Learning

## TL;DR
Supervised learning trains a model on labeled input-output pairs so it can predict outputs for
unseen inputs. The most common ML paradigm — underpins classification, regression, and ranking.

## Core Intuition
A student learning from an answer key. You show the model thousands of (question, correct answer)
pairs. It adjusts internal parameters until it reliably predicts answers. The "supervision" is
the label — the signal that tells the model when it is wrong.

## How It Works
Given dataset $\{(x_i, y_i)\}_{i=1}^n$, learn $f: X \to Y$ minimizing a loss $\mathcal{L}$.

**Regression** (continuous $y$): Mean Squared Error
$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Classification** (discrete $y$): Cross-Entropy Loss
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \sum_{c} y_{ic} \log(\hat{p}_{ic})$$

Optimization via gradient descent: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.

## Key Properties / Trade-offs
- Requires labeled data — expensive at scale
- Generalizes poorly outside training distribution (distribution shift)
- Bias-variance trade-off: high-capacity models have low bias but high variance

## Common Mistakes / Gotchas
- Evaluating on training set — always use a held-out test set
- Data leakage: future information in training features
- Class imbalance: accuracy misleads when 99% of samples are one class
- Not normalizing features for gradient-based or distance-based models

## Code Example
```python
import numpy as np

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)
X_b = np.hstack([np.ones((len(X), 1)), X])
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # normal equation
print(theta)  # [intercept, slope]
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is supervised learning?" | Learning input→output mapping from labeled data by minimizing a loss function |
| "What is the bias-variance trade-off?" | Bias = underfitting (too simple); Variance = overfitting (too sensitive to training data) |
| "How do you prevent overfitting?" | Regularization, dropout, early stopping, more data, cross-validation |
| "What is data leakage?" | Future or test information contaminating training — causes inflated metrics that don't hold in production |

## Related Topics
- [Optimization](optimization.md) — [Regularization](regularization.md) — [Evaluation Metrics](evaluation-metrics.md)

## Resources
- [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf)
- [Understanding Bias-Variance](http://scott.fortmann-roe.com/docs/BiasVariance.html)
