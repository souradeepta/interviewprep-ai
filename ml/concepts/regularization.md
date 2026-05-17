# Regularization

## TL;DR
Regularization reduces overfitting by constraining model complexity. The main techniques:
L1 (Lasso), L2 (Ridge), dropout, early stopping, and data augmentation. Every production
ML model uses at least one form of regularization.

## Core Intuition
A penalty for complexity. Without it, a model memorizes training data. With it, the model
is forced to learn simpler rules that generalize. Occam's Razor baked into the math.

## How It Works

**L2 (Ridge):** $\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_j w_j^2$ — drives weights toward zero, not exactly zero.

**L1 (Lasso):** $\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_j |w_j|$ — drives many weights to exactly zero (sparsity).

**Dropout:** randomly zero neurons during training with probability p. At inference, disable.

**Early stopping:** monitor validation loss and stop when it starts increasing. Free regularization.

**Data augmentation:** artificially expand training data (flips, crops, noise). Especially effective for images.

## Key Properties / Trade-offs
- L1 vs L2: L1 for feature selection (sparse); L2 otherwise (smooth)
- Higher λ = stronger regularization = more bias, less variance
- Dropout most effective in fully connected layers

## Common Mistakes / Gotchas
- Not tuning λ — regularization strength needs cross-validation
- Applying dropout at inference time
- Over-regularizing: causes underfitting

## Code Example
```python
import numpy as np

def l2_loss(y_true, y_pred, weights, lam=0.01):
    return np.mean((y_true - y_pred)**2) + lam * np.sum(weights**2)

def l1_loss(y_true, y_pred, weights, lam=0.01):
    return np.mean((y_true - y_pred)**2) + lam * np.sum(np.abs(weights))
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "L1 vs L2?" | L1 penalizes absolute weights — drives some to exactly zero (feature selection). L2 penalizes squared weights — all shrink toward zero but none exactly zero. |
| "How does dropout work?" | Randomly zeros neurons during training, forcing redundant representations. Acts as an ensemble of sub-networks. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Optimization](optimization.md) — [Neural Networks](neural-networks.md)

## Resources
- [Dropout paper](https://jmlr.org/papers/v15/srivastava14a.html) — Srivastava et al.
