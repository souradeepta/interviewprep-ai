# Optimization

## TL;DR
Finding model parameters that minimize a loss function. Gradient descent and its variants (SGD,
Adam) are the standard. Understanding optimization is essential for debugging training, tuning
hyperparameters, and explaining why models behave as they do.

## Core Intuition
Standing on a hilly landscape in fog, trying to reach the lowest valley. Gradient descent says:
check which direction is downhill (the gradient), take a step that way. The learning rate controls
step size — too large and you overshoot; too small and you never arrive.

## How It Works

**Gradient Descent:** $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$

**SGD (stochastic):** one sample per update — noisier but faster iteration.

**Mini-batch SGD:** gradient over B samples — best trade-off. Standard in practice.

**Momentum:** adds velocity term to smooth oscillations:
$v \leftarrow \beta v + (1-\beta)\nabla\mathcal{L}$, $\theta \leftarrow \theta - \eta v$

**Adam:** adaptive per-parameter learning rates via first and second moment estimates.
Defaults: lr=1e-3, β₁=0.9, β₂=0.999. Converges fast; may not generalize as well as SGD.

## Key Properties / Trade-offs
- SGD generalizes better than Adam in some settings (finds flatter minima)
- Adam converges faster but can overfit to sharp minima
- Learning rate is the most impactful hyperparameter — use a scheduler
- Larger batch = less gradient noise = faster convergence but sometimes worse generalization

## Common Mistakes / Gotchas
- Not normalizing inputs — gradients explode or vanish
- Learning rate too high: loss oscillates; too low: painfully slow
- Forgetting learning rate warmup for transformers

## Code Example
```python
import numpy as np

def adam(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t += 1
    m = b1 * m + (1 - b1) * grads
    v = b2 * v + (1 - b2) * grads**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    params -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v, t
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Why use Adam over SGD?" | Adam adapts learning rates per parameter and converges faster; SGD with momentum can generalize better |
| "What is a learning rate scheduler?" | Adjusts lr during training — warmup avoids early instability, cosine decay improves final convergence |
| "Why does SGD generalize better?" | Noisier gradients find flatter minima that tend to generalize better |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Regularization](regularization.md) — [Neural Networks](neural-networks.md)

## Resources
- [Overview of Gradient Descent Optimizers](https://arxiv.org/abs/1609.04747) — Ruder
- [Adam paper](https://arxiv.org/abs/1412.6980) — Kingma & Ba
