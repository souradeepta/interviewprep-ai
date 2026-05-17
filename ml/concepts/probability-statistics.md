# Probability & Statistics

## TL;DR
The mathematical foundation of ML. Key concepts: probability distributions, Bayes' theorem,
MLE, hypothesis testing, CLT. These appear directly in interviews and underpin every ML algorithm.

## Core Intuition
ML is reasoning under uncertainty. A model that outputs "80% confident" is making a probability
claim. Statistics tells us: is this signal real, or just noise?

## How It Works

**Bayes' Theorem:** $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$ — prior belief updated by evidence.

**Common distributions:**
- Normal $\mathcal{N}(\mu, \sigma^2)$: sum of many independent r.v.s; central to statistics
- Bernoulli / Binomial: binary outcomes
- Poisson: count of events in a fixed interval

**MLE:** $\hat{\theta} = \arg\max_\theta \prod P(x_i | \theta)$ — parameters most likely to produce observed data.
In practice use log-likelihood (converts product to sum).

**Central Limit Theorem:** sample mean of n i.i.d. r.v.s converges to Normal as n→∞, regardless
of underlying distribution.

**p-value:** P(data this extreme | null hypothesis is true). p < 0.05 → reject null.

## Key Properties / Trade-offs
- MLE can overfit with small data — MAP (+ prior) = regularized MLE
- p-values measure significance, not effect size or practical importance
- Multiple comparisons inflate false positive rate — Bonferroni or FDR correction

## Common Mistakes / Gotchas
- Confusing P(A|B) with P(B|A) — base rate fallacy
- Thinking p < 0.05 means the result is "real" or "important"
- Mixing up standard deviation (σ) and standard error (σ/√n)

## Code Example
```python
import numpy as np
from scipy import stats

data = np.random.normal(loc=5, scale=2, size=1000)
mu_mle, sigma_mle = np.mean(data), np.std(data, ddof=0)
print(f"MLE: mu={mu_mle:.2f}, sigma={sigma_mle:.2f}")

a, b = np.random.normal(10, 2, 100), np.random.normal(10.5, 2, 100)
t, p = stats.ttest_ind(a, b)
print(f"t={t:.3f}, p={p:.4f}")
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is Bayes' theorem?" | P(A|B) = P(B|A)·P(A)/P(B). Updates prior belief with observed evidence to get posterior. |
| "What is MLE?" | Find parameters that maximize P(observed data | parameters). For Gaussian: sample mean and variance. |
| "What is CLT?" | Distribution of sample means → Normal as n→∞, regardless of underlying distribution. Justifies many statistical tests. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [System Design: A/B Testing](../../system-design/patterns/ab-testing.md)

## Resources
- [Think Stats](https://greenteapress.com/wp/think-stats-2e/) — Downey. Free. Python-based.
