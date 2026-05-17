# Ensemble Methods

## TL;DR
Combining multiple models to outperform any individual model. Three paradigms: bagging
(Random Forest), boosting (XGBoost, LightGBM), and stacking. Gradient boosted trees are
the dominant approach for tabular data in industry.

## Core Intuition
Ask 100 experts instead of one. Each expert is sometimes wrong, but errors are uncorrelated,
so averaging cancels noise. This reduces variance (bagging) or reduces bias iteratively (boosting).

## How It Works

**Bagging:** train T models on different bootstrap samples, aggregate by vote/mean.
Random Forest adds feature subsampling at each split to de-correlate trees.

**Boosting:** train sequentially, each model correcting previous errors.
- AdaBoost: upweight misclassified samples
- Gradient Boosting: fit new tree to negative gradient (residuals)
- XGBoost/LightGBM: GPU-optimized, regularized, histogram-based gradient boosting

**Stacking:** train a meta-model on out-of-fold predictions of diverse base models.

## Key Properties / Trade-offs
- Bagging: reduces variance — best when base models overfit
- Boosting: reduces bias and variance — best when base models underfit; sensitive to outliers
- XGBoost > Random Forest on most tabular benchmarks, but harder to tune

## Common Mistakes / Gotchas
- n_estimators with boosting: lower learning_rate needs more trees
- Stacking without proper CV leaks meta-training labels
- Boosting on noisy labels amplifies noise — consider robust loss functions

## Code Example
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
print("RF:", cross_val_score(rf, X, y, cv=5, scoring='roc_auc').mean())
print("GB:", cross_val_score(gb, X, y, cv=5, scoring='roc_auc').mean())
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "How does Random Forest reduce variance?" | Bootstrap sampling + feature subsampling at each split produces uncorrelated trees. Averaging uncorrelated predictions reduces variance. |
| "Bagging vs boosting?" | Bagging: parallel, reduces variance. Boosting: sequential, corrects errors, reduces bias. |
| "XGBoost vs Random Forest?" | XGBoost typically achieves lower error but requires more tuning; RF is faster and more robust to hyperparameters. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Implementations: Random Forest](../implementations/random-forest.ipynb)

## Resources
- [XGBoost paper](https://arxiv.org/abs/1603.02754) — Chen & Guestrin
- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) — Breiman
