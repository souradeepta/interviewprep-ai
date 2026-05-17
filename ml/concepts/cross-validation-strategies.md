# Cross-Validation Strategies

## TL;DR
K-fold CV: partition data into k folds, train on k-1, evaluate on remaining, repeat k times, average scores. Reduces variance in generalization estimate. Different stratification strategies (stratified, time-based, group-based) for different data types.

## Core Intuition
One train/test split gives you one noisy estimate. K-fold gives you k estimates that average to something more reliable.

## How It Works

**K-Fold:** split into k disjoint folds. Train on k-1, eval on 1. Repeat k times.
- Score = average of k scores
- Confidence = std of k scores
- Standard: k=5 or k=10

**Stratified K-Fold:** preserve class proportions in each fold.
- For imbalanced data: each fold has same positive rate as full dataset
- Always use for classification

**Time-Series CV:** respect temporal order (no future leakage).
- Fold 1: train on [0,100), test on [100,150)
- Fold 2: train on [0,150), test on [150,200)
- Expanding window (train size grows), or fixed window (sliding)

**Group K-Fold:** ensure different groups in train and test.
- User-based: if data has user_id, ensure no user appears in both train and test
- Prevent: model learning user-specific patterns that don't generalize

**Leave-One-Out CV (LOOCV):** k=n (each fold has 1 test sample).
- Unbiased estimate but computationally expensive
- Use for small datasets only

## Key Properties / Trade-offs
- **Bias-variance:** k=3 faster but noisier; k=10 more reliable but slower
- **Stratification:** essential for imbalance; not needed if classes balanced
- **Temporal:** always use time-based split for time series, never random
- **Computational cost:** O(k · train_time)

## Common Mistakes / Gotchas
- **Non-stratified on imbalanced data:** some folds may have 0 positives
- **Temporal CV on non-temporal data:** wastes information
- **Hyperparameter tuning on same CV folds:** nested CV needed (outer for eval, inner for tuning)
- **Reporting only mean, not std:** underestimates uncertainty

## Code Example
```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold

# Stratified
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    # train and evaluate

# Time-series
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # respects temporal order

# Group-based
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, groups=user_ids):
    # ensures no user in both train and test
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "K-fold purpose?" | Reduce variance in generalization estimate. Average k folds for more reliable evaluation. |
| "When stratified?" | Always for classification. Essential for imbalanced data. |
| "Time series CV?" | Respect temporal order. No future leakage. Use expanding or fixed window. |

## Related Topics
- [Data Leakage](data-leakage.md) — [Hyperparameter Tuning](hyperparameter-tuning.md)

## Resources
- [sklearn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
