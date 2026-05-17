# Class Imbalance

## TL;DR
When positive class is rare (0.1% fraud), accuracy is misleading — a model predicting all negatives gets 99.9% accuracy. Use precision-recall AUC, F1, or cost-sensitive learning. Resampling (oversample minority, undersample majority) or class weights are standard fixes.

## Core Intuition
If 99.9% of transactions are legitimate, a naive model learns "always say legitimate" and gets 99.9% accuracy but zero fraud detection. The metric and the model must match the business problem.

## How It Works

**Metrics that work:** precision-recall AUC, F1, recall@precision(threshold).

**Class weights:** give more penalty to misclassifying the minority class.
```python
# sklearn
LogisticRegression(class_weight='balanced')  # weight inversely proportional to frequency
```

**Resampling:**
- **Oversample minority:** duplicate or synthesize examples (SMOTE)
- **Undersample majority:** randomly drop examples
- **Hybrid:** combine both

**SMOTE:** for each minority sample, find k nearest neighbors. Generate synthetic samples on line between original and neighbor.

## Key Properties / Trade-offs
- **Oversampling:** risk of overfitting to synthetic minority examples
- **Undersampling:** lose information from majority class
- **Threshold tuning:** adjust decision threshold for desired recall/precision trade-off
- **Cost-sensitive learning:** weight examples inversely to class frequency

## Common Mistakes / Gotchas
- **Using accuracy:** meaningless for imbalanced data
- **Resampling before CV:** causes data leakage. Do it inside CV folds.
- **Ignoring recall:** if FN is costly, optimize recall, not precision

## Code Example
```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

X_train, y_train = load_data()
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
clf = LogisticRegression()
clf.fit(X_resampled, y_resampled)
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Accuracy misleads on imbalance?" | Yes. 99.9% accuracy if always predicting majority. Use precision-recall AUC or F1. |
| "Class weights vs resampling?" | Class weights: simple, no data duplication. Resampling: more flexibility, SMOTE for synthesis. |
| "SMOTE risks?" | Overfitting to synthetic minority samples. Use carefully in small datasets. |

## Related Topics
- [Evaluation Metrics](evaluation-metrics.md) — [Supervised Learning](supervised-learning.md)

## Resources
- [Imbalanced Learning (imbalanced-learn)](https://imbalanced-learn.org/)
