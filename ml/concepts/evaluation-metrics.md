# Evaluation Metrics

## TL;DR
Choosing the right metric is as important as choosing the right model. Accuracy misleads on
imbalanced data. Precision/recall trade-off depends on the cost of each error type. AUC-ROC
summarizes performance across all thresholds.

## Core Intuition
The metric you optimize determines what your model learns. If you optimize accuracy on a
99%-negative dataset, predicting "negative" always gets 99% accuracy while being useless.
Match the metric to the business cost of different error types.

## How It Works

**Confusion matrix:** TP, TN, FP (Type I error), FN (Type II error)

$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}$$
$$\text{F1} = 2 \cdot \frac{P \cdot R}{P + R}$$

**AUC-ROC:** probability that a random positive ranks above a random negative. Threshold-independent.

**Regression:** MSE (penalizes outliers), MAE (robust), RMSE (interpretable units), R².

## Key Properties / Trade-offs
- High precision = few false alarms; high recall = few missed detections
- Increasing threshold: raises precision, lowers recall
- Use F1 when both matter equally; AUC when threshold is unknown

## Common Mistakes / Gotchas
- Using accuracy on imbalanced datasets
- Reporting precision/recall without specifying threshold
- AUC doesn't tell you performance at your operating threshold

## Code Example
```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
y_prob = np.array([0.9, 0.2, 0.8, 0.4, 0.1, 0.85])
print(f"F1: {f1_score(y_true, y_pred):.2f}, AUC: {roc_auc_score(y_true, y_prob):.2f}")
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "When would you use precision over recall?" | When FP is expensive (spam filter). Use recall when FN is expensive (cancer detection). |
| "What is AUC-ROC?" | P(random positive ranks above random negative). Threshold-independent. 0.5=random, 1.0=perfect. |
| "How do you handle class imbalance?" | Resample, class weights, use F1/AUC over accuracy, precision-recall curve. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Feature Engineering](feature-engineering.md)

## Resources
- [sklearn metrics docs](https://scikit-learn.org/stable/modules/model_evaluation.html)
