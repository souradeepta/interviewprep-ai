# Class Imbalance

**TL;DR:** When classes have very different prevalence, choose metrics, thresholds, and training procedures from error costs and deployment prevalence rather than accuracy alone.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), confusion matrices, and probability thresholds. You can select metrics for rare events, compare class weighting with resampling, tune a threshold against explicit costs, and avoid leakage during validation. This ML lesson focuses on decision quality for imbalanced prediction, complementing the general AI classification overview.

## Learning Outcomes

- Explain the central assumptions and mechanism of class imbalance.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

If one in a thousand transactions is fraud, a classifier that always says “legitimate” scores 99.9% accuracy while catching no fraud. Accuracy answers “how often was the label right?” but ignores which mistakes matter and how often the rare class appears. A useful system instead asks how many frauds it finds, how many alerts are false, what those errors cost, and how much investigation capacity exists.

Imbalance itself is not a model defect. Some tasks naturally have rare events; a calibrated model can still rank and estimate risk well. Resampling may change the effective training prior, so raw predicted probabilities can stop matching production prevalence.

## How It Works

For binary decisions, TP, FP, FN, and TN form the confusion matrix. Precision=TP/(TP+FP) measures alert purity; recall=TP/(TP+FN) measures captured positives. PR curves show precision versus recall across thresholds and are often more informative than ROC curves when positives are very rare. ROC AUC measures ranking across false-positive rates but can look strong while the operational alert volume is unacceptable. The Fβ score weights recall more when β>1; it is still a proxy rather than a business cost.

Class weighting changes the loss contribution per example. For logistic loss, a positive weight w+ makes missed positives more expensive. Oversampling duplicates or synthesizes minority training points; undersampling discards majority points. These can change optimization and decision boundaries. If sampling changes class prior, recalibrate on a validation set reflecting deployment prevalence. Always split first, then resample only within each training fold.

Metric choice should reflect the action after scoring. Precision answers what fraction of alerted cases are positive; recall answers what fraction of all positives were found. If review capacity is fixed, precision at the top K cases or recall at K may match operations better than a threshold-independent summary. If missing a positive is much more expensive than investigating a false alert, expected cost at candidate thresholds can be calculated from a cost matrix. The model can be well ranked while the chosen threshold is poor, so threshold selection is a separate step from training.

Class weighting and resampling solve different optimization problems. A class weight multiplies the loss for minority examples but leaves the observed feature distribution intact. Random oversampling repeats minority examples, effectively changing their contribution and the number of gradient updates. Synthetic sampling interpolates neighbors and can produce impossible combinations, especially for one-hot categories, counts, or constrained domains. Undersampling can make training manageable but may remove difficult negative examples that define the boundary. Compare against threshold adjustment alone; changing training data is not always necessary.

The deployment prior can differ from the training prior. A fraud dataset may be deliberately balanced to make training easier, while production positives are far rarer. A model trained on the balanced data can still rank cases usefully, but its estimated probabilities and default threshold may be wrong. Prior correction is only justified under assumptions about what changed and requires trustworthy deployment prevalence estimates. Validate calibration on data that preserves the natural class rate; a resampled validation set can conceal probability error.

Imbalance can vary across time and subgroups. Overall prevalence may remain constant while a particular customer segment experiences a sharp increase in positive events. Monitor class-conditional recall, precision, calibration, and alert volume for important segments, including support counts so uncertainty is visible. Rare-event systems often need delayed labels and careful backfill. Measure the age and completeness of labels, and avoid counting not-yet-observed outcomes as negatives. A correct evaluation protocol may be more valuable than a sophisticated resampling method.

## Trade-offs and Selection Boundary

| Intervention | Useful when | Risk / boundary |
|---|---|---|
| Threshold tuning | Ranking is useful but default 0.5 misses the operating objective | Threshold depends on costs, capacity, and prevalence |
| Class weights | Need to emphasize minority mistakes without changing rows | May distort probability calibration; tune weights |
| Random oversampling | Few positive examples limit gradient signal | Duplicates can overfit; do it inside training folds |
| SMOTE | Continuous features support meaningful interpolation | Synthetic points may be invalid for categorical or constrained data |
| Undersampling | Massive redundant majority data makes training costly | Throws away potentially valuable negative examples |

Choose the operating point using an explicit cost or capacity constraint. If downstream review can process 300 cases per day, evaluate precision and recall at that volume. Report prevalence and a confusion matrix at the selected threshold.

## Implementation

```python
def confusion(y, score, threshold):
    tp = fp = fn = tn = 0
    for truth, s in zip(y, score):
        pred = s >= threshold
        tp += truth == 1 and pred
        fp += truth == 0 and pred
        fn += truth == 1 and not pred
        tn += truth == 0 and not pred
    return tp, fp, fn, tn

y = [1, 1, 0, 0, 0, 0, 0, 0]
score = [.80, .30, .70, .20, .10, .05, .40, .01]
tp, fp, fn, tn = confusion(y, score, .5)
precision = tp/(tp+fp) if tp+fp else 0
recall = tp/(tp+fn) if tp+fn else 0
assert (tp, fp, fn, tn) == (1, 1, 1, 5)
assert precision == recall == .5
```

## Failure Modes and Diagnostics

1. **Accuracy-only evaluation:** majority prediction appears successful. Compare PR curve, recall at review capacity, and per-class confusion counts.
2. **Resampling before splitting:** near-duplicate synthetic or copied examples cross train/test. Split first and fit resampling inside each training fold.
3. **Treating weighted scores as probabilities:** estimated risk no longer reflects production prevalence. Evaluate calibration on untouched validation data with natural class frequencies.
4. **Optimizing recall without alert capacity:** recall rises while investigators are overwhelmed. Plot precision/recall against actual cases per day and set an operating constraint.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why is accuracy misleading? | A majority-only classifier can be highly accurate while never detecting the rare class. |
| Weights or resampling? | Weights alter loss; resampling alters the training sample. Compare both inside validation and check calibration. |
| Design follow-up: investigators handle 300 alerts/day. | Choose the score cutoff at that volume, report precision and recall there, then monitor prevalence and capacity drift. |

## Related Concepts

- [Evaluation metrics](evaluation-metrics.md), [cross-validation](cross-validation-strategies.md), and [data leakage](data-leakage.md)
- [Matching notebook](../notebooks/04-class-imbalance.ipynb)
- [Interview practice: evaluation and experiments](../interview-prep/evaluation-experiment-questions.md)

## References

- Saito and Rehmsmeier, [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://doi.org/10.1371/journal.pone.0118432).
- He and Garcia, [Learning from Imbalanced Data](https://doi.org/10.1109/TKDE.2008.239).
