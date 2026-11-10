# Evaluation Metrics

**TL;DR:** A metric turns prediction outcomes into a score, so choose it to reflect the task, class prevalence, error costs, and deployment decision.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), confusion matrices, and probability estimates. You can distinguish ranking from thresholded decisions, select regression/classification metrics, quantify uncertainty, and recognize when aggregate scores hide a failure. The ML lesson grounds metric choice in decision consequences; the AI overview's evaluation discussion is broader.

## Learning Outcomes

- Explain the central assumptions and mechanism of evaluation metrics.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Metrics are measuring instruments, not definitions of quality. A bathroom scale does not tell a pilot whether an aircraft is safe; accuracy does not tell a fraud team whether its alert queue is useful. Start from the action the model supports and the cost of each error, then select metrics and slices that reveal those outcomes.

No single metric summarizes every trade-off. AUC measures ranking over thresholds, not calibration or the chosen operating point. RMSE punishes large errors more than MAE. AUC or accuracy can hide group, temporal, or rare-class failures.

## How It Works

For binary classification, precision=TP/(TP+FP), recall=TP/(TP+FN), and specificity=TN/(TN+FP). F1 is the harmonic mean of precision and recall. ROC AUC evaluates ranking via true-positive versus false-positive rates; PR AUC emphasizes positive-class retrieval and is often more informative with rare positives. Log loss scores probabilistic forecasts and penalizes confident mistakes. Brier score is mean squared probability error. Calibration asks whether events assigned probability p occur about p fraction of the time.

For regression, MAE=(1/n)Σ|y−ŷ| is robust to large residuals relative to squared error; RMSE=√((1/n)Σ(y−ŷ)²) retains target units but emphasizes outliers. R² compares against a mean baseline and may be negative on held-out data. For ranking, NDCG weights correct ordering near the top. Always calculate metrics on data not used to fit the model, and report uncertainty or repeated split variability where decisions require it.

Metric aggregation is itself a modeling choice. Micro-averaging sums counts across examples and lets frequent classes dominate; macro-averaging computes a score per class and weights classes equally. Weighted averages sit between those objectives. A metric can therefore change substantially without any prediction changing if class prevalence changes. State the averaging rule, label support, and missing-class behavior whenever reporting a multi-class summary. For ranking, the candidate set and relevance labels determine interpretation; NDCG at 10 says little about relevant items that were never retrieved into the candidate pool.

Thresholded metrics require a threshold. The conventional 0.5 cutoff is not automatically optimal, especially with asymmetric costs or prevalence shift. Select the cutoff on validation data according to an explicit criterion such as cost, recall at fixed precision, or capacity. Once selected, freeze it before final testing and include it in the deployed artifact. If scores are not calibrated, a threshold may still work empirically but will need revalidation when population or model changes.

Uncertainty in a metric matters when samples are few or decisions are close. Bootstrap confidence intervals should resample the independent unit; for clustered data, resampling rows independently understates uncertainty. Paired comparisons should evaluate models on the same cases to reduce noise, and the analysis should account for multiple variants if many are compared. An interval describes sampling variability under the chosen procedure, not all uncertainty from label quality, distribution shift, or measurement error.

Monitoring should separate model quality from data and operational health. A decline in recall may be caused by delayed labels, threshold drift, a change in prevalence, an upstream feature outage, or actual model degradation. Track input missingness, score distributions, label latency, metric support counts, and subgroup metrics. A dashboard of a single scalar can hide the mechanism and prompt the wrong response, such as retraining when a point-in-time feature join has broken.

## Trade-offs and Selection Boundary

| Metric | Answers | Does not answer |
|---|---|---|
| Accuracy | Fraction of correct hard labels | Which errors matter or ranking quality |
| PR AUC / recall@precision | Rare positive retrieval trade-off | Calibration and exact business cost |
| ROC AUC | Pairwise ranking quality across thresholds | Operational alert volume or probability quality |
| Log loss / Brier | Quality of probability forecasts | Best action without costs/threshold |
| MAE / RMSE | Typical absolute / squared-error magnitude | Whether errors are acceptable by subgroup |
| NDCG@k | Quality near top of ranked list | Overall calibration or recall beyond k |

Use metrics aligned to the deployment action; for high-stakes decisions, report multiple complementary measures, thresholds, confidence intervals, and subgroup slices. Select a threshold on validation data, freeze it, then report test performance once.

## Implementation

```python
def binary_metrics(y, score, threshold=.5):
    tp=fp=fn=tn=0
    for truth, probability in zip(y, score):
        pred = probability >= threshold
        if truth and pred: tp += 1
        elif pred: fp += 1
        elif truth: fn += 1
        else: tn += 1
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    accuracy = (tp+tn)/len(y)
    return accuracy, precision, recall

y=[1,0,0,0,0]; scores=[.4,.2,.1,.05,.03]
acc, precision, recall = binary_metrics(y, scores, .5)
assert acc == .8 and precision == recall == 0.0
assert binary_metrics(y, scores, .3)[2] == 1.0
```

## Failure Modes and Diagnostics

1. **Accuracy on rare positives:** majority baseline appears excellent. Compare PR curve and recall at operational alert capacity.
2. **AUC presented as launch readiness:** a good ranker can have poor calibration or a bad chosen threshold. Measure calibration and costs at the frozen operating point.
3. **Aggregate score hides segment harm:** one large cohort dominates. Report per-group, time, geography, and key subpopulation metrics with support counts.
4. **Repeated test-set tuning:** test score becomes another validation score. Track experiment exposure and preserve an untouched evaluation set.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Precision versus recall? | Precision is the fraction of alerts that are true positives; recall is the fraction of positives caught. |
| AUC versus log loss? | AUC measures ranking; log loss evaluates probabilities and penalizes confident errors. |
| Design follow-up: fraud ops can review 100 cases/day. | Select threshold at that capacity, report precision/recall and cost there, and monitor prevalence and drift. |

## Related Concepts

- [Class imbalance](class-imbalance.md), [model selection](model-selection.md), and [cross-validation](cross-validation-strategies.md)
- [Matching notebook](../notebooks/13-evaluation-metrics.ipynb)
- [Interview practice: evaluation and experiments](../interview-prep/evaluation-experiment-questions.md)

## References

- Saito and Rehmsmeier, [Precision-Recall Plot Is More Informative than ROC for Imbalanced Data](https://doi.org/10.1371/journal.pone.0118432).
- scikit-learn, [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
