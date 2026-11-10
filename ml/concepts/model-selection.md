# Model Selection

**TL;DR:** Model selection chooses a model family and configuration using a prespecified validation procedure, balancing predictive quality with data, latency, cost, interpretability, and maintenance constraints.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), [evaluation metrics](evaluation-metrics.md), and [cross-validation](cross-validation-strategies.md). You can establish a baseline, select metrics and splits that match deployment, compare model families under fair budgets, and avoid tuning the final test set. This ML lesson focuses on selection methodology; the AI roadmap links domain-specific model families.

## Learning Outcomes

- Explain the central assumptions and mechanism of model selection.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Choosing a model is like selecting a tool for a job, but predictive performance is only one requirement. A complex model may improve offline score while missing latency targets, requiring expensive serving hardware, or being difficult to audit. The right model is the simplest option that meets the quality and operational requirements with acceptable uncertainty. “Simplest” is relative to the task; it does not imply always choosing a linear model.

Selection also includes a statistical procedure. If a practitioner tries many architectures, features, and thresholds and reports the best validation result, that score is optimistically selected. The test set is reserved for estimating the chosen procedure after selection, not for choosing among candidates. On small datasets, nested cross-validation can estimate the combined tuning process at higher compute cost.

## How It Works

First define the prediction unit, target horizon, eligible population, action, and costs of false positives/negatives. Choose a metric or metric set and split strategy before comparing candidates. For example, a churn prediction task may need chronological validation and recall at an intervention budget; random accuracy is not a sufficient objective. Record latency, memory, training time, interpretability needs, and update cadence as explicit constraints.

Build a baseline that is hard to misunderstand: majority/mean predictor, logistic or linear model, or a shallow tree. Confirm that features and labels align, the training pipeline fits a tiny subset, and the evaluation score is plausible. Then compare model families suited to the data. Linear models are efficient and interpretable but need useful features for nonlinear structure. Trees capture thresholds and interactions on tabular data; ensembles improve predictive power with additional cost. SVMs can work well on smaller scaled datasets but kernel methods scale poorly in sample count. Neural networks exploit large-scale, unstructured modalities and learned representations but demand more data and compute. K-nearest neighbors can be intuitive but has prediction-time and storage costs.

Keep comparisons fair: use the same split, preprocessing information, label availability, and comparable search budget. Tune hyperparameters using only training/validation data. If candidate families require different preprocessing, put each pipeline inside cross-validation. Repeat finalists across seeds or folds when score uncertainty could alter the choice. Evaluate the selected model once on a locked test set, then separately benchmark it on target serving infrastructure.

The selection decision is often a Pareto choice. Candidate A dominates B if A is at least as good on every relevant criterion and strictly better on one. When candidates trade quality against latency, present the frontier and let product constraints set feasibility. Model interpretability should also be specified precisely: a globally linear coefficient is different evidence from a post hoc feature attribution. Explainability methods do not automatically establish causality or fairness.

## Trade-offs and Selection Boundary

| Model family | Strength | Limitation / useful boundary |
|---|---|---|
| Linear/logistic | Fast, small-data baseline, transparent coefficients | Needs feature representation for nonlinearities and interactions |
| Single tree | Nonlinear thresholds and easy local reasoning | High variance; unstable trees may change with data |
| Random forest | Robust bagging and feature subsampling | Larger memory and less direct global interpretation |
| Gradient boosting | Strong tabular accuracy and interactions | More tuning and sequential training cost |
| SVM | Maximum-margin model; kernels enable nonlinear boundaries | Scaling and probability calibration can be difficult |
| Neural network | Learns rich representations for images, text, and large-scale data | More data/compute, tuning, and serving complexity |

Prefer a strong baseline when it meets quality and serving constraints. Move to a richer family when residual analysis shows structure the baseline misses. Choose a neural model naturally when raw modality structure and data scale support it; do not assume it wins for tabular data. If latency dominates, measure prediction cost and batch behavior rather than infer it from model size.

## Implementation

```python
def feasible(candidate, min_quality, max_latency, max_memory):
    return (candidate["quality"] >= min_quality and
            candidate["latency_ms"] <= max_latency and
            candidate["memory_mb"] <= max_memory)

candidates = [
    {"name":"linear", "quality":.84, "latency_ms":3, "memory_mb":40},
    {"name":"boosted", "quality":.90, "latency_ms":15, "memory_mb":220},
    {"name":"large_nn", "quality":.92, "latency_ms":60, "memory_mb":900},
]
eligible = [m for m in candidates if feasible(m, .85, 20, 300)]
assert [m["name"] for m in eligible] == ["boosted"]
```

## Failure Modes and Diagnostics

1. **Only the highest metric is considered:** selected model violates cost or interpretability needs. Define feasibility constraints and report the Pareto frontier.
2. **Test set becomes a tuning dashboard:** adaptive choices inflate final performance. Lock test access and use validation or nested CV for decisions.
3. **Unfair preprocessing:** one family sees globally fitted transforms or more feature information. Put preprocessing inside each fold and audit identical input cutoffs.
4. **Random split ignores time or group structure:** the selected family benefits from leakage. Compare with realistic forward/group-held-out evaluation.
5. **Winner selected on noise:** score differences are smaller than fold/seed variability. Repeat finalists and report uncertainty, not only point estimates.

## Interview Quick Reference

| Question | Answer |
|---|---|
| How choose a model? | Define task and constraints, build a baseline, compare suitable families fairly on validation, then test the selected procedure once. |
| What is the no-free-lunch idea? | No learner is uniformly best across all possible tasks; performance depends on assumptions and data structure. |
| Design follow-up: model A is 2% more accurate but 5× slower. | Check whether the score gain is reliable and whether latency meets constraints; compare feasible points with cost and operational needs. |

## Related Concepts

- [Hyperparameter tuning](hyperparameter-tuning.md), [evaluation metrics](evaluation-metrics.md), and [overfitting](overfitting-underfitting.md)
- [Matching notebook](../notebooks/24-model-selection.ipynb)
- [Interview practice: cross-validation](../interview-prep/ml-theory-questions.md#q-how-does-cross-validation-work-and-when-do-you-use-it)

## References

- scikit-learn, [Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html).
- Cawley and Talbot, [On Over-fitting in Model Selection and Subsequent Selection Bias](https://jmlr.org/papers/v11/cawley10a.html).
