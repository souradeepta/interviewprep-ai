# Supervised Learning

**TL;DR:** Supervised learning fits a mapping from labeled examples so predictions remain useful on future examples drawn from the intended deployment population.

## Prerequisites

Prerequisites: basic probability, vectors, and [probability and statistics](probability-statistics.md). You can formulate a supervised task, distinguish regression/classification/ranking, define a loss and metric, create a valid evaluation split, and recognize label/feature timing risks. This ML lesson covers the learning problem and generalization contract; the AI roadmap gives an overview of when models are used.

## Learning Outcomes

- Explain the central assumptions and mechanism of supervised learning.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A labeled dataset is a collection of examples paired with outcomes. A model is shown input-output pairs and adjusts its parameters to predict outcomes for examples it has not seen. The labels provide supervision: they indicate the desired response. The key objective is not to reproduce the training rows but to learn a relationship that continues to hold when the system encounters future users, objects, or events.

Think of teaching a student from worked examples. If the student memorizes each exact answer, they may fail on a new problem; if they learn a useful rule, performance transfers. The difference is generalization. Whether a model generalizes depends on representative data, a sound split, useful features, suitable capacity, and a target that is measured consistently.

## How It Works

Let D={(xᵢ,yᵢ)}ᵢ₌₁ⁿ be labeled observations and fθ a parameterized predictor. Training minimizes empirical risk J(θ)=n⁻¹Σᵢℓ(fθ(xᵢ),yᵢ), possibly with a regularization term. Regression predicts a continuous quantity, classification predicts a discrete class or probability, and ranking orders candidates. The loss provides an optimization signal; the evaluation metric estimates task utility and may differ.

Squared-error regression minimizes (y−ŷ)² and emphasizes large residuals; absolute error is more robust and targets a conditional median. Binary cross-entropy trains probabilistic classification; multiclass cross-entropy typically models one class among K. Pairwise/listwise ranking losses optimize ordering. A loss must match target representation and output semantics. For example, mutually exclusive classes use a softmax distribution, while independent multi-label targets require separate probabilities.

Evaluation divides data according to the claim the model must support. A validation set guides architecture and hyperparameter choices; a final test set estimates the selected process after choices are locked. For repeated users, patients, devices, or future outcomes, split by group or time as deployment requires. Feature transforms such as scaling or vocabulary fitting are learned from training data only. A pipeline should capture transformations and schema with the model so predictions at serving reproduce training semantics.

Supervised labels have a data-generating process and may be noisy, delayed, biased, or selectively observed. If only cases investigated by a human are labeled, the observed sample may not represent all future cases. If a feature is produced after the target event, performance may be inflated by leakage. Define the prediction timestamp, label window, observation unit, and availability of each feature before training. These are part of the task definition, not post hoc implementation details.

Generalization depends on the relation between training and deployment distributions. Under a simple IID assumption, examples are independently sampled from a stable distribution. Real production data often violates this through seasonality, policy changes, user populations, or feedback loops. Track performance over time and segments, and establish an update/retraining policy if distributions move. A single offline score is evidence for a specific split, not a permanent guarantee.

## Trade-offs and Selection Boundary

| Learning setup | Use when | Boundary |
|---|---|---|
| Regression | Target is continuous and numeric error matters | Outliers and asymmetric costs require another loss/metric |
| Classification | Outcomes are categorical | Class prevalence, calibration, and thresholds affect action quality |
| Ranking | Ordering candidates matters more than absolute score | Candidate generation and position bias influence results |
| Semi-supervised | Unlabeled data is plentiful and label budget is limited | Pseudo-labels can reinforce errors |
| Weak supervision | Labels come from rules or noisy sources | Correlated label functions create overconfidence |

Use supervised learning when representative labels are available and the desired output can be specified. If labels are sparse, expensive, or systematically biased, consider active learning, weak supervision, or unsupervised discovery, but keep the final evaluation independent. For tabular tasks, begin with a simple linear or tree baseline before assuming a deep network is necessary.

## Implementation

This tiny deterministic least-squares example fits a line through labeled examples with the normal equation. It is suitable only when the matrix inverse exists and the problem is small; robust code uses a numerical solver.

```python
X = [1., 2., 3., 4.]
y = [3., 5., 7., 9.]  # y = 1 + 2x
x_mean, y_mean = sum(X)/len(X), sum(y)/len(y)
slope = sum((x-x_mean)*(t-y_mean) for x,t in zip(X,y)) / sum((x-x_mean)**2 for x in X)
intercept = y_mean-slope*x_mean
pred = [intercept+slope*x for x in X]
assert abs(slope-2) < 1e-12 and abs(intercept-1) < 1e-12
assert sum((a-b)**2 for a,b in zip(pred,y)) < 1e-20
```

## Failure Modes and Diagnostics

1. **Training and test rows are near duplicates:** score measures memorization. Split by identity/source and quantify entity overlap.
2. **Target unavailable when features are computed:** offline model uses future information. Audit event and availability timestamps; replay features at the actual prediction cutoff.
3. **Wrong label semantics:** missing or delayed outcomes are treated as negatives. Define censoring and label windows, and inspect adjudicated samples.
4. **Metric/loss mismatch:** a better training loss does not improve the action. Evaluate a prespecified deployment metric and threshold/cost curve.
5. **Distribution shift:** accuracy drops on later cohorts. Report temporal and subgroup metrics, monitor drift, and collect labels for affected populations.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What is supervised learning? | Fitting a predictor from labeled examples and evaluating whether it generalizes to the intended unseen population. |
| Why separate validation and test? | Validation guides choices; untouched test estimates the chosen procedure without those adaptive decisions. |
| Design follow-up: predict whether a user churns next month. | Define cutoff and label horizon, build only past-available features, split chronologically/group-aware, and evaluate calibrated risk plus business recall/cost. |

## Related Concepts

- [Evaluation metrics](evaluation-metrics.md), [data leakage](data-leakage.md), and [model selection](model-selection.md)
- [Matching notebook](../notebooks/35-supervised-learning.ipynb)
- [Interview practice: supervised learning and evaluation](../interview-prep/evaluation-experiment-questions.md)

## References

- Stanford CS229, [Machine Learning Course Notes](https://cs229.stanford.edu/main_notes.pdf).
- Shalev-Shwartz and Ben-David, [Understanding Machine Learning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
