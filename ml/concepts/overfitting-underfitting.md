# Overfitting and Underfitting

**TL;DR:** Underfitting means a model fails to capture useful structure even on training data; overfitting means training success does not generalize to new examples.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), train/validation/test splits, and evaluation metrics. You can diagnose fit from learning curves, distinguish model capacity from data problems, choose interventions for high bias or variance, and explain why validation design matters. This ML lesson gives diagnostic methods; the AI track discusses model training more broadly.

## Learning Outcomes

- Explain the central assumptions and mechanism of overfitting underfitting.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A student who cannot solve practice problems has not learned enough of the underlying material: this resembles underfitting. A student who memorizes the practice answer sheet but fails a new exam resembles overfitting. Neither diagnosis is based solely on whether the model is “large.” It comes from training performance, validation performance, data quality, metric choice, and the distribution of future examples.

The familiar bias-variance picture is useful but simplified. Real systems can fail because labels are noisy, features are unavailable, validation is shifted, or the objective does not match deployment. A widening training-validation gap is evidence to investigate, not proof that regularization is the only remedy.

## How It Works

Empirical risk minimizes average training loss over available samples. A low-capacity model may have high approximation error: it cannot represent the target relation even with abundant data. A high-capacity model can fit sampling noise, giving low training loss but high variance across datasets. More representative data often reduces variance; changing features or model class may reduce approximation error. Irreducible label noise places a floor on achievable prediction error.

Plot training and validation metrics against examples seen, model capacity, or training updates. High training and validation error that are close suggests underfitting, optimization failure, or irreducible noise. Low training error with substantially worse validation suggests overfitting, leakage issues, or a train-validation distribution mismatch. Learning curves against dataset size help distinguish whether collecting more data may close the gap. The diagnosis should be made on correctly separated data: a leaked validation score can conceal both behaviors.

Regularization constrains effective capacity through penalties, dropout, augmentation, pruning, or early stopping. These choices add bias in exchange for lower variance. More training data helps only when it is representative and correctly labeled. Increasing model size may help underfitting but can worsen overfitting if data and regularization do not support it. Tuning many choices against one validation set can itself overfit that set.

Learning curves provide a practical diagnostic. Plot performance as a function of training examples and optimization steps, with training and validation values measured under the same metric where appropriate. If validation performance continues improving as more training examples are added and the gap narrows, acquiring additional representative data may be valuable. If both curves plateau at a poor level with little gap, new features, a more suitable objective, or better optimization may be more promising. If training improves while validation worsens, compare regularization and capacity, but first rule out temporal or population mismatch. Curve shapes suggest experiments; they do not identify a cause without further checks.

There is also a distinction between epistemic uncertainty (lack of knowledge because data or model evidence is limited) and aleatoric variation (inherent variability in outcomes). Collecting more data can reduce uncertainty about the first, but it cannot make inherently random outcomes perfectly predictable. A model may achieve a stable score below 100 percent for this reason. Identify a realistic baseline and assess whether remaining errors are systematic, actionable, and represented in the target metric before increasing complexity.

Finally, apparent overfitting can be created by evaluation mistakes. A tiny validation set has high variance; a shifted validation set can exaggerate or hide a gap; repeated manual experiments adapt to validation labels. A robust diagnosis compares folds, cohorts, and time periods, reports support counts, and preserves a final test set. This protects against responding to sampling noise with unnecessary simplification or data collection.

## Trade-offs and Selection Boundary

| Evidence | Likely issue | First interventions |
|---|---|---|
| High train and validation error, small gap | Underfitting or optimization failure | Check data/labels and features; improve capacity or train longer if optimization is incomplete |
| Low train error, high validation error | Overfitting or distribution mismatch | Add representative data, regularize, simplify, or improve split realism |
| Both metrics poor and noisy across seeds | Data or optimization instability | Audit labels, learning rate, split size, and randomness |
| Validation strong, production weak | Leakage or shift | Audit prediction-time features and time/entity distribution |

Use a regularization intervention when training fits well and validation lags on a representative split. Increase capacity when both train and validation remain poor and the optimization process is healthy. Avoid using a single gap threshold across unrelated datasets; compare against baselines and uncertainty.

## Implementation

```python
def diagnose(train_score, valid_score, high_score=0.8, gap=0.15):
    if train_score < high_score:
        return "underfit_or_optimization"
    if train_score-valid_score > gap:
        return "overfit_or_shift"
    return "no_large_gap"

assert diagnose(.55, .50) == "underfit_or_optimization"
assert diagnose(.98, .70) == "overfit_or_shift"
assert diagnose(.91, .88) == "no_large_gap"
```

## Failure Modes and Diagnostics

1. **Treating every train-validation gap as overfitting:** the validation population may differ. Compare feature distributions and time/entity cohorts, and use a deployment-representative split.
2. **Training score is low because optimization failed:** model capacity is increased unnecessarily. Inspect loss curves, gradient norms, and a tiny-batch overfit test first.
3. **Regularization added to already-underfit model:** both train and validation worsen. Reduce penalties or increase capacity after confirming data quality.
4. **More data is duplicated or mislabeled:** nominal dataset size rises without adding signal. Audit entity duplication, annotation quality, and coverage of important slices.
5. **Validation set repeatedly tuned:** decisions leak through human iteration. Reserve a final test set and document the number of adaptive choices made against validation.

## Interview Quick Reference

| Question | Answer |
|---|---|
| How diagnose underfitting versus overfitting? | Compare train and representative validation curves; high error on both suggests underfit/optimization issues, a large gap suggests overfit or shift. |
| Does more data help? | Representative, correctly labeled data often reduces variance, but does not fix wrong features, optimization, or distribution mismatch. |
| Debug follow-up: train score is excellent but production drops. | Audit leakage, feature availability, and temporal/entity shift before simply adding regularization. |

## Related Concepts

- [Regularization](regularization.md), [early stopping](early-stopping.md), and [model selection](model-selection.md)
- [Matching notebook](../notebooks/29-overfitting.ipynb)
- [Interview practice: bias-variance trade-off](../interview-prep/ml-theory-questions.md#q-what-is-the-bias-variance-trade-off)

## References

- Goodfellow, Bengio, and Courville, [Deep Learning, Chapter 5: Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html).
- Domingos, [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf).
