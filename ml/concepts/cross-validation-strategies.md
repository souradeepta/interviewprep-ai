# Cross-Validation Strategies

**TL;DR:** Cross-validation estimates how a training procedure generalizes by repeating train/validation splits that respect the data's dependence structure.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), evaluation metrics, and [data leakage](data-leakage.md). You can choose stratified, grouped, or temporal folds; distinguish model selection from final evaluation; and place preprocessing inside each fold. This ML lesson explains validation design; the broader AI overview describes the training lifecycle.

## Learning Outcomes

- Explain the central assumptions and mechanism of cross validation strategies.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A single holdout split may be unusually easy or hard. K-fold cross-validation rotates which part of the available training data is held out, giving several estimates and allowing each observation to contribute to validation once. It is like checking several fair mock exams rather than trusting one draw. The folds are not independent experiments, though: their training sets overlap, so the standard deviation is descriptive rather than a formal confidence interval in many settings.

## How It Works

Partition indices into K disjoint folds F₁…Fₖ. For fold j, fit every learned transformation and the model on D\Fⱼ, then score on Fⱼ. Aggregate fold scores (mean, spread, and per-fold results). Approximate cost is K training runs. Stratification preserves class proportions where possible; group splitting keeps all samples from an entity in one fold; temporal splitting trains on earlier observations and validates on later ones, often with a gap to avoid label-window overlap.

Hyperparameter search uses CV scores repeatedly, so the best score is optimistically biased. Nested CV uses inner folds for selection and outer folds for evaluation, at substantial cost. With enough data and a single prespecified holdout, CV may be unnecessary. The deployment unit and prediction time determine what must not cross fold boundaries.

The number of folds controls a trade-off. More folds train each model on a larger fraction of available examples, which can reduce the mismatch between fold training size and a final model fit, but it multiplies compute and can leave smaller validation folds with noisy metrics. Fewer folds are cheaper and may be adequate with large datasets, but fold composition can matter more. Repeated cross-validation evaluates multiple partitions and can reveal sensitivity to the particular grouping, though it still does not create independent samples from the population. Report fold scores and sample counts rather than only a mean.

Fold construction must happen at the unit of generalization. If five records belong to one patient, patient-level grouping prevents those records from appearing on both sides of a fold. For forecasting with horizon H, the training label window may extend into the validation period; insert a purge or embargo that excludes examples whose outcomes overlap. A random split can still be valid for an IID image benchmark where images are independent, but near-duplicates, subjects, or acquisition sessions may invalidate that assumption. The data-generating process, not the file format, dictates the splitter.

All learned operations must be refit within each training fold. This includes scaling, imputation, target encoding, feature selection, dimensionality reduction, and resampling. A scikit-learn Pipeline makes this composition explicit and reduces accidental fit-on-all-data errors. Unsupervised transforms can also leak information about the held-out distribution if the evaluation claim is strictly inductive. Decide whether deployment is inductive (new observations arrive after training) or transductive (the full unlabeled test batch is available) and evaluate the same scenario.

Cross-validation estimates performance for a training procedure on the folds used. After selection, teams often retrain the chosen configuration on all development data before deployment. This is reasonable, but the final model has seen more data than each CV fold model. The CV estimate is an estimate of the procedure, not a guarantee for that exact final fit. Keep a final test set or external cohort when a high-stakes or public performance claim requires a separate estimate.

## Trade-offs and Selection Boundary

| Split strategy | Use when | Important boundary |
|---|---|---|
| Stratified K-fold | IID classification, especially rare classes | Does not prevent shared-user or temporal leakage |
| Group K-fold | Multiple rows per user, patient, device, or site | Folds may have unequal class balance |
| Forward temporal split | Forecasting future behavior | Fewer training samples; choose realistic horizon and gap |
| Repeated random holdout | Large IID datasets and inexpensive training | Some observations may be evaluated repeatedly or not at all |
| Nested CV | Need estimate of full model-selection procedure | Compute grows roughly with outer × inner folds |

Choose the split from how the model will encounter new examples: new rows, new entities, or future time. Do not default to shuffled K-fold when data has groups or chronology.

## Implementation

```python
def kfold_indices(n, k):
    if not 2 <= k <= n: raise ValueError("require 2 <= k <= n")
    base, extra = divmod(n, k); start = 0
    for fold in range(k):
        size = base + (fold < extra)
        valid = list(range(start, start+size))
        train = list(range(0, start)) + list(range(start+size, n))
        yield train, valid
        start += size

folds = list(kfold_indices(11, 4))
valid = [i for _, v in folds for i in v]
assert sorted(valid) == list(range(11))
assert all(set(tr).isdisjoint(va) and len(tr)+len(va)==11 for tr, va in folds)
# For ordered data, validate that every training index precedes every validation index.
train, test = list(range(6)), list(range(7, 9))
assert max(train) < min(test)  # one-step embargo at index 6
```

## Failure Modes and Diagnostics

1. **Preprocessing before fold split:** validation information affects imputation, scaling, or feature selection. Put the full pipeline inside each fold.
2. **Entity overlap:** the same patient or user appears in train and validation. Audit group intersections explicitly.
3. **Future leakage through labels:** features are time-ordered but labels span a future horizon overlapping the next fold. Define prediction timestamps and add a purge/embargo gap.
4. **Tuning and reporting on the same folds:** selected score is optimistic. Use a final untouched test set or nested CV.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why cross-validate? | It reduces dependence on one arbitrary split and supports model selection when data is limited. |
| When use GroupKFold? | When deployment involves unseen entities and rows from the same entity are correlated. |
| Design follow-up: predict next-month churn per user. | Split by time and ensure users/label windows cannot leak across folds; use a realistic forward horizon. |

## Related Concepts

- [Data leakage](data-leakage.md), [model selection](model-selection.md), and [hyperparameter tuning](hyperparameter-tuning.md)
- [Matching notebook](../notebooks/06-cross-validation.ipynb)
- [Interview practice: cross-validation](../interview-prep/ml-theory-questions.md#q-how-does-cross-validation-work-and-when-do-you-use-it)

## References

- scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html).
- Roberts et al., [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881).
