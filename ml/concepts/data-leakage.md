# Data Leakage

**TL;DR:** Data leakage occurs when training or model selection uses information unavailable at the real prediction moment, inflating offline performance and breaking deployment trust.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), timestamps, and [cross-validation](cross-validation-strategies.md). You can define a prediction-time contract, identify target, preprocessing, group, and temporal leakage, audit pipelines, and explain why an offline/online gap may signal leakage. The ML lesson makes leakage a concrete evaluation discipline; the AI overview discusses data lifecycle at a broader level.

## Learning Outcomes

- Explain the central assumptions and mechanism of data leakage.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Leakage is using an answer key while practicing and then claiming the score predicts performance on a real exam. A feature may correlate strongly with the label and still be invalid if it is created after the prediction must be made. Similarly, test-set statistics can quietly influence training even when no test labels are used.

The difficult part is not detecting suspiciously high metrics; it is defining what information exists at the exact prediction timestamp and which entities are genuinely new. Leakage can be accidental and useful-looking, so source lineage matters.

## How It Works

For each prediction, define event time t, allowed feature cutoff t−δ, target window, entity, and serving availability. Target leakage includes post-outcome fields (chargeback status used to predict fraud). Preprocessing leakage occurs when scaling, imputation, feature selection, or vocabulary fitting uses validation/test records. Temporal leakage occurs when future rows or overlapping label horizons cross a split. Group leakage occurs when correlated rows from the same user, patient, or device appear in both train and test, making the task easier than predicting a new entity.

Correct evaluation splits first, then fits all learned transforms on training data only; pipelines automate this. Point-in-time joins must select the latest record available as of each prediction event, not the latest record eventually recorded. Even unsupervised transforms can leak distributional information when the intended claim is generalization to unseen data.

Point-in-time correctness has two clocks in many systems: when an event happened and when the record became available. A lab test may be collected before a prediction timestamp but finalized afterward; joining on event time alone incorrectly exposes the result. Late-arriving logs can also be backfilled into historical aggregates, changing what a replayed training row appears to know. Store both event and ingestion/availability times where possible, define a strict cutoff, and test examples around that boundary. The same principle applies to labels: the label window must end after the prediction, and examples whose outcomes are not yet observed should not be treated as negatives.

Leakage can enter during model selection even when the final model never reads test labels directly. Repeatedly checking a test metric and adjusting features, architecture, or thresholds gradually adapts human decisions to that set. This is a form of selection leakage. Keep the test set locked until all design choices are complete, record any prior exposure, and use validation or nested CV for iterative choices. If the test was used extensively, collect a fresh evaluation cohort before making an unbiased claim.

Target encoding and aggregate features deserve special scrutiny. A category's mean label computed on the full dataset directly contains validation targets; use out-of-fold encodings for training rows and training-only mappings for validation/test. User-level aggregates can leak if they include post-outcome events or information from examples in the held-out fold. When a feature is shared across rows, define the aggregation window and entity boundary explicitly. A high feature importance or an implausibly strong score can help prioritize an audit but is not itself proof of leakage.

Preventive controls can be automated. Add assertions that every feature timestamp is at or before the prediction cutoff, verify disjoint entity IDs across grouped splits, and compare training and serving transformations on the same frozen records. Store feature lineage and snapshot version alongside training data. When offline and online metrics disagree, replay a sample of production predictions using the exact historical feature snapshot; this helps separate leakage from drift, label delay, and implementation skew.

## Trade-offs and Selection Boundary

| Risk | Example | Prevention |
|---|---|---|
| Target leakage | Refund status used to predict refund | Review feature provenance and availability timestamp |
| Preprocessing leakage | Standardizer fit before CV | Put transforms inside each training fold |
| Temporal leakage | Random split of future churn labels | Forward split plus horizon gap/embargo |
| Group leakage | Same patient across train and test | Group-aware split and entity intersection audit |
| Selection leakage | Repeatedly tune against final test | Lock test set; use validation or nested CV |

Use random stratified splits only when IID assumptions match deployment. For users, sites, or future time, split on the corresponding unit. Some transductive tasks legitimately expose unlabeled test inputs; document that assumption and evaluate the exact intended protocol.

## Implementation

```python
from statistics import mean, pstdev

train = [1.0, 2.0, 3.0]
test = [100.0]  # deliberately extreme held-out value
mu, sd = mean(train), pstdev(train)
assert mu == 2.0 and abs(sd-0.81649658) < 1e-6
scaled_test = [(x-mu)/sd for x in test]
assert scaled_test[0] > 100
# If the test value had been used to fit scaling, the training transform would change.
leaky_mu = mean(train+test)
assert leaky_mu != mu
```

## Failure Modes and Diagnostics

1. **Great offline metric, poor launch results:** audit post-event features, split design, and training/serving feature parity; score by time and entity cohorts.
2. **Scaler or target encoder fit globally:** move it into a fold-local pipeline and compare old/new CV scores.
3. **Point-in-time join uses latest eventual state:** replay historical feature snapshots and assert every feature timestamp is ≤ prediction cutoff.
4. **Random split hides entity memorization:** compare random and group-held-out performance and quantify entity overlap.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Define leakage. | Training/evaluation used information unavailable at the actual prediction time or from the intended held-out population. |
| How prevent scaler leakage? | Split first and fit the scaler only on each training fold; transform validation with those fitted values. |
| Debug follow-up: offline AUC is high but online performance drops. | Trace feature availability and point-in-time joins, audit entity/time splits, and check train-serving parity. |

## Related Concepts

- [Cross-validation](cross-validation-strategies.md), [feature engineering](feature-engineering.md), and [evaluation metrics](evaluation-metrics.md)
- [Matching notebook](../notebooks/07-data-leakage.ipynb)
- [Interview practice: point-in-time joins](../interview-prep/sql/02-point-in-time-joins.sql)

## References

- Kaufman et al., [Leakage in Data Mining: Formulation, Detection, and Avoidance](https://doi.org/10.1145/2382577.2382579).
- scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
