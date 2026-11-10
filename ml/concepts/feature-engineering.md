# Feature Engineering

**TL;DR:** Feature engineering turns raw, time-available data into representations that make the target signal learnable while preserving a reproducible train-serving contract.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), basic statistics, and [data leakage](data-leakage.md). You can build numeric, categorical, and temporal features; choose transformations by model family; prevent point-in-time leakage; and test feature parity between training and inference. This ML lesson focuses on representation and data contracts; the AI track provides broad application context.

## Learning Outcomes

- Explain the central assumptions and mechanism of feature engineering.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A raw timestamp such as “2026-09-22 08:17” is not necessarily useful as a number. The model may learn more from hour of day, day of week, or elapsed time since a prior event. A feature is a hypothesis about what information matters. Good features expose stable structure; bad features encode accidental correlations, unavailable future information, or brittle identifiers.

Deep models learn representations from data, but engineered features remain valuable for tabular tasks, limited datasets, domain constraints, and latency-sensitive serving. Feature engineering is part of the model and must be versioned with it.

## How It Works

Numeric features may be imputed, scaled, clipped, or transformed (e.g., log1p for a long-tailed nonnegative count). Categorical features may be one-hot encoded, frequency encoded, or represented by learned embeddings. Missingness can carry signal, but missing indicators should be validated across deployment populations. Time features include calendar cycles, lags, rolling aggregates, and event counts; rolling values must be computed only from observations available at the prediction time.

Linear and distance-based models are scale-sensitive; tree splits are generally invariant to monotonic scaling but still depend on sensible missing/category handling. Fit learned transformations on training folds and serialize the exact schema, category vocabulary, and order. For online systems, point-in-time joins and feature freshness define correctness as much as model weights do.

Feature engineering is best treated as a versioned transformation graph. Each output feature should have a name, type, units, source fields, aggregation window, missing-value policy, and availability time. This metadata helps reviewers answer whether the feature exists at prediction time and helps the serving pipeline reproduce its training meaning. A column called “recent_spend” is not a complete definition: the entity key, lookback duration, timezone, late-arrival behavior, and inclusion of refunds all affect its value. Versioning those choices prevents a seemingly harmless upstream change from silently changing model inputs.

For temporal aggregates, define the as-of boundary precisely. A rolling seven-day count for an event at time t may use events in the interval from t minus seven days up to but excluding t, but the right endpoint can be inclusive or exclusive depending on event ordering and serving latency. If late-arriving events are backfilled, historical training features may contain information unavailable online unless availability time is respected. Feature stores can enforce point-in-time joins, but their API does not guarantee correctness unless timestamps and keys are modeled correctly. Validate boundary cases with hand-calculated examples and compare training-time snapshots to the values computed by serving code.

Categorical features require an explicit unseen-value strategy. One-hot encoding can route unknown categories to all zeros or a dedicated unknown column; frequency encoding can assign an unseen frequency; learned embeddings need a fallback ID. Target encoding can be powerful but must be computed out-of-fold for training examples, smoothed for rare categories, and fit only on training labels. Otherwise each example's label leaks directly into its encoded feature. Cardinality reduction by grouping rare values can improve reliability, but the grouping threshold should be selected without peeking at final test outcomes.

Feature selection should consider data collection and fairness. An identifier may look predictive because the train/test split contains repeated entities but fail on new entities. A proxy feature may encode a sensitive attribute even if that attribute is removed. Evaluate performance and missingness across relevant subgroups and ask whether the feature is legally and operationally permissible. Feature importance is not causal evidence: a feature can be predictive without being an appropriate intervention target.

Feature design begins with the prediction contract: what is known at decision time, what entity receives a prediction, and how often the schema changes. A feature can be statistically predictive and still be unusable because it arrives after the action, is expensive to compute, or encodes an identifier that disappears for new users. Write down event time and availability time separately. For example, a transaction that occurred before midnight but arrived the next morning cannot be included in a prediction issued at midnight. Backtests should reconstruct what the system could actually have observed then.

Transformations should be fitted within the training partition. Means, scales, category vocabularies, imputers, and feature selectors all learn from data; fitting them before a split leaks information even if labels are not used. Put preprocessing and estimator in one pipeline so every cross-validation fold gets its own fitted transform. For temporal prediction, split by time and ensure rolling windows end strictly before each target timestamp. For repeated users, split by user when the intended deployment includes unseen users. The split rule is part of the feature definition because it determines what “available” means.

Missingness can itself carry signal, but it must be interpreted carefully. Add an indicator when a missing value reflects a stable collection process, then fit imputation values on training data only. A model can otherwise infer operational changes such as a sensor outage and perform poorly when that process changes. Tree libraries differ in their handling of missing values; test the chosen implementation and record the behavior. For numeric variables, scaling is usually important for distance-based methods, linear models with regularization, and neural networks, while many tree ensembles are largely invariant to monotonic rescaling. Scale only when it serves the model or numerical stability.

Feature interactions are another selection decision. A linear model needs explicit products or basis expansions to represent interactions; a tree can learn threshold interactions but may need enough data and depth. High-cardinality categories may call for hashing, embeddings, or out-of-fold target encoding. Hashing controls dimensionality but introduces collisions. Target encoding reduces width but requires fold-aware computation and smoothing, especially for rare categories. One-hot encoding is transparent for low-cardinality variables, though unknown categories need a defined fallback. Compare these choices with a validation split that resembles deployment.

Operationally, define a feature registry or schema with name, type, units, null policy, owner, freshness, and transformation version. Validate ranges and category rates at training and serving boundaries. Golden-record tests should send identical records through offline and online code and compare values within an explicit tolerance. Track freshness and missingness as well as model metrics; a feature pipeline may silently degrade before labels arrive. Feature importance is a diagnostic of model reliance, not evidence that changing a feature causes an outcome. Review sensitive proxies and retention constraints before launching a feature.

Feature selection should account for latency and reliability as well as offline lift. A rolling aggregate computed with a costly join can be valuable in a batch scoring system and impossible on a strict low-latency path. Prefer a simpler feature when gains are within uncertainty or the feature cannot be reproduced consistently. Measure ablations on the same folds, inspect subgroup and time-slice results, and retain a feature only when its incremental value survives the intended serving contract.

## Trade-offs and Selection Boundary

| Transformation | Useful when | Risk |
|---|---|---|
| Standardization | Linear, neural, and distance-based models need comparable scales | Outliers distort mean/std; fit on training only |
| One-hot encoding | Low-cardinality categories with no natural order | High cardinality expands width and unseen categories need policy |
| Log transform | Positive heavy-tailed quantities | Cannot handle negatives without a deliberate alternative |
| Rolling aggregates | Behavior history predicts future outcomes | Window boundaries and late events can leak future data |
| Target encoding | High-cardinality categories | Severe target leakage without out-of-fold smoothing |

Prefer simple domain-grounded features before complex transformations. Avoid hand-crafted features when they cannot be computed reliably at serving time or are fragile under drift. For tree ensembles on tabular data, test whether scaling adds value instead of assuming it does.

## Implementation

```python
from math import log1p

events = [(1, 10.0), (3, 4.0), (5, 7.0), (8, 2.0)]  # (day, amount)
cutoff = 6
past = [(day, amount) for day, amount in events if day < cutoff]
features = {
    "event_count_5d": sum(cutoff-5 <= day < cutoff for day,_ in past),
    "spend_5d_log": log1p(sum(amount for day,amount in past if cutoff-5 <= day < cutoff)),
}
assert features == {"event_count_5d": 2, "spend_5d_log": log1p(11.0)}
assert all(day < cutoff for day,_ in past)  # point-in-time boundary
```

## Failure Modes and Diagnostics

1. **Future information in aggregates:** same-day late events enter an earlier prediction. Recompute features with strict event/availability-time cutoffs and test boundary timestamps.
2. **Train-serving skew:** training uses cleaned batch data while serving handles nulls/categories differently. Run parity tests on identical records through both transformations.
3. **Unseen categories or schema reorder:** silently map categories incorrectly. Version schema/vocabulary and define unknown-category behavior explicitly.
4. **Target encoding overfit:** category means reveal fold labels. Use smoothed out-of-fold encoding and validate with a separate temporal/group split.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why engineer features for trees? | Trees learn nonlinear thresholds and interactions, but representation, missingness, time boundaries, and cardinality still matter. |
| How avoid leakage in rolling features? | Compute each aggregate only from events available before that example's prediction cutoff. |
| Design follow-up: training AUC is high but online features differ. | Compare identical records through training and serving pipelines, schema versions, null handling, and point-in-time joins. |

## Related Concepts

- [Data leakage](data-leakage.md), [evaluation metrics](evaluation-metrics.md), and [model selection](model-selection.md)
- [Matching notebook](../notebooks/14-feature-engineering.ipynb)
- [Interview practice: feature aggregation SQL](../interview-prep/sql/01-feature-aggregations.sql)

## References

- scikit-learn, [Common preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- Google Cloud, [Rules of ML: Training-serving skew](https://developers.google.com/machine-learning/guides/rules-of-ml).
