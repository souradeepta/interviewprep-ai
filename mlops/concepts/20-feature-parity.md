# Online and Offline Feature Parity

**TL;DR:** Training and serving must compute semantically identical features from information available at prediction time; parity audits should measure value mismatch and freshness separately.

## Prerequisites

[Feature stores](02-feature-stores.md), [data validation](03-data-validation.md), and event-time joins.

## Learning Outcomes

- Explain training-serving skew, point-in-time correctness and feature freshness.
- Compare offline and online values using a numeric tolerance and explicit null policy.
- Define a parity sample, mismatch-rate threshold and response before model rollout.
- Diagnose event-time, default-value, transformation-version and materialization causes.

## Core Intuition

A model is trained on a contract about what each input means. If training sees “purchases in the preceding seven days” but serving sees “purchases so far today,” the same column name hides different semantics. The model can be perfectly valid and still fail because it receives a different feature function online. Offline/online parity is like checking that two thermometers use the same units, calibration and timestamp—not merely that both return a number.

## How It Works

Training examples need point-in-time features: values that were available at the prediction timestamp, not the latest values currently in the warehouse. For entity e and prediction time t, a feature should be derived only from events with event time and availability time no later than the allowed cutoff. Using future state leaks information into training and inflates offline metrics. Event time alone may be insufficient when data arrives late; retain ingestion/availability time when the production system only knew the value after that point.

Online serving adds freshness and materialization concerns. A feature can match the offline value for yesterday and still be stale for the current request. Track event timestamp, materialization time and observed read time. Define a freshness SLO per feature class; the demonstration policy blocks parity when a value is more than 300 seconds old or has a negative age caused by clock/data inconsistency. The threshold is configurable and should reflect feature update cadence and business impact.

Parity tests need stable examples. Sample entity/time pairs and log the offline value, online value, feature-definition version, source offset and timestamps. For numeric values, compare absolute or relative tolerance appropriate to the units; for categorical and missing values, compare exact normalized representations. Treat null versus default zero explicitly. A default may be a legitimate contract value, but silent divergence is not parity. Report mismatch rate by feature and meaningful segment, not just one aggregate pass/fail.

The implementation compares a single value and calculates freshness from ISO timestamps, then aggregates mismatch and stale rates across rows. Its operational decision blocks when either rate exceeds 1%. That policy is intentionally conservative for a small teaching sample: production use should require enough examples, compare against a versioned threshold, and perhaps block model promotion only for critical features. For low-severity features, open an alert while preserving rollout controls.

Use one shared transformation library where feasible, but do not assume code reuse guarantees semantic parity. SQL and Python may differ in null handling, timezone conversion, integer overflow, floating-point rounding and window boundaries. Validate a set of golden examples through both paths. Compare edge cases: entity with no history, late event, duplicate event, daylight-saving boundary, future timestamp, missing category and value exactly at a threshold. Pin transformation versions in the model artifact and monitor parity after changes.

Parity is also a rollout concern. Before deploying a model trained with a new feature definition, dual-compute or shadow-read the new values, quantify mismatch and stale rates, and validate online latency. During migration, support both old and new feature schemas or provide an adapter; otherwise rollback may pair an old model with incompatible features. Keep the feature contract, model version and serving transformation version tied together in release metadata.

Parity sampling should cover both frequent and high-risk entities. Uniform random samples are useful for an overall estimate, but they can miss rare cohorts such as new users, high-value accounts or regions with delayed ingestion. Stratify samples while reporting the sampling method so rates remain interpretable. If the audit changes logging or reads the online store, bound the added request volume and avoid exposing raw sensitive values. Hash entity identifiers and compare values in a controlled environment with limited retention.

Feature transformations need explicit semantic metadata: units, timezone, aggregation window, null/default policy, categorical vocabulary, and whether the value is based on event or processing time. For example, “transactions in seven days” is ambiguous unless the window endpoint and late-event rule are specified. Add boundary tests for exactly-at-cutoff timestamps and duplicate event IDs. A model registry should retain the feature schema identifier so inference can reject an incompatible feature payload instead of producing a plausible but invalid score.

Thresholds should reflect feature criticality. A 1% mismatch may be unacceptable for a feature that directly gates a safety decision, while a small difference in a low-impact approximate aggregate may be within numerical tolerance. Define a maximum mismatch rate and freshness SLO per feature class, require a minimum sample, and block only when the evidence is sufficient. Track consecutive failing windows for noisy sources, but do not wait through a persistence window if a hard schema or privacy violation is detected.

When parity fails, preserve representative examples and the full transformation lineage before fixing the pipeline. Compare source records, timestamp cutoffs, intermediate values, defaults and serialization at each stage. A single aggregate metric cannot locate the first divergence. Once corrected, replay the same examples through both paths, backfill affected offline training data if needed, and validate the candidate model against a clean time-based holdout. This closes the loop between feature correctness and model quality.

## Trade-offs

| Strategy | Advantage | Limitation |
|---|---|---|
| Shared transformation code | Reduces duplicated logic | Runtime and data-source differences remain |
| Offline replay | Reproducible comparisons at scale | May not reproduce online timing and cache state |
| Online shadow reads | Measures real serving values | Adds traffic, cost and privacy/latency constraints |
| Exact equality | Simple for categorical/discrete features | Too strict for floating-point calculations |
| Tolerance-based check | Handles numeric rounding | Can hide material errors if tolerance is too wide |

## Implementation with Python

Run `python3 mlops/implementations/20-feature-parity.py`. The implementation compares value equality under a numeric tolerance, calculates online feature age, and blocks an audit when mismatch or stale rates exceed 1%. It uses local timestamps and synthetic rows only.

```python
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("parity", "mlops/implementations/20-feature-parity.py")
parity = module_from_spec(spec); spec.loader.exec_module(parity)
report = parity.audit_rows([{"offline": {"value": 2}, "online": {
    "value": 2, "event_time": "2026-01-01T00:00:00", "observed_at": "2026-01-01T00:01:00"}}])
assert report["decision"] == "pass"
```

## Failure Modes

- Future values leak into offline training: enforce as-of joins on event and availability time.
- Defaults diverge online and offline: test null, absent and zero as distinct contract cases.
- Feature is numerically equal but stale: track event age separately from value parity and enforce the freshness SLO.
- A migration breaks rollback: version feature schemas and model compatibility as one release unit.

## Interview Quick Reference

| Question | Expected answer | Follow-up |
|---|---|---|
| Why can offline AUC be high while online quality falls? | Training-serving skew, future leakage, stale or differently defaulted online features. | Debug with a traced entity/timestamp example. |
| How do you validate parity before launch? | Compare sampled online/offline values, versions, missingness and freshness against explicit thresholds. | Design a canary gate for a critical feature. |
| Is a shared feature function sufficient? | It helps but source, time, null and runtime semantics can still differ. | Design edge cases for a seven-day rolling count. |

## Related Concepts

See [feature stores](02-feature-stores.md), [data validation](03-data-validation.md), [data testing](10-data-testing.md), and [MLOps interview practice](../interview-prep/mlops-interview-questions.md#data-contracts-and-feature-reliability).

## References

[Feast: Point-in-Time Joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) and [Google Cloud: Training-Serving Skew](https://cloud.google.com/architecture/monitoring-ml-models#training-serving_skew).
