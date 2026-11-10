# Project 01: Tabular ML in Production

**Timebox:** 8 hours. **Role path:** Entry-level ML engineer. **Prerequisites:** [Supervised learning](../ml/concepts/supervised-learning.md), [evaluation metrics](../ml/concepts/evaluation-metrics.md), [data leakage](../ml/concepts/data-leakage.md), [logistic regression](../ai/concepts/07-logistic-regression.md).

## Problem brief

Build a small purchase-propensity model that could be reviewed for release. The emphasis is not leaderboard performance: demonstrate a leakage-safe split, compare against a simple baseline, calibrate or explicitly reject calibration when evidence is insufficient, and define an operational promotion gate.

## Data contract

Create deterministic synthetic rows with `customer_id`, `event_day`, numeric behavior features, a categorical region, and binary `purchased_within_7d`. Include a deliberately post-outcome field named `receipt_status` for leakage tests. Use a fixed seed; do not include real personal data or external downloads. Keep train/validation/test periods separate by time and customer as appropriate.

## Deliverables

1. A runnable Python module or notebook with data generation, validation, baseline, candidate model, and evaluation.
2. A short model card describing target, intended use, split, metrics, known failure modes, and threshold choice.
3. A promotion checklist with one data-contract gate, one quality gate, and one rollback trigger.
4. A five-minute interview debrief: why the split is valid, why the metric fits, and what evidence would change the release decision.

## Local success criterion

From the project directory, `python3 solution.py` exits zero, asserts that `receipt_status` is excluded from model features, asserts test rows occur after training rows, and prints baseline and candidate metrics on the same held-out test set. Use only Python standard library unless an included environment dependency is documented.

## Evaluation rubric — 100 points

| Dimension | Points | Evidence |
|---|---:|---|
| Engineering and reproducibility | 25 | Fixed-seed synthetic data, clear validation, rerunnable pipeline, no target leakage. |
| Measurement and model decision | 25 | Baseline comparison, suitable metrics, threshold reasoning, uncertainty or slice caveat. |
| Failure handling | 20 | Tests for malformed/missing fields, temporal leakage, imbalance, and unsupported calibration. |
| Communication | 15 | Model card and concise release recommendation with explicit limitations. |
| Interview debrief | 15 | Explains alternative, failure mode, and evidence required before production. |

## Extension prompt

Add a simulated delayed-label monitor and define a canary policy that separates immediate service/data alerts from later model-quality alerts.
