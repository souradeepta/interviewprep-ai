# Project 03: ML Platform Reliability

**Timebox:** 8 hours. **Role path:** Production MLE. **Prerequisites:** [Data pipelines](../mlops/concepts/01-data-pipelines.md), [data validation](../mlops/concepts/03-data-validation.md), [reproducibility](../mlops/concepts/07-reproducibility.md), [model testing](../mlops/concepts/09-model-testing.md), [monitoring and observability](../system-design/patterns/16-monitoring-and-observability.md).

## Problem brief

Create a local batch prediction pipeline with versioned input contract, deterministic baseline model, monitoring summary, and incident response for a feature outage. Show how a job can complete successfully yet still violate data-quality expectations.

## Data contract

Generate a fixed-seed synthetic CSV-like dataset in memory or a checked-in tiny fixture with event timestamp, entity ID, two numeric features, and label. Include a second batch with a missing-feature spike, stale timestamps, and an invalid schema. Never use production data, credentials, or external services.

## Deliverables

1. A batch validation and prediction pipeline with explicit model/data version identifiers.
2. A health report showing row counts, missingness, freshness, prediction distribution, and delayed-label metric placeholder.
3. An incident runbook: detect, contain, diagnose, recover, and verify.
4. A release/rollback plan that explains compatibility between model and feature versions.

## Local success criterion

`python3 solution.py` exits zero, accepts the valid batch, rejects or quarantines the injected bad batch with a reason, flags the missing-feature threshold breach, and demonstrates that the prior approved model version remains selectable. Print the audit summary.

## Evaluation rubric — 100 points

| Dimension | Points | Evidence |
|---|---:|---|
| Engineering and reproducibility | 25 | Versioned inputs/artifacts, deterministic run, explicit contracts and failure output. |
| Measurement and observability | 20 | Useful denominators, freshness/missingness and distribution summaries, delayed label caveat. |
| Failure handling and recovery | 25 | Quarantine, containment, tested previous-version selection, recovery verification. |
| Communication | 15 | Clear incident timeline, ownership, limits, and operational decision. |
| Interview debrief | 15 | Explains SLO trade-offs, rollback compatibility, and next reliability test. |

## Extension prompt

Add an idempotent replay key and simulate retrying the same batch twice. Explain the transaction boundary required to make a real replay safe.
