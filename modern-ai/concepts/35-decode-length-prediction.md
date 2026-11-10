# Decode-Length Prediction

**TL;DR:** Predicting output length can support batching and cache reservation, but underprediction creates truncation or queue pressure; evaluate calibration and tail risk, not only mean error.

## Prerequisites

[KV cache optimization](29-kv-cache-optimization.md), regression and quantiles.

## Learning Outcomes

- Calibrate decode-length predictions against held-out outcomes.
- Compute underprediction rate and a tail reservation from a sample.
- Compare mean error with quantile-aware resource allocation.
- Explain when a scheduler should reserve conservatively or defer admission.

## Core Intuition

A venue planner does not book seats for the average number of guests when a full room has a large cost. It considers uncertainty and the cost of running short. Likewise, a generation service may use predicted output length to pack batches or reserve KV blocks. Over-reservation wastes memory; under-reservation can force eviction, preemption or truncation. The right threshold depends on the cost of each error.

## How It Works

Features can include prompt length, task type, retrieval count, user-selected maximum and historical behavior, subject to privacy and leakage rules. A point regressor estimates expected length, while quantile regression estimates a conditional quantile such as Q_0.95. Calibration asks whether events assigned 95% coverage occur about 95% of the time on held-out, representative traffic. Isotonic regression can monotonically map scores to observed means or probabilities, but requires enough representative examples and periodic recalibration under drift.

Useful metrics include MAE, pinball loss for quantiles, underprediction rate, reserved-to-used token ratio, OOM/preemption rate and p95/p99 latency. Average MAE hides rare very long outputs. A maximum-output cap remains a separate guardrail. If the predictor is uncertain or out of distribution, reserve a conservative amount or defer admission rather than trusting a brittle point estimate.

The target definition must match the scheduler's decision. Predicting the final number of generated tokens before decoding is different from predicting remaining tokens after a partial output. For admission control, the feature vector can include prompt tokens, route or task class, requested maximum and model identity. For mid-generation growth, recent output pace and stop-token evidence become available. Maintain separate evaluations for each decision point and ensure that every feature was known at the moment the reservation was made.

Length distributions are often skewed and multimodal. A mixture of short answers, code blocks and long explanations can make a single conditional mean representative of none of them. Buckets by task or route can improve calibration, but too many buckets leave little data per estimate. Quantile regression directly targets an asymmetric reserve, while conformal methods can provide marginal coverage under exchangeability assumptions. Neither removes the need to check coverage across important slices or after distribution shift.

Underprediction and overprediction have different consequences. Underprediction can force cache growth, interrupt batching, trigger preemption or truncate output if hard limits are reached. Overprediction holds blocks unused and can reduce admission capacity for other requests. Set the reservation quantile by comparing these costs and the latency objective. For one workload, a P90 reserve may be acceptable; for a workload where truncation is catastrophic, a more conservative policy may be justified. Make the decision explicit rather than defaulting to the average.

Calibration must be monitored after deployment. Prompt templates, model releases, user behavior and product features alter length distributions. Track observed-versus-predicted plots, exceedance rate, calibration error, reserve utilization and queue delay over time. Slice by task class and output cap. Retrain only with examples sampled under the current logging policy and protect sensitive prompt content. A drift detector should trigger investigation; automatic recalibration on biased or incomplete logs can make the estimate worse.

Prediction should degrade safely. If the input is out of distribution, if confidence is low, or if the requested maximum is absent, use a conservative default and enforce a hard generation limit. Dynamic block growth can reduce waste but needs atomic allocation and a clear full-pool response. Scheduling should avoid letting a long, repeatedly growing request monopolize capacity. Evaluate not only prediction accuracy but also service-level outcomes: admitted requests, completion rate, truncation, fairness and p99 latency.

## Trade-offs

| Policy | Strength | Boundary |
|---|---|---|
| Mean prediction | Low average squared error target | Can systematically under-reserve tails |
| High quantile | Controls a chosen exceedance rate | More memory stranded per request |
| Length buckets | Simple scheduler interface | Coarse boundaries waste capacity |
| Dynamic growth | Reserves incrementally | Needs safe block acquisition and can cause queue stalls |

## Implementation with Python

Run `python3 modern-ai/implementations/35-decode-length-prediction.py`. A fixed trace shows monotone calibration pooling, underprediction rate, absolute error and a p95 reserve. The sample is intentionally tiny and cannot establish a production percentile.

```python
import runpy
m = runpy.run_path("modern-ai/implementations/35-decode-length-prediction.py")
underprediction_rate = m.get("underprediction_rate")
assert underprediction_rate(list((3, 5)), list((4, 4))) == .5
```

## Failure Modes

| Failure | Diagnostic | Mitigation |
|---|---|---|
| Prompt class changes over time | Calibration error by time and task slice | Monitor drift and retrain/recalibrate with recent data |
| Rare long outputs are hidden by average | Exceedance rate and tail latency | Use quantile loss and enforce an output cap |
| Leakage from generated answer features | Feature timestamp audit | Use only information available before scheduling |

- Calibration drifts by request type: measure exceedance and reliability per route, then update the model using representative recent data.
- Rare long generations exceed capacity: track tail coverage and apply a quantile reserve plus a hard output cap.
- Future information leaks into prediction: audit feature timestamps and remove any value unavailable at admission time.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| Why is MAE insufficient for cache reservation? | It does not capture asymmetric cost or tail exceedance; ask which quantile the SLA requires. |
| How do you validate a P95 length estimate? | Held-out calibration coverage and slice-level exceedance; follow up on drift. |
| What if prediction is uncertain? | Reserve conservatively or dynamically grow/limit; ask how this affects admission fairness. |

## Related Concepts

See [KV cache](29-kv-cache-optimization.md), [dynamic batching](47-dynamic-batching.md), [notebook 35](../notebooks/35-decode-length-prediction.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#decode-length-prediction).

Prediction should be evaluated as part of the serving policy, not in isolation. A more accurate length estimate may still reduce throughput if it reserves too many blocks early and prevents other requests from entering a batch. Conversely, a modest underprediction rate can be acceptable if cache growth is cheap and requests complete without preemption. Replay an arrival trace through the scheduler using actual lengths and predicted reservations; compare completion rate, queueing, unused memory and tail latency to expose the operational effect.

## References

[Conformal Prediction](https://arxiv.org/abs/2107.07511) and [A Tutorial on Conformal Prediction](https://arxiv.org/abs/0706.3188).
