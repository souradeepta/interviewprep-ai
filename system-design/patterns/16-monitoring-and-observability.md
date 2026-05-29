# Monitoring and Observability

## TL;DR
Track: prediction distribution, input feature distribution, latency, errors, business metrics. Alert when drift detected or latency spikes. Observability = logs + metrics + traces.

## Core Intuition
Blind deployment: ship model, hope it's good. Observable: see what model does, when it's failing, why.

## How It Works

**Three pillars:**
1. **Metrics:** latency p50/p99, prediction distribution, error rate
2. **Logs:** detailed trace of each prediction (for debugging)
3. **Traces:** end-to-end request path (feature extraction → inference → response)

| Pillar | Example | Purpose |
|--------|---------|---------|
| Metrics | p99 latency 150ms | Catch performance degradation |
| Logs | "user_id=123, feature_age=35, prediction=0.92" | Debug specific predictions |
| Traces | Request took 10ms feature + 50ms inference | Find bottleneck |

## Key Properties / Trade-offs
- Cost: continuous monitoring is expensive (storage, processing)
- Latency: adding observability adds overhead
- Coverage: sample all traffic or 10%?

## Detailed Trade-off Analysis

| Aspect | Sample 100% | Sample 10% | Sample 1% | ML-Driven Sampling |
|--------|------------|-----------|----------|-------------------|
| Storage cost | $5K/month | $500/month | $50/month | $300/month |
| Detection latency | Immediate | ~1 hour | ~10 hours | ~30 min (adaptive) |
| False positive rate | Low | Medium | High | Low |
| Coverage on rare events | Complete | 10% | 1% (miss) | Targeted |

**Decision:** Critical path 100%, high-volume sampled 1-10%, rare events ML-routed.

---

## Production Failure Scenarios

**Scenario 1: Alert fatigue from overly sensitive thresholds**
- Alert on latency >150ms. Triggers 100x/day. Team ignores all alerts.
- Fix: Set baseline from production data. Alert on >2σ deviation (p99+30%). Validate before deploying.

**Scenario 2: Missing the failure in noise**
- Error rate 0.5% baseline. Spikes to 0.6%. Below alert threshold. Silently serving wrong predictions.
- Fix: Track both absolute (>1%) AND relative (>2x baseline) metrics. Catch both sudden and gradual failures.

**Scenario 3: Logs growing unbounded**
- Log all predictions (1K QPS × 24h = 86M events/day). Disk full in 2 days. No monitoring.
- Fix: Implement sampling + tiering (hot logs 7 days, cold 30 days). Or structured logging with smart retention.

**Scenario 4: Silent failure in batch prediction**
- Batch job produces 0 predictions. No error logged. Model stopped working undetected for 2 days.
- Fix: Monitor pipeline completion (did job finish?), output shape (was output produced?), latency (did job hang?).

---

## Implementation Guidance

**Wrong:** Log all predictions (storage explosion). Alert on single prediction >0.95 (noise).
**Right:** Sample 10% with stratification (ensure rare classes captured). Alert on distribution shift (KS-test) not single anomalies.

**Wrong:** Monitor only latency. Ignore accuracy.
**Right:** Monitor 3 pillars: (1) system metrics (latency, error rate), (2) data metrics (input/output distribution), (3) business metrics (revenue, engagement).

---

## Sophisticated Interview Q&A

**Q1: Latency p99 usually 100ms, now 200ms. Problem?**
A: Not necessarily. If p50 (median) unchanged, p99 spike could be one slow request. Check: (1) is p50 affected? (2) is it persistent (20+ min) or transient? (3) what changed (code, data, load)? If p50 stable and transient, likely noise.

**Q2: Monitoring 1M predictions/day. Can't log all. What to do?**
A: Stratified sampling: (1) log 100% of errors, 10% of correct predictions. (2) Log 100% of rare classes, 1% of common. (3) Use ML to detect anomalies—log only anomalies. Total storage: 50K-100K events/day instead of 1M.

**Q3: Alert tuning: how many false alarms acceptable?**
A: Depends on response cost. If alert → engineer pages = expensive. Aim for <1 false positive per week per alert. If alert → automated rollback, tolerate 5-10% false positive rate.

**Q4: How distinguish between data drift vs model degradation?**
A: (1) Input distribution changed? = Data drift. (2) Input same, prediction changed? = Model drift. (3) Prediction same, but accuracy dropped? = Ground truth drift. Monitor all three separately.

---

## Cost & Resource Analysis

**Logging infrastructure:** 1M events/day = ~50 GB/day (compressed). Storage: $100-500/month depending on retention (7-30 days).
**Metrics infrastructure:** Prometheus, Grafana, alerting: $200-1000/month.
**Analysis tools:** Notebooks, dashboards: $500-2K/month.
**Total:** $1-3.5K/month for production system.

**ROI:** If monitoring prevents 1 outage/month (worth $100K), cost is 1-3.5% of value prevented.

---

## Monitoring & Observability

**Key metrics:** Latency (p50, p99), error rate, prediction distribution (mean, std, p10/p90), inference throughput, queue depth, model version distribution

**Alerts:** Latency >2x baseline, error rate spike >2x, distribution KS-test p<0.05, queue depth >threshold, pipeline failures, prediction deadlock (no output for 1h)

## Common Mistakes / Gotchas
- No baseline: don't know if latency is abnormal
- Alert fatigue: alert on everything → alerts ignored
- Insufficient retention: logs deleted after 7 days → can't debug old issues
- Only happy path monitoring: forget to monitor errors

## Best Practices
- **Establish baseline:** p50, p99 latency, error rate from first week
- **Alert on anomalies:** latency > 2x baseline = alert
- **Retention policy:** logs 7 days (hot), 30 days (cold archive)
- **Sampling:** sample high-volume low-signal data (don't log all successful predictions)
- **Business metrics:** track revenue impact (not just latency)

## Code Example
```python
import time, logging
from prometheus_client import Histogram, Counter

latency_metric = Histogram('inference_latency_ms', 'Inference latency')
error_counter = Counter('inference_errors_total', 'Total errors')

def predict_with_monitoring(features):
    start = time.time()
    try:
        prediction = model.predict(features)
        latency = (time.time() - start) * 1000
        latency_metric.observe(latency)
        logging.info(f"Prediction: {prediction}, latency: {latency}ms")
        return prediction
    except Exception as e:
        error_counter.inc()
        logging.error(f"Prediction failed: {e}")
        raise
```

## Interview Q&A
**Q: Latency spikes to 500ms. Cause?**
A: Check pillars: (1) Metrics: is CPU/GPU util high? (2) Logs: is feature extraction slow? (3) Traces: which component is slow? Investigate in order.

**Q: Alert on prediction >0.95 (suspicious). Problem?**
A: Too noisy. Better: alert on distribution shift (0.95 used to be 0.01% of predictions, now 10%). Track distribution, not individual predictions.

## Interview Quick-Reference
| Signal | Alert Threshold |
|--------|-----------------|
| Latency | >2x baseline |
| Error rate | >1% (from 0.1%) |
| Drift | KS-test p<0.05 |

## Failure Scenarios

### Failure 1: Cardinality Explosion in Metrics
**Symptom:** Prometheus runs out of memory (OOM). The time-series database shows 10M+ active series. Dashboards stop loading. Alerts stop firing.
**Root Cause:** A developer added a high-cardinality label — such as `user_id` or `request_id` — to a per-request counter or histogram. With 1M unique users, a single metric creates 1M time series.
**Detection:** Monitor `prometheus_tsdb_head_series` metric. Alert when it exceeds 5M. Review new metrics added in recent deployments by diffing the metric registry.
**Fix:** Remove high-cardinality labels from counters and histograms immediately. Use exemplars (Prometheus native histograms) to link individual requests to traces without storing per-request labels. Enforce a cardinality budget review as part of the metrics PR review checklist.

### Failure 2: Alert Fatigue Leading to Missed Outage
**Symptom:** Oncall receives 200 alerts per day. Engineers habituate to ignoring them. A real outage (model serving 100% errors) fires the same alert channel and goes unaddressed for 40 minutes.
**Root Cause:** Alerts are configured on symptoms (CPU > 80%, memory > 2 GB) rather than on SLO burn rate. Too many low-signal alerts drown out high-signal ones.
**Detection:** Measure the alert-to-incident correlation rate. If fewer than 10% of alerts correspond to a real incident requiring human action, alert fatigue is present.
**Fix:** Redesign alerting around SLO burn rate exclusively. Example: "error budget consuming at > 5× the sustainable rate for the last 1 hour" fires a page. Symptom-based alerts (CPU, memory) become INFO-level tickets, never pages. This typically reduces page volume by 80-90%.

### Failure 3: P50 Looks Fine, P99 Broken
**Symptom:** Latency dashboard shows 50ms (healthy, within SLO). Users are reporting 10-second timeouts and the support queue is growing.
**Root Cause:** The dashboard shows p50 (median). The p99 tail latency is 12 seconds due to a GC pause affecting 1% of requests — invisible at the median but devastating for real users.
**Detection:** Always display p50, p95, and p99 in the same panel. Set SLO alerts on p99, not p50. The difference between p50 and p99 should be less than 3×; if it exceeds 10×, investigate tail latency causes immediately.
**Fix:** Audit all latency dashboards: replace any mean or p50 panels with multi-percentile panels. Page on p99 SLO breach. For the GC issue specifically, tune JVM heap sizing or switch to a non-JVM inference server.

### Failure 4: Metric vs. Log Mismatch
**Symptom:** The error rate counter shows 0.1% (healthy). Log analysis reveals 5% of requests are returning malformed predictions that clients silently discard.
**Root Cause:** The error counter only increments on uncaught exceptions. Malformed predictions are returned with HTTP 200 but contain semantically wrong data. The client SDK silently falls back to a default — these are never counted as errors in the metrics.
**Detection:** Reconcile the metric error rate against log-level error patterns weekly. Set up a log-based metric for semantically invalid responses (e.g., prediction confidence = -1 or output schema validation failures).
**Fix:** Instrument all error paths — including semantic errors that do not raise exceptions — to increment the error counter. Add a response schema validator at the serving layer that increments `schema_validation_error_total` for malformed outputs.

### Failure 5: Batch Job Produces Zero Output Silently
**Symptom:** Batch prediction job completes with exit code 0. Downstream system uses stale predictions from 48 hours ago. No alert fires.
**Root Cause:** The job ran, wrote 0 rows to the output table, and exited cleanly. The monitoring only checks job completion status, not output shape or row count.
**Detection:** Monitor three signals for every batch job: (1) job exit code, (2) output row count (must be > 0 and within ±20% of historical average), (3) output freshness timestamp (must be < 1.5× the expected interval).
**Fix:** Add a post-job validation step that asserts `row_count > 0`. Wire this to the same alerting channel as job failures. Treat "zero output" as equivalent to job failure for alerting purposes.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Datadog metrics ingestion | $0.10/1,000 custom metrics | 1M metrics/day | $3,000 |
| Log ingestion and storage (ELK/OpenSearch) | $1.50/GB | 50 GB/day | $2,250 |
| Distributed trace storage (Jaeger/Tempo) | $0.023/GB | 10 GB/day | $7 |
| On-call tooling (PagerDuty) | $19/user/month | 5 engineers | $95 |
| Dashboard and alert engineering time | $200/hr | 10 hr/month | $2,000 |
| **Total** | | | **~$7,352/month** |

Log ingestion is the largest variable cost driver. Teams commonly reduce it by 60-80% by sampling high-volume, low-signal logs (successful predictions at high QPS) to 10% retention while keeping 100% of errors, anomalies, and low-confidence predictions. This brings log costs to approximately $450/month without sacrificing debuggability for the cases that matter.

---

## Interview Q&A

**Q1: Latency p99 usually 100ms, now 200ms. Problem?**
A: Not necessarily. If p50 is unchanged, the p99 spike could be a single slow request or a transient GC pause. Check: (1) is p50 affected? (2) is it persistent over 20+ minutes? (3) what changed recently (code deploy, traffic pattern, data size)? If p50 is stable and the spike is transient, treat as noise. Page only if the p99 SLO is breached sustainably.

**Q2: Monitoring 1M predictions per day. Cannot log all. What to do?**
A: Stratified sampling: (1) log 100% of errors and low-confidence predictions; (2) log 10% of correct predictions from minority classes; (3) log 1% of high-confidence majority-class predictions. This reduces volume to 50-100K events/day while preserving full coverage of the failure modes that matter.

**Q3: Alert tuning — how many false alarms are acceptable?**
A: Depends on response cost. If an alert wakes an oncall engineer (expensive), target fewer than 1 false positive per week per alert. If an alert triggers automated rollback (cheap), tolerate 5-10% false positive rate. False negatives (missed real outages) are almost always more costly than false positives in production ML.

**Q4: How do you distinguish data drift from model degradation?**
A: Three-way check: (1) if input distribution changed but predictions did not — data drift, model may be robust; (2) if inputs unchanged but prediction distribution changed — model drift or serving code bug; (3) if predictions unchanged but accuracy dropped — ground-truth label drift or a labeling pipeline bug. Monitor all three separately with independent alerts.

**Q5: When would you NOT monitor at the prediction level?**
A: For very high-frequency low-value predictions (e.g., content ranking at 100K QPS), per-prediction logging is cost-prohibitive ($2,250/month at 50 GB/day). Use aggregate-level monitoring only — distribution histograms, p50/p99 latency, error rate — and rely on anomaly detection rather than per-event analysis.

**Q6: What breaks first when your observability stack scales to 10× traffic?**
A: Cardinality explosion in metrics (Prometheus OOM) is typically the first failure. Log ingestion costs balloon from $2K to $22K/month if sampling rates are not adjusted. The fix is pre-planned: have a sampling rate adjustment playbook ready before 10× traffic arrives, triggered automatically when ingestion cost crosses a budget threshold.

**Q7: Latency spikes to 500ms. How do you diagnose it?**
A: Work through the three pillars in order: (1) Metrics — is CPU/GPU utilization high? Is queue depth elevated? (2) Logs — is feature extraction the slow step? Is inference taking longer? (3) Traces — find the specific span with high duration. Most latency issues resolve at the metrics or logs layer before requiring trace-level analysis.

**Q8: Alert on prediction > 0.95 (suspicious). What's wrong with this approach?**
A: Too noisy. Individual prediction values have no baseline context. Better: alert on distribution shift — "0.95+ confidence used to be 0.01% of predictions; now it is 10% of predictions" (KS-test p < 0.05). Track the distribution, not individual values.

## Related Topics
- [Drift Detection](15-drift-detection.md)
- [Model Registry](04-model-registry.md)

## Resources
- [Observability Engineering](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
