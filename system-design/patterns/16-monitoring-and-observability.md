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

## Related Topics
- [Drift Detection](15-drift-detection.md)
- [Model Registry](04-model-registry.md)

## Resources
- [Observability Engineering](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
