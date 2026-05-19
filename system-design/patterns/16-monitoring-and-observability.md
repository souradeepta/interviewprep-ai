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
