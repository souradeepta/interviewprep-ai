# Monitoring & Observability

## TL;DR
Track model, system, and business metrics in production. Metrics: prediction latency, accuracy, feature distributions (drift), input/output distributions, data quality, system errors. Observability: deep understanding of system behavior (logs, traces, metrics). Enables early detection of failures, model degradation, data drift.

## Core Intuition
Models don't stay correct. Data changes, users change, the world changes. Monitoring watches for these shifts. You can't fix what you don't see. Comprehensive monitoring is an early warning system.

## How It Works

**Key Metrics to Monitor:**

**1. Model Performance Metrics:**
- Accuracy, precision, recall (if labels available)
- Latency: prediction time, p50/p95/p99 percentiles
- Throughput: predictions/second
- Cost: $ per prediction

**2. Data Quality Metrics:**
- Feature distributions (mean, median, std dev)
- Missing values percentage
- Feature correlations
- Input data schema violations

**3. Drift Detection:**
- **Covariate shift:** input distribution changed
  ```
  P(X) changed, but P(Y|X) same
  Example: More mobile users than before, but mobile→purchase rate unchanged
  ```
- **Label shift:** output distribution changed
  ```
  P(Y) changed, but P(X|Y) same
  Example: More fraud attempts, but fraud characteristics unchanged
  ```
- **Concept drift:** P(Y|X) changed (model becomes wrong)
  ```
  Example: User preferences shifted; old model doesn't capture new patterns
  ```

**4. Business Metrics:**
- Click-through rate (CTR)
- Conversion rate
- Revenue impact
- User satisfaction (NPS, ratings)

**Implementation Stack:**

```
Data Sources → Collectors → Storage → Dashboards/Alerts
    ↓
  Logs:       application logs, errors, exceptions
  Metrics:    latency, throughput, accuracy
  Traces:     request flow, dependency calls
    ↓
  Collectors: OpenTelemetry, DataDog, Prometheus
    ↓
  Storage:    TimeSeries DB (InfluxDB, Prometheus), data warehouse
    ↓
  Visualization: Grafana, Kibana, Datadog dashboards
    ↓
  Alerting: Page oncall if thresholds breached
```

**Monitoring Setup Example:**
```
Model performance baseline (from validation):
  - Accuracy: 92%
  - Latency p95: 50ms
  - Error rate: <0.1%

Production monitoring (continuous):
  - Accuracy drops below 90% → Alert (possible label shift)
  - Latency p95 > 100ms → Alert (degradation, needs investigation)
  - Error rate > 1% → Alert (data issues, bugs)
  - Feature: "user_age" has 30% missing → Alert (data pipeline broke)
```

## Key Properties / Trade-offs

| Aspect | Manual Monitoring | Automated |
|--------|---|---|
| Effort | High (manual checks) | Low (automatic dashboards) |
| Detection lag | Days (noticed during review) | Minutes (automated alerts) |
| Cost | High (engineer time) | Medium (storage + compute) |
| Coverage | Incomplete (what do you check?) | Comprehensive (logs everything) |

**Monitoring Cadence:**
- **Real-time:** latency, errors (sec-level alerts)
- **Hourly:** throughput, basic accuracy (if labels available)
- **Daily:** model performance (accuracy), drift (need labels, slower to compute)
- **Weekly:** business metrics, trend analysis

## Common Mistakes / Gotchas

- **Monitoring only model metrics:** System can fail (no alerts), data pipeline can break (no alerts). Monitor system + data + model.
- **Ignoring baseline:** "accuracy is 85%" says nothing without baseline. Always compare to production baseline.
- **Too many alerts:** If everything triggers alerts, engineers ignore them. Set reasonable thresholds, avoid noise.
- **No labels for accuracy:** Can't compute real accuracy without true labels. Use proxy (CTR for recommend), human labeling samples, or estimation (calibration on small labeled set).
- **Alert fatigue:** Alerts every 5 minutes → people stop responding. Aggregate, use percentiles, set proper thresholds.
- **Not versioning models:** Can't tell which model version caused accuracy drop. Always tag model versions.
- **Logs without context:** "Error: X" means nothing. Log request ID, model version, input hash, model prediction, so can reproduce.
- **Ignoring user impact:** "Accuracy dropped 1%" may be statistically insignificant but still hurt revenue. Monitor business metrics too.

## Code Example

```python
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model_version'])
latency_histogram = Histogram('prediction_latency_seconds', 'Prediction latency', ['model_version'])
accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy', ['model_version'])
data_quality_gauge = Gauge('feature_missing_percent', 'Missing feature percentage', ['feature_name'])

def monitor_prediction(model_version, input_data, prediction, latency_sec):
    """Log and track prediction metrics."""
    
    # Log prediction
    logger.info(f"Prediction: version={model_version}, pred={prediction}, latency={latency_sec:.3f}s")
    
    # Track metrics
    prediction_counter.labels(model_version=model_version).inc()
    latency_histogram.labels(model_version=model_version).observe(latency_sec)
    
    # Check data quality
    missing_count = sum(1 for v in input_data.values() if v is None)
    missing_pct = (missing_count / len(input_data)) * 100
    if missing_pct > 5:  # Alert if >5% missing
        logger.warning(f"High missing data: {missing_pct:.1f}%")
        data_quality_gauge.labels(feature_name="global").set(missing_pct)

def monitor_model_drift(prev_predictions, curr_predictions, threshold=0.05):
    """Detect distribution shift (covariate/label drift)."""
    import numpy as np
    
    # Simple KS test for drift
    prev_mean = np.mean(prev_predictions)
    curr_mean = np.mean(curr_predictions)
    
    shift = abs(curr_mean - prev_mean) / (prev_mean + 1e-6)
    if shift > threshold:
        logger.warning(f"Potential drift detected: {shift:.2%} shift in mean prediction")
        return True
    return False

def monitor_accuracy(true_labels, predicted_labels, model_version):
    """Compute and track accuracy."""
    import numpy as np
    
    accuracy = (true_labels == predicted_labels).mean()
    accuracy_gauge.labels(model_version=model_version).set(accuracy)
    
    # Log with context
    logger.info(f"Model accuracy: {accuracy:.3f} (version {model_version})")
    
    # Alert if degradation
    baseline_accuracy = 0.92
    if accuracy < baseline_accuracy * 0.95:  # >5% drop
        logger.error(f"ALERT: Accuracy dropped below threshold! ({accuracy:.3f} < {baseline_accuracy * 0.95:.3f})")
        return False  # Flag for manual review
    return True

# Example usage
if __name__ == "__main__":
    import time
    
    # Simulate predictions
    true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    predicted_labels = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    model_version = "v2.1"
    
    # Monitor predictions
    for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
        start = time.time()
        # ... do prediction ...
        latency = time.time() - start
        
        input_data = {"feature_a": 0.5, "feature_b": None}  # Some missing data
        monitor_prediction(model_version, input_data, pred, latency)
    
    # Monitor accuracy
    monitor_accuracy(true_labels, predicted_labels, model_version)
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "What to monitor?" | Model (accuracy, latency, throughput), data (distributions, quality, drift), system (errors, availability). |
| "How to detect drift?" | Compare feature distributions (current vs baseline). Use statistical tests (KS test, Chi-square). |
| "Accuracy without labels?" | Use proxy metrics (CTR for ranking), sample-label small % of data, or estimate from confidence scores. |
| "Alert thresholds?" | Based on baselines + business impact. Set >5% degradation as alert, not every 1% fluctuation. |
| "Covariate vs concept drift?" | Covariate: input distribution shifted. Concept: P(Y\|X) shifted. Both break models; both need monitoring. |
| "Frequency?" | Real-time for latency/errors. Hourly for throughput. Daily for accuracy (needs labels). |

## Related Topics
- [Drift Detection](drift-detection.md) — deeper dive on detecting distribution shifts
- [Model Versioning](model-versioning.md) — tracking which model version is in production
- [A/B Testing](ab-testing.md) — comparing models; monitoring during experiments
- [Production Readiness](production-readiness.md) — monitoring is part of production setup
- [Disaster Recovery](disaster-recovery.md) — monitoring feeds into recovery strategies

## Resources
- [What Every Machine Learning Team Should Know About Monitoring](https://www.aporia.com/blog/monitoring-machine-learning-models/)
- [Observability in Machine Learning (Datadog)](https://www.datadoghq.com/blog/ml-observability/)
- [WhyLabs: ML Observability Platform](https://whylabs.ai/)
- [Prometheus and Grafana for ML Monitoring](https://prometheus.io/)
