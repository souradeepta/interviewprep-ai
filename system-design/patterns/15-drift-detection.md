# Drift Detection

## TL;DR
Monitor: input feature distribution, model output distribution, target distribution (if available). Drift detected = model performance degrading. Trigger retraining.

## Core Intuition
Model trained on 2024-01 data. In 2024-03, user behavior changed—different feature distribution. Model still uses old patterns → performance drops. Drift detection catches this.

## How It Works

**Three types of drift:**

1. **Data drift:** input feature distribution changed (KS-test p<0.05)
2. **Concept drift:** same inputs but relationship changed (output distribution changed)
3. **Target drift:** ground truth distribution changed

| Drift Type | Example | Detection |
|------------|---------|-----------|
| Data | User age distribution shifted younger | Feature histogram changed |
| Concept | Buying behavior changed post-pandemic | Output distribution changed |
| Target | Class imbalance increased | Label distribution changed |

## Key Properties / Trade-offs
- Sensitivity: 1% threshold catches noise, 10% threshold misses real drift
- Latency: immediate detection vs 1-hour delay
- Overhead: continuous monitoring vs weekly checks

## Detailed Trade-off Analysis

| Aspect | KS-Test (Statistical) | MMD (Distribution) | Evidently (Rules) | Manual Review |
|--------|----------------------|-------------------|-----------------|---------------|
| Detection latency | Immediate | Batch hourly | Configurable | 1-7 days |
| False positive rate | 5% (by design) | 2-3% | <1% (rule-based) | ~0% |
| Sensitivity to noise | High | Medium | Low (stable) | N/A |
| Setup time | 1 day | 2 days | 4 hours | Ongoing |
| Action: automatic | Yes (trigger retrain) | Yes | Optional | No |

**Decision:** High-traffic real-time → KS-test. Batch pipelines → MMD. Explainability critical → rule-based.

---

## Production Failure Scenarios

**Scenario 1: False positive cascade**
- Minor feature drifts (noise). Auto-retrain triggered. Old model was better. Accuracy drops 2%.
- Fix: Require multiple feature agreement. Check if model performance actually dropped (not just distribution).

**Scenario 2: Seasonal drift not detected**
- Winter traffic pattern looks like drift. System retrains. Summer comes, retraining was unnecessary.
- Fix: Model seasonality separately. Compare to same week last year, not last week.

**Scenario 3: Gradual drift below threshold**
- Each day 0.5% shift. P-value stays >0.05. Over 30 days = 15% drift, model fails silently.
- Fix: Use rolling window (month-over-month comparison). Check if multiple consecutive periods show drift direction.

**Scenario 4: Retraining data contaminated**
- Drift detected, retrain triggered. New training data includes bad labels from before drift fixed. Model worsens.
- Fix: Validate training data quality before retraining. Check label distribution for anomalies.

---

## Implementation Guidance

**Wrong:** Trigger retraining on any single feature drift.
**Right:** Require (a) multiple features drifting OR (b) model performance drop confirmation, AND (c) multi-day stability check.

**Wrong:** Compare current week to last week (seasonal noise).
**Right:** Compare current to baseline + same season last year. Use rolling baselines.

---

## Sophisticated Interview Q&A

**Q1: Drift p-value=0.03 (significant). But model accuracy unchanged. Retrain?**
A: No. Drift in input ≠ drift in output relationship. Monitor accuracy directly. If accuracy drops next week after drift, retrain. If accuracy stable, drift is benign (model robust to this shift).

**Q2: 100K features. Can't monitor all. What to do?**
A: (1) Feature importance ranking—monitor top-10 by impact. (2) Aggregate drift signal—PCA or ensemble drift metrics for dimensionality. (3) Proxy metrics—track summary statistics (mean, std) instead of full distributions.

**Q3: Retraining expensive. Drift on non-critical feature. Skip?**
A: Depends on momentum. If single feature, skip. If drifting multiple days and affecting other features, escalates to concept drift—retrain. Cost benefit: 1 bad decision × $10K vs retrain cost $1K.

**Q4: How long wait after drift detected before retraining?**
A: (1) If detected drift has high confidence (p<0.001)—retrain immediately. (2) If marginal (p=0.03-0.05)—wait 1-3 days, check if persists. (3) If unstable (alternating drifts)—weekly summary decision.

---

## Cost & Resource Analysis

**Monitoring infrastructure:** Continuous drift checks require baseline storage and daily computations.
- KS-test: ~$100/month (lightweight)
- MMD or Evidently: ~$500/month (more sophisticated)
- Manual daily review: 1-2 hours engineer/day = ~$15K/month

**Retraining cost:** $1K-$10K per model depending on data size and frequency.
**ROI:** If drift causes 1% accuracy drop on $10M revenue platform = $100K loss. Detecting saves this. Break-even at 1 drift event per month.

---

## Monitoring & Observability

**Key metrics:** P-value trajectory per feature, model accuracy time series, training data distribution, retraining frequency, false positive rate

**Alerts:** p-value <0.05 for 2+ consecutive hours (persistent drift), accuracy drops >1% post-prediction, retraining pipeline fails, baseline stale (>30 days)

## Common Mistakes / Gotchas
- No baseline: don't know if distribution is different
- No retraining trigger: detect drift but don't act
- False positives: alert fatigue from noise
- Ignoring seasonal patterns: summer traffic pattern ≠ drift

## Best Practices
- **Baseline:** establish training data distribution as baseline
- **Rolling window:** compare last week to last 4 weeks (detects drift, ignores noise)
- **Multiple metrics:** check multiple features (not just 1)
- **Action plan:** detect drift → automatically trigger retraining
- **Manual review:** for uncertain cases, notify team for decision

## Code Example
```python
from scipy.stats import ks_2samp

class DriftDetector:
    def __init__(self, baseline_data, threshold=0.05):
        self.baseline = baseline_data
        self.threshold = threshold
    
    def check_drift(self, current_data):
        drifts = []
        
        for feature in current_data.columns:
            statistic, p_value = ks_2samp(
                self.baseline[feature].values,
                current_data[feature].values
            )
            
            if p_value < self.threshold:
                drifts.append({
                    "feature": feature,
                    "p_value": p_value,
                    "action": "Retrain"
                })
        
        return drifts
```

## Interview Q&A
**Q: Drift detected on 1 feature. Trigger retraining?**
A: Maybe not. If multiple features normal, single feature drift could be noise. Check: (1) is it consistent over time? (2) do other features support it? (3) is it explainable (e.g., seasonal)? If unsure, hold for now. Retrain if drift persists >1 week.

**Q: Retraining costs $10K. Drift on minor feature. Retrain?**
A: No. Prioritize: rank features by impact on prediction. Retrain only if high-impact feature drifts. Use quick retraining (not full pipeline) for low-impact features first.

## Interview Quick-Reference
| Drift Type | Cause | Solution |
|-----------|-------|----------|
| Data drift | Feature distribution changed | Retrain |
| Concept drift | Relationship changed | Retrain |
| Seasonal | Expected pattern | Ignore (normal) |

## Related Topics
- [Model Monitoring](16-monitoring-and-observability.md)
- [Retraining Pipelines](02-data-pipelines.md)

## Resources
- [Drift Detection Methods](https://en.wikipedia.org/wiki/Concept_drift)
