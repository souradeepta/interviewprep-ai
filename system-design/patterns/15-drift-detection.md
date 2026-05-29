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

## Failure Scenarios

### Failure 1: PSI Threshold Too Permissive
**Symptom:** Significant feature drift goes undetected for 3+ months, causing silent model degradation. Accuracy drops 8% before anyone notices.
**Root Cause:** The Population Stability Index (PSI) warning threshold was set at 0.25 (the "critical" level in industry literature) rather than 0.1 (the "warning" level). The drift was real but measured below the alert boundary.
**Detection:** Retrospective audit of PSI history will show the value crossed 0.1 weeks before the accuracy drop was detected. Cross-correlate PSI history with model accuracy time series to find lag.
**Fix:** Use 0.1 as the warning threshold and 0.25 as critical. Alert on three consecutive weeks where PSI exceeds the warning level, even if it never hits critical — gradual drift is often more dangerous than sudden spikes.

### Failure 2: Monitoring Too Few Features
**Symptom:** A drifting feature causes a 6% accuracy drop, but no alert fires because that feature was not in the monitored set.
**Root Cause:** The team monitors only the top-5 features by training-time importance. The drifting feature ranked 12th — important, but excluded from monitoring.
**Detection:** Post-mortem feature correlation analysis will identify which unmonitored features correlate with the accuracy drop period.
**Fix:** Monitor all features in the top-30 by importance. Refresh the importance ranking monthly — feature importance changes as data distributions shift. Automate the list update as part of the monthly model health review.

### Failure 3: Reference Window Staleness
**Symptom:** Drift alerts fire constantly during the holiday season even though model performance is fine. Oncall is flooded and begins ignoring alerts.
**Root Cause:** The reference distribution is a static 12-month-old baseline from a non-holiday period. Holiday traffic patterns look like "drift" compared to that baseline even though it is expected seasonal variation.
**Detection:** Compare PSI trend lines against a seasonal calendar. Alerts that cluster around known seasonal events (Black Friday, back-to-school) without accompanying accuracy drops are reference window mismatches.
**Fix:** Use a rolling 30-day reference window for high-seasonality features. For known seasonal periods, maintain a per-season baseline (e.g., "compare this Black Friday to last Black Friday").

### Failure 4: Alert Fatigue from Oversensitive Thresholds
**Symptom:** Oncall receives 50+ drift alerts per week. Engineers stop acting on them. A real concept drift event goes unaddressed for two weeks, causing a 12% accuracy regression.
**Root Cause:** PSI threshold set at 0.05 (too sensitive), no severity tiering, and every feature alerts with equal urgency regardless of business impact.
**Detection:** Measure the alert-to-action rate: if fewer than 20% of alerts result in any human action within 48 hours, alert fatigue is present.
**Fix:** Implement three severity tiers — INFO (PSI 0.05-0.1, log only), WARNING (PSI 0.1-0.25, ticket created), CRITICAL (PSI > 0.25 or accuracy drop confirmed, page oncall). Only page on CRITICAL. Use SLO burn rate as the primary paging signal, not raw PSI.

### Failure 5: Retraining Triggered on Benign Drift
**Symptom:** Auto-retrain fires on minor seasonal feature shift. The newly trained model performs 2% worse because it over-indexed on the shifted recent distribution.
**Root Cause:** Retraining pipeline has no human gate — any drift above threshold triggers immediate retraining without checking whether model performance actually degraded.
**Detection:** Compare model accuracy before and after retraining. If post-retrain accuracy is worse, the retrain was unnecessary.
**Fix:** Gate retraining on two conditions: (1) drift detected AND (2) accuracy drop confirmed (>1% relative, sustained for 48 hours). Drift alone is not sufficient. Add a champion-challenger evaluation before promoting the retrained model.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| PSI computation (AWS Lambda) | $0.0001/GB processed | 50 GB/day | $150 |
| Shadow model inference (10% traffic sample) | $0.002/request | 100K req/day | $6,000 |
| CloudWatch custom metrics storage | $0.30/metric/month | 100 metrics | $30 |
| Ground-truth labeling for accuracy evaluation | $0.10/sample | 1,000 samples/week | $400 |
| Engineer oncall time for drift investigations | $200/hr | 4 hr/month | $800 |
| **Total** | | | **~$7,380/month** |

The dominant cost is shadow model inference ($6K/month). Teams often reduce this by sampling 1% instead of 10%, bringing it to $600/month — but detection latency increases from minutes to hours. For models where a 1-hour delay in detecting drift is acceptable, 1% sampling is the right economic trade-off. Label acquisition is the second largest driver; prioritizing labeling budget toward high-drift periods maximizes ROI.

---

## Interview Q&A

**Q1: Drift p-value = 0.03 (significant). But model accuracy unchanged. Retrain?**
A: No. Input distribution drift does not automatically mean output quality degraded. Monitor accuracy directly. If accuracy drops next week following the drift, retrain. If accuracy remains stable, the model is robust to this particular shift — retraining would be unnecessary cost.

**Q2: 100K features. Cannot monitor all. What do you do?**
A: Three-tier strategy: (1) monitor the top-10 features by SHAP importance for every prediction — these drive the most variance; (2) use a PCA-based aggregate drift signal across the remaining features to detect broad shifts without per-feature overhead; (3) track proxy summary statistics (mean, standard deviation, null rate) instead of full distribution tests for features ranked 11-50.

**Q3: Retraining is expensive. Drift detected on a non-critical feature. Skip retraining?**
A: Depends on trajectory. A single non-critical feature drifting in isolation — skip. If the same feature drifts for 5+ consecutive days, or if it begins correlating with drift in other features, it is likely a leading indicator of concept drift — retrain. Cost-benefit: if retraining costs $1K and the potential accuracy loss is worth $10K/week, the break-even is less than one week of degradation.

**Q4: How long to wait after drift detection before retraining?**
A: Three-tier decision: (1) PSI > 0.25 AND accuracy drop confirmed — retrain immediately; (2) PSI 0.1-0.25 with no accuracy drop — wait 3 days and recheck; (3) PSI < 0.1 with marginal drift — weekly review decision, no immediate action.

**Q5: When would you NOT use drift detection at all?**
A: For models with very short retraining cycles (daily retraining regardless of drift), the overhead of drift monitoring may exceed its value. Also, for models whose inputs are fully synthetic or controlled (e.g., a model operating only on internal system signals that cannot drift), continuous drift monitoring adds cost without benefit.

**Q6: What breaks first when drift detection scales to 1,000 models in production?**
A: Alert management becomes unmanageable — 1,000 models × 30 features each = 30,000 potential daily alerts. The engineering solution is aggregated drift dashboards with per-model health scores rather than per-feature alerts. Automation must handle the first tier (auto-retrain for high-confidence critical drifts) so engineers only review ambiguous cases.

**Q7: How do you distinguish seasonal drift from structural concept drift?**
A: Compare the current period against the same period in the prior year, not just against last month. If PSI is high compared to last month but near zero compared to the same month last year, it is seasonal variation. If PSI is high against both, it is structural concept drift and retraining is warranted.

**Q8: Retraining costs $10K. Drift on a minor feature. Retrain?**
A: No. Rank features by prediction impact. Retrain only when high-impact features drift. For low-impact features, consider targeted re-weighting or partial updates rather than full retraining.

## Related Topics
- [Model Monitoring](16-monitoring-and-observability.md)
- [Retraining Pipelines](02-data-pipelines.md)

## Resources
- [Drift Detection Methods](https://en.wikipedia.org/wiki/Concept_drift)
