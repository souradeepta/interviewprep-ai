# Feature Importance Tracking

## TL;DR
Monitor: which features contribute to predictions, which are unused. Track over time. If important feature becomes unused (correlation dropped), investigate (data quality, feature computation bug).

## Core Intuition
Model relies on 10 features today, 5 tomorrow (5 stopped being useful). Track this: indicates data drift or feature pipeline issue.

## How It Works

**Tracking workflow:**
1. Compute feature importance (SHAP, permutation, tree-based)
2. Store importance over time
3. Compare: is top-5 feature set stable?
4. Alert: if important feature importance drops >50%

| Date | Top Feature | Importance | Change |
|------|---|---|---|
| Jan 1 | income | 0.35 | - |
| Jan 8 | income | 0.32 | -8% |
| Jan 15 | income | 0.18 | -47% ⚠️ |

## Key Properties / Trade-offs
- Computation cost: importance computation is expensive
- Frequency: daily computation vs weekly (trade cost vs freshness)

## Detailed Trade-off Analysis

| Approach | Frequency | Computation | Latency to Detect Drift | Storage | Cost |
|----------|-----------|-------------|------------------------|---------|------|
| Daily SHAP | Once/day | 100-1000ms per batch | 1 day | 1GB/month | $500/month |
| Weekly TreeSHAP | Once/week | 10-100ms per batch | 1 week | 100MB/month | $100/month |
| Sampled importance | Daily (10%) | 10ms per batch | 1 day | 100MB/month | $50/month |
| On-demand (no tracking) | Ad-hoc | 1000ms | Alert-based | N/A | $0 (misses drift) |

**Decision:** Real-time critical → daily sampling. Audit compliance → weekly full. Cost-sensitive → on-demand with alerts.

---

## Production Failure Scenarios

**Scenario 1: False positive alert on seasonal change**
- Winter: "temperature" feature importance drops 60% (seasonal data pattern).
- System alerts, team investigates, finds no issue. Alert fatigue.
- Fix: Model seasonality. Compare to same season last year, not last week.

**Scenario 2: Feature importance drops but model accuracy unchanged**
- Top feature importance drops 50%. Alert triggered, team investigates, model still works.
- Root cause: feature highly correlated with another. Model learned to use backup feature.
- Fix: Alert only if BOTH importance drops AND accuracy drops.

**Scenario 3: Importance computation bug**
- Compute importance twice per week. One report shows feature A at 0.8, other shows 0.3.
- Root cause: different data samples, different random seeds in SHAP.
- Fix: Use same data subset, fix random seed. Importance should be reproducible.

**Scenario 4: Importance stops updating**
- Track daily importance. One day no data comes in (pipeline failed). Importance frozen.
- Team doesn't notice. Data quality issue undetected for 3 days.
- Fix: Alert if importance hasn't updated in 24 hours. Monitor pipeline health separately.

---

## Implementation Guidance

**Wrong:** Alert on every 5% change in importance. Causes alert fatigue.
**Right:** Alert on >50% sustained change (e.g., >50% for 2+ consecutive days). Use rolling average to smooth noise.

**Wrong:** Track importance but don't act on it.
**Right:** Define action plan upfront. If top feature drops 50%: (1) check data quality, (2) retrain model, (3) escalate if both OK.

---

## Sophisticated Interview Q&A

**Q1: Feature importance tracking: what metric to use? SHAP vs permutation vs tree?**
A: Use tree-based (fast, <10ms). If need post-hoc SHAP, batch daily (not real-time). Permutation slowest, avoid. All should show same top-5 features—if not, investigate. Most important: consistency over time, not absolute value.

**Q2: Top feature drops from 0.35 to 0.18 importance. Investigate or alert immediately?**
A: Check first: (1) Is this single measurement or sustained? (2) Calculate confidence interval (recompute 5x, what's std dev?). (3) If <5% std dev AND sustained, investigate. If >10% std dev (high noise), wait 1 week for trend.

**Q3: 100 features, importance computation takes 10 minutes. How track daily?**
A: (1) Batch sample: compute on 10K random rows, not all 100M. (2) Track only top-20 features (80% of importance anyway). (3) Use lightweight method (tree importance <1ms) vs SHAP (100ms). (4) Or: compute weekly, not daily.

**Q4: Feature A importance high in training, zero in production. Why?**
A: (1) Feature A missing in production data (pipeline issue). (2) Feature A values changed (drift). (3) Feature A not computed correctly (bug). Debug: (1) Check raw data, (2) check pipeline logs, (3) check data distribution vs training. Then: (1) fix pipeline, (2) retrain if distribution changed permanently.

---

## Cost & Resource Analysis

**Importance computation:** SHAP 100-1000ms per batch. For 1M daily predictions: 10-100 min = $50-500/month compute.
**Storage:** 12 months importance history for 100 features ≈ 1GB = $10/month.
**Alerting + infrastructure:** $100-500/month.
**Total:** $200-1000/month.

**ROI:** Early detection of data pipeline issues prevents 1-2 incidents/month. Cost per incident: $10K+ (downtime, debug time). Break-even at 1 incident per year.

---

## Monitoring & Observability

**Key metrics:** Top-5 feature importance stability, importance variance (noise level), feature importance correlation (are top features correlated?), computation latency, tracking update frequency

**Alerts:** Top feature importance drops >50%, new features enter top-10 (could indicate data issue), importance computation fails (pipeline problem), importance stops updating (staleness >24 hours)

## Common Mistakes / Gotchas
- Not tracking: ship model, never revisit feature importance
- Over-sensitive: alert on 1% change (noise)
- Ignoring correlation: importance might change due to seasonal pattern (not a problem)

## Best Practices
- **Baseline importance:** establish from training, compare to production
- **Threshold:** alert if importance drops >30% (noise-resistant)
- **Root cause:** when importance drops, investigate: data quality, correlation drift, model retraining
- **Action plan:** if feature broke, decide: fix pipeline or retrain model

## Code Example
```python
import shap

def track_feature_importance(model, X, date):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = np.abs(shap_values).mean(axis=0)
    
    importance_dict = {
        'date': date,
        'features': X.columns.tolist(),
        'importance': importance.tolist()
    }
    
    # Store in database
    db.save_importance(importance_dict)
```

## Interview Q&A
**Q: Feature importance changes by 5% weekly. Alert?**
A: No, probably noise. Set threshold at 30-50% change. Or: compute 7-day rolling average to smooth noise.

**Q: Important feature disappeared from top-10. Cause?**
A: (1) Check if feature values changed (drift). (2) Check if feature is highly correlated with another feature (collinearity). (3) Retrain model, importance should stabilize.

## Interview Quick-Reference
| Alert Threshold | When |
|---|---|
| 20% change | Sensitive (may have false positives) |
| 50% change | Standard |
| 100% change | Critical (feature no longer used) |

## Related Topics
- [Drift Detection](15-drift-detection.md)
- [Model Monitoring](16-monitoring-and-observability.md)

## Resources
- [Feature Importance Guide](https://christophgerstner.github.io/interpretable_machine_learning_book/)
