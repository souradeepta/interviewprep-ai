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

**Failure 1: Rank Shift After Retraining**
- **Symptom:** Top feature "age" drops from rank 1 to rank 15 after routine retraining — silent model behavior change, no accuracy alert fires.
- **Root cause:** New training data introduced a correlation shift. "Age" is now less predictive because the incoming cohort has a narrower age distribution.
- **Detection:** Track the top-10 SHAP rank weekly. Alert if any top-5 feature's rank increases by more than 5 positions in a single week.
- **Fix:** Investigate distribution shift in the displaced feature. Visualize its marginal distribution week-over-week. Retrain only after the root cause is understood — a hasty retrain locks in the shift without explanation.

**Failure 2: Permutation Importance Computed on Test Set**
- **Symptom:** Post-mortem reveals "date" as the number-one feature, exposing data leakage that was invisible before launch.
- **Root cause:** Permutation importance was computed on the test set instead of the validation set. Shuffling test-set features reveals what the test set actually leaked.
- **Detection:** Code review policy — importance computation must reference the held-out validation split. Audit scripts at every release.
- **Fix:** Always compute importance on the validation set (held back from training). Treat test data as write-once: no feature engineering, no importance computation.

**Failure 3: Importance Drift Without Accuracy Change**
- **Symptom:** Feature ranks shift dramatically over six months while accuracy stays flat — the drift goes unnoticed until a stakeholder asks why the model "ignores income."
- **Root cause:** Two correlated features co-move. The model silently switched reliance from one to the other, preserving accuracy but fundamentally changing its logic.
- **Detection:** Track pairwise feature correlation alongside importance. Alert when importance changes by more than 40% even if accuracy is stable.
- **Fix:** Investigate the correlated-feature pair. Document the substitution. If the new anchor feature is less stable or interpretable, add an explicit constraint to prefer the business-meaningful feature.

**Failure 4: Static Importance Masks Temporal Patterns**
- **Symptom:** Importance report consistently shows irrelevant features as high-rank for a time-series churn model.
- **Root cause:** SHAP was computed on a randomly shuffled dataset, destroying temporal order. Features that are only meaningful in sequence appear weak when shuffled.
- **Detection:** Audit whether the dataset used for SHAP preserves the temporal ordering of the original data.
- **Fix:** For time-series models, use time-ordered SHAP windows (compute importance on rolling 30-day slices in chronological order). Never shuffle time-series data for importance analysis.

**Failure 5: False-Positive Alerts from Seasonal Features**
- **Symptom:** "Temperature" feature importance drops 60% each winter, triggering alerts and wasting investigation time (alert fatigue builds over two seasons).
- **Root cause:** Alert compares to prior week, not to the same week last year. Seasonal patterns look like drift.
- **Detection:** Before classifying as anomalous, compare to the same period in the prior year. If year-over-year change is < 15%, flag as seasonal, not drift.
- **Fix:** Build a seasonal baseline alongside the rolling baseline. Alert only when importance deviates from the seasonal expectation, not just from the recent average.

---

## Implementation Guidance

**Wrong:** Alert on every 5% change in importance. Causes alert fatigue.
**Right:** Alert on >50% sustained change (e.g., >50% for 2+ consecutive days). Use rolling average to smooth noise.

**Wrong:** Track importance but don't act on it.
**Right:** Define action plan upfront. If top feature drops 50%: (1) check data quality, (2) retrain model, (3) escalate if both OK.

---

## Interview Q&A

**Q1: Feature importance tracking: what metric to use? SHAP vs permutation vs tree-based?**
A: Start with tree-based importance (< 10ms, available natively in sklearn/XGBoost). Use SHAP for post-hoc explanations that need additive decomposition (batch daily, not real-time — 100-1000ms per batch). Permutation importance is the slowest (can take minutes on large datasets) and is only justified when you need model-agnostic importance or want to confirm SHAP rankings. The most important property is *consistency over time*, not absolute value: all three methods should agree on the top-5 features. If they disagree, investigate before trusting any of them.

**Q2: Top feature drops from 0.35 to 0.18 importance. Do you alert immediately?**
A: Not immediately. First check if this is a single noisy measurement or a sustained trend. Recompute importance five times on different random samples — if the standard deviation is > 10%, the drop is measurement noise, not a real signal. If standard deviation is < 5% and the drop persists for two consecutive measurements, investigate. A single 48% drop in one measurement is common; the same drop sustained over 3+ days is a genuine signal.

**Q3: 100 features, daily SHAP computation takes 10 minutes. How do you make it tractable?**
A: Three levers: (1) Sample — compute SHAP on 10K randomly selected rows instead of all 100M (95% of the information at 0.01% of the cost). (2) Track only the top-20 features — they typically account for 80% of total importance. (3) Use TreeSHAP (< 1ms per row) instead of KernelSHAP (1000ms per row) for tree-based models. Combined, you get daily importance tracking in under 30 seconds.

**Q4: Feature A has high training importance but near-zero importance in production. Why?**
A: Three root causes to check in order: (1) Feature A is missing or null in the production feature pipeline (most common — check the feature store logs). (2) Feature A's distribution shifted dramatically from training to production (drift). (3) Feature A is being computed differently — a pipeline code change changed the semantics. Debug by checking raw feature values at the serving layer, not just the model's input.

**Q5: When would you NOT use feature importance tracking?**
A: When the model is updated too frequently (e.g., re-trained hourly), tracking daily importance creates noise without signal. When the model is a black-box ensemble with no natural importance method, SHAP becomes expensive enough that sampling errors dominate. And when the feature set is deliberately obfuscated for privacy (some regulatory contexts require that feature attributions not be stored or surfaced at all).

**Q6: Feature importance is stable but model accuracy drops 8%. What's happening?**
A: This is the "silent shift" pattern. The model has found a new way to use the same features to produce confident-but-wrong predictions. Likely causes: (1) Label shift — the ground truth distribution changed but features didn't. (2) The model is over-relying on a proxy feature that recently became spuriously correlated with the label. (3) Population shift — a new cohort is entering the system that looks like existing users but behaves differently. Importance stability is a necessary but not sufficient condition for model health.

**Q7: How do you scale feature importance tracking to 50 models running in production?**
A: Three design decisions: (1) Centralize the tracking infrastructure — one shared SHAP compute cluster with a job queue, not 50 independent tracking pipelines. (2) Tier models by business risk: high-risk models (fraud, pricing) get daily SHAP; medium-risk get weekly TreeSHAP; low-risk get monthly on-demand. (3) Standardize the importance schema across all models so a single dashboard can surface cross-model anomalies. The goal is that importance tracking is a platform capability, not a per-model implementation.

**Q8: What breaks at 10× scale (from 1M to 10M daily predictions)?**
A: SHAP computation time scales linearly with sample size — 10× predictions means 10× compute if you don't sample. The fix is to keep the SHAP sample size fixed at 10K rows regardless of traffic volume (importance estimates converge well before 10K). Storage for importance history becomes non-trivial at scale: 100 features × 50 models × 365 days × 8 bytes = ~1.5GB/year, which is manageable. The real scaling problem is alert noise: 50 models × daily alerts × tuning debt compounds — invest in per-model baseline seasonality models early.

**Q9: Feature importance changes by 5% weekly. Do you alert?**
A: No. A 5% weekly change is almost always measurement noise from sampling variance. Set the alert threshold at 30-50% sustained change. Use a 7-day rolling average to smooth single-day spikes. Only if the rolling average crosses 30% do you escalate to investigation.

**Q10: An important feature disappeared from the top-10 in one week. What's the investigation path?**
A: Follow this sequence: (1) Check if the feature has missing values in recent data (pipeline failure). (2) Check if the feature is highly correlated with another top-10 feature that recently moved up (collinearity switch). (3) Check if the population has shifted such that the feature is less discriminative for the new cohort. (4) If all three are clear, retrain the model and confirm importance stabilizes. Document the episode in the model card as a known sensitivity.

---

## Cost & Resource Analysis

**Importance computation:** SHAP 100-1000ms per batch. For 1M daily predictions: 10-100 min = $50-500/month compute.
**Storage:** 12 months importance history for 100 features ≈ 1GB = $10/month.
**Alerting + infrastructure:** $100-500/month.
**Total:** $200-1000/month.

**ROI:** Early detection of data pipeline issues prevents 1-2 incidents/month. Cost per incident: $10K+ (downtime, debug time). Break-even at 1 incident per year.

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| SHAP weekly compute (GPU) | $2/hr | 4 hr/run × 4 runs | $32 |
| Feature store reads for SHAP data | $0.0001/row | 100K rows/week | $1.60 |
| Dashboard hosting (Grafana) | $25/mo | 1 instance | $25 |
| Engineer review time | $200/hr | 1 hr/week | $800 |
| **Total** | | | **~$859/month** |

The dominant cost is engineer review time ($800/month, ~93% of total). Compute and storage are negligible — $58/month combined. This means the first optimization lever is reducing alert noise (fewer false positives = less review time), not cheaper SHAP computation. A well-tuned alert threshold that eliminates seasonal false positives can cut review time to 30 minutes/week, reducing the total to ~$315/month. At that rate, the system pays for itself by preventing a single data pipeline incident (average cost $10K) roughly every four months.

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
