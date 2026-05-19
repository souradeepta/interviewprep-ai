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
