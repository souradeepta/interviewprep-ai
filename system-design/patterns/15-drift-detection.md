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
