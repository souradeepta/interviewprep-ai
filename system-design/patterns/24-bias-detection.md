# Bias Detection

## TL;DR
Check: does model accuracy differ by cohort (age, gender, race)? If yes, investigate root cause (imbalanced training data, unfair features). Measure: fairness metrics (demographic parity, equalized odds).

## Core Intuition
Model 95% accurate overall, but 80% accurate on underrepresented group. Biased. Find bias, fix data or retrain.

## How It Works

**Bias detection steps:**
1. **Slice accuracy:** compute accuracy by cohort (age, gender, race, geo)
2. **Fairness metrics:** 
   - Demographic parity: approval rate same across groups
   - Equalized odds: true positive rate same across groups
   - Predictive parity: precision same across groups
3. **Root cause:** is training data imbalanced? Are features biased?
4. **Fix:** collect more data for underrepresented group, remove biased features

| Cohort | Accuracy | Fairness Gap |
|--------|----------|---|
| Male | 95% | - |
| Female | 85% | 10% gap |
| White | 94% | - |
| Black | 82% | 12% gap |

## Key Properties / Trade-offs
- Audit overhead: fairness metrics add complexity
- Fairness vs accuracy: enforcing fairness might reduce overall accuracy
- Which metric: demographic parity vs equalized odds have trade-offs

## Common Mistakes / Gotchas
- Not checking for bias: assume model is fair (wrong)
- Proxy variables: remove gender, but model learns gender from zip code
- Greenwashing: report good fairness metric that doesn't matter
- Ignoring context: fairness metric appropriate for one domain might not be for another

## Best Practices
- **Multiple metrics:** use demographic parity + equalized odds
- **Regular audits:** check for bias monthly
- **Stakeholder input:** ask affected groups what fairness means
- **Fix root cause:** don't hide bias, fix underlying data/features
- **Documentation:** record fairness metrics, explain trade-offs

## Code Example
```python
from sklearn.metrics import confusion_matrix

def compute_fairness_metrics(y_true, y_pred, groups):
    for group_name in groups.unique():
        mask = groups == group_name
        accuracy = (y_true[mask] == y_pred[mask]).mean()
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
        tpr = tp / (tp + fn)  # True positive rate
        print(f"{group_name}: accuracy={accuracy:.1%}, TPR={tpr:.1%}")
```

## Interview Q&A
**Q: Model has 10% fairness gap (male 95%, female 85%). Acceptable?**
A: Unacceptable for high-stakes (hiring, loans). Investigate: why the gap? Is it training data imbalance? Then collect more female examples. Is it biased features? Remove them. Fix root cause.

**Q: Removing gender feature but model still learns it. Why?**
A: Proxy variables: gender correlated with zip code, age, income. Model reconstructs gender from correlated features. Solution: identify all proxy features, remove or decorrelate them.

## Interview Quick-Reference
| Metric | Definition |
|--------|---|
| Demographic parity | approval rate same for all groups |
| Equalized odds | TPR same for all groups |
| Predictive parity | precision same for all groups |

## Related Topics
- [Fairness Metrics](25-fairness-metrics.md)
- [Model Debugging](17-model-debugging.md)

## Resources
- [Fairness and Machine Learning](https://fairmlbook.org/)
