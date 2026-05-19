# Fairness Metrics

## TL;DR
Measure fairness quantitatively: demographic parity (approval rate equal), equalized odds (TPR equal), individual fairness (similar inputs treated similarly). Each has trade-offs.

## Core Intuition
Fairness = different groups get treated similarly. Metrics quantify this. Metric should match your fairness definition.

## How It Works

**Three fairness paradigms:**

1. **Demographic parity:** 
   - Same approval rate for all groups
   - Formula: P(pred=1|male) = P(pred=1|female)
   - Use when: want equal outcome

2. **Equalized odds:**
   - Same true positive rate for all groups
   - Formula: P(pred=1|label=1,male) = P(pred=1|label=1,female)
   - Use when: want equal opportunity

3. **Individual fairness:**
   - Similar inputs treated similarly
   - Formula: similar users get similar predictions
   - Use when: want consistency

| Metric | Pros | Cons |
|--------|------|------|
| Demographic parity | Intuitive | Ignores base rates |
| Equalized odds | Principled | Complex to optimize |
| Individual | Consistent | Hard to define "similar" |

## Key Properties / Trade-offs
- Demographic parity vs equalized odds: can't satisfy both (impossibility theorem)
- Fairness vs accuracy: enforcing fairness reduces accuracy
- Group definition: fairness depends on how you define groups

## Common Mistakes / Gotchas
- Picking wrong metric: demographic parity wrong for hiring (base rate differences)
- Ignoring accuracy impact: fairness audit shows 5% accuracy drop → unacceptable
- Ignoring context: fairness metric depends on use case

## Best Practices
- **Stakeholder input:** ask affected groups which metric matters
- **Multiple metrics:** report all three, explain trade-offs
- **Fairness-accuracy frontier:** show accuracy cost of fairness
- **Regulatory compliance:** check which metric your domain requires

## Code Example
```python
def fairness_metrics(y_true, y_pred, protected_attr):
    groups = protected_attr.unique()
    
    for group in groups:
        mask = protected_attr == group
        approval_rate = (y_pred[mask] == 1).mean()
        tpr = (y_pred[mask][y_true[mask] == 1] == 1).mean()
        print(f"{group}: approval={approval_rate:.1%}, TPR={tpr:.1%}")
```

## Interview Q&A
**Q: Which fairness metric to use?**
A: Depends on context. Hiring: equalized odds (equal opportunity). Lending: demographic parity (equal outcome). Healthcare: equalized odds (equal true positive rate). Ask stakeholders first.

**Q: Equalized odds requires 2% accuracy drop. Accept?**
A: Business decision. If accuracy 98% → 96%, might be acceptable. If accuracy 70% → 68%, unacceptable. Measure fairness-accuracy frontier, let stakeholders choose.

## Interview Quick-Reference
| Use Case | Best Metric |
|----------|---|
| Hiring | Equalized odds |
| Lending | Demographic parity |
| Diagnosis | Equalized odds |
| Content ranking | Demographic parity |

## Related Topics
- [Bias Detection](24-bias-detection.md)
- [Interpretability](19-interpretability.md)

## Resources
- [Fairness and Machine Learning Book](https://fairmlbook.org/)
