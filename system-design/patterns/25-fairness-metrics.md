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

## Detailed Trade-off Analysis

| Metric | Accuracy Impact | Implementation | Legal Standing | Interpretability |
|--------|-----------------|----------------|-----------------|-----------------|
| Demographic parity | 2-5% drop | Easy (adjust thresholds) | Weak (ignores merit) | High |
| Equalized odds | 3-8% drop | Medium (retrain+threshold) | Strong (equal opportunity) | Medium |
| Predictive parity | 5-10% drop | Hard (requires different models) | Variable | Low |
| Individual fairness | 1-3% drop | Hard (similarity metric) | Undefined | Very low |
| Calibration | <1% drop | Easy (post-processing) | Strong (explainable) | High |

**Decision:** Regulatory requirement exists → use that metric. Legal uncertainty → equalized odds (defensible). Accuracy critical → calibration.

---

## Production Failure Scenarios

**Scenario 1: Enforce demographic parity, hiring discrimination lawsuit**
- Equal approval rates (50% male, 50% female). But male threshold 40%, female 70%.
- Hired qualified males over more-qualified females. "Reverse discrimination" lawsuit.
- Prevention: Use equalized odds (equal opportunity) for hiring, not demographic parity (equal outcome).

**Scenario 2: Fairness metric says fair, stakeholders say unfair**
- 5% TPR gap (equalized odds threshold). Within tolerance. But affected group reports systematic rejection.
- Metric doesn't capture lived experience.
- Prevention: Fairness metrics + stakeholder feedback. Metrics necessary but not sufficient.

**Scenario 3: Fairness-accuracy trade-off hidden from stakeholders**
- Enforce fairness, accuracy drops 5%. Business complains. Conflict between fairness and performance.
- Prevention: Show fairness-accuracy frontier. Let stakeholders choose acceptable trade-off upfront.

**Scenario 4: Group definition changes, fairness metrics invalidated**
- Initially: fairness for male/female. Later add: fairness by age. Metrics were gender-focused, age-naive.
- Fairness audit looks good for gender, bad for age.
- Prevention: Fairness for multiple protected attributes simultaneously. Include all relevant dimensions.

---

## Implementation Guidance

**Wrong:** Pick demographic parity for all use cases (simplest metric).
**Right:** Match metric to use case. Hiring/lending → equalized odds. Recommendations → demographic parity. Check regulatory requirements.

**Wrong:** Enforce fairness by adjusting decision threshold alone.
**Right:** (1) Try threshold adjustment first (fast). (2) If insufficient, retrain with fairness constraint. (3) If still insufficient, accept accuracy drop or change problem formulation.

---

## Sophisticated Interview Q&A

**Q1: Model satisfies demographic parity but equalized odds is violated. Acceptable?**
A: Depends on use case. (1) Hiring: NO—violates equal opportunity (equalized odds). (2) Content ranking: YES—demographic parity sufficient. (3) Lending: MAYBE—if base rates different, demographic parity might violate fairness intent. Clarify requirements first.

**Q2: Equalized odds requires 7% accuracy drop. Too much. Alternative?**
A: (1) Individual fairness (similar users similar treatment, <1% accuracy drop). (2) Calibration (post-hoc, <1% drop). (3) Separate models per group (potentially better for both fairness and accuracy). (4) Change problem: can you change target variable to remove bias? (5) Accept lower accuracy as fairness cost.

**Q3: What's "similar" for individual fairness? 100 feature space, hard to define.**
A: (1) Distance metric: Euclidean on important features only. (2) Fairness constraints: don't let similar individuals get very different predictions (add constraint to loss). (3) Example-based: "person A and B differ only in protected attribute, should get similar predictions". (4) Hard problem—mostly theoretical, less used in practice.

**Q4: Group definition: male/female vs male/non-binary/female. How define fairness across 3 groups?**
A: (1) Compute all pairwise fairness gaps (male vs female, male vs non-binary, female vs non-binary). (2) Or: max gap (compare most-disparate groups). (3) Report all gaps in audit. (4) If max gap >5%, unacceptable. (5) Include all relevant groups in fairness definition.

---

## Cost & Resource Analysis

**Fairness audit:** 8-16 hours per quarter = $4-8K per year.
**Fairness-aware model retraining:** 1-2 weeks if constraint added = $5-10K per iteration.
**Legal review of fairness metrics:** 40 hours law firm = $10K.
**Regulatory compliance (hiring, lending):** dedicated team = $100K+/year.

**Cost of fairness violations:** Lawsuit $500K-5M+, reputational damage, regulatory fines 1-4% revenue.
**ROI:** Proactive fairness work $10-100K/year. Prevents incidents worth $100K-5M. Break-even: 1 incident per year.

---

## Monitoring & Observability

**Key metrics:** Demographic parity ratio (approval rates by group), equalized odds gap (max TPR difference), predictive parity gap, individual fairness distance (min prediction difference for similar users), fairness-accuracy frontier

**Alerts:** Demographic parity gap >5%, equalized odds gap >10%, fairness metric changes >2% month-over-month, new disparity discovered (e.g., intersectional groups), accuracy drops to unacceptable level

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
