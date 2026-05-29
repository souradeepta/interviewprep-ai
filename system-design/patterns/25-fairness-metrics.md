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

**Failure 1: Wrong Metric Selection Drives Business Rejection of Fairness**
- **Symptom:** The team optimizes for demographic parity (equal approval rates). Equal rates are achieved, but the majority group's accuracy drops 8%. Business rejects the fairness fix as "making the model worse."
- **Root cause:** Demographic parity was the wrong metric for this use case. Equalizing approval rates forced a lower decision threshold for the majority group and a higher one for the minority group, introducing a different form of unfairness.
- **Detection:** Before selecting a fairness metric, simulate each candidate metric's impact on per-group accuracy and business outcomes. If any group's accuracy regresses by more than 5%, the metric choice may be wrong.
- **Fix:** Use equalized odds as the default for high-stakes decisions. Negotiate the fairness-accuracy trade-off with product and legal stakeholders before optimization — not after an already-failing implementation is in production.

**Failure 2: Fairness on Benchmark Does Not Match Deployment Population**
- **Symptom:** Fairness metrics pass on the standard benchmark test set. Six months post-deployment, an independent audit finds fairness violations in the live population.
- **Root cause:** The test set was drawn from historical data. The deployment population shifted (new geography, new user cohort) with a different group distribution. The fairness gap that was absent in the test set is present in the deployment population.
- **Detection:** Re-evaluate fairness metrics quarterly on a sample from the live deployment population, not just the fixed benchmark.
- **Fix:** Maintain a continuously updated fairness evaluation set that reflects the current deployment population. Treat fairness as an ongoing monitoring concern, not a one-time launch gate.

**Failure 3: Intersectionality Ignored**
- **Symptom:** A hiring tool passes fairness audits for gender (no gap) and age (no gap). An employee advocacy group's analysis shows that women over 55 are approved at 41% vs. 79% for men under 40.
- **Root cause:** Fairness evaluation only checked single-attribute splits. Intersectional groups (elderly women) were not evaluated.
- **Detection:** Compute fairness metrics for all two-way attribute combinations: age × gender, age × race, gender × race. Require at least two-way intersection coverage for any model with multiple protected attributes.
- **Fix:** Add intersectional groups to the fairness evaluation suite. Document any gaps above the minimum threshold in the model card, with a remediation plan.

**Failure 4: Rebalancing to Fix Class Imbalance Creates New Bias**
- **Symptom:** Class rebalancing (oversampling minority class) improves minority class recall from 42% to 68%. But precision for a different subgroup drops from 91% to 76%, causing a new fairness violation.
- **Root cause:** Resampling was applied uniformly to the entire dataset without considering the interaction between class balance and subgroup balance. Oversampling minority class instances happened to oversample a specific demographic, shifting the training distribution.
- **Detection:** Compute the full confusion matrix per subgroup before and after resampling. Any precision or recall change greater than 5% for any subgroup is a flag.
- **Fix:** Use stratified resampling that maintains both class balance and protected-attribute balance simultaneously. The resampling strategy must be validated against all fairness metrics, not just aggregate class balance.

---

## Implementation Guidance

**Wrong:** Pick demographic parity for all use cases (simplest metric).
**Right:** Match metric to use case. Hiring/lending → equalized odds. Recommendations → demographic parity. Check regulatory requirements.

**Wrong:** Enforce fairness by adjusting decision threshold alone.
**Right:** (1) Try threshold adjustment first (fast). (2) If insufficient, retrain with fairness constraint. (3) If still insufficient, accept accuracy drop or change problem formulation.

---

## Interview Q&A

**Q1: A model satisfies demographic parity but violates equalized odds. Is this acceptable for a hiring tool?**
A: No. Demographic parity (equal approval rates) is the wrong metric for hiring because it ignores base rates. If qualified female candidates exist at a different rate than qualified male candidates in the applicant pool, forcing equal approval rates requires different qualification thresholds by gender — which is itself discriminatory. Equalized odds (equal true positive rate) is the correct metric: qualified candidates of all groups should have equal probability of being identified as qualified. When these metrics conflict, equalized odds takes precedence in employment contexts.

**Q2: Enforcing equalized odds requires a 7% accuracy drop. The product team says that is too high. What are the alternatives?**
A: Three alternatives in increasing order of implementation cost: (1) Calibration — post-hoc threshold adjustment that ensures predicted probabilities are equally calibrated across groups; typically < 1% accuracy cost. (2) Individual fairness — constrain the model so that similar individuals receive similar predictions (Euclidean distance in feature space); typically 1-3% cost. (3) Separate models per group — train a separate model optimized for each protected group; can improve both fairness and per-group accuracy but adds maintenance overhead. If none of these are acceptable, change the problem framing: is the target variable itself biased? Can you use a proxy outcome that is less discriminatory?

**Q3: How do you define "similar" for individual fairness in a 100-feature space?**
A: This is primarily a theoretical problem — individual fairness is rarely implemented in production because the similarity metric is application-specific and contested. The practical approaches are: (1) Reduce dimensionality to the 5-10 features most relevant to the decision and compute distance there. (2) Use a task-specific similarity metric: in a credit context, two people are "similar" if their credit history, income, and debt level are within specified tolerances, regardless of protected attributes. (3) Operationalize as a fairness constraint in training: add a regularization term that penalizes large prediction differences for pairs of examples that differ only in protected attributes. Caution: individual fairness can conflict with equalized odds — a system satisfying one may violate the other.

**Q4: Your protected attribute has three categories (male, non-binary, female) rather than two. How do you compute and report fairness?**
A: Compute all pairwise fairness gaps: male vs. female, male vs. non-binary, female vs. non-binary. Report the maximum pairwise gap as the headline fairness metric — if the maximum pairwise gap exceeds the threshold (typically 5-10%), the model fails the fairness audit regardless of which pair drives it. This approach generalizes to N categories. For N > 5, the pairwise comparison matrix becomes large; use a summary statistic (max gap, or coefficient of variation across groups) as the primary metric.

**Q5: When would you NOT optimize for a fairness metric even in a high-stakes application?**
A: When the fairness constraint conflicts with a safety constraint that has higher priority. Example: a medical diagnosis model that achieves equal TPR across demographic groups but achieves it by reducing specificity (false positive rate) to the point where healthy patients are systematically over-treated — the safety harm of over-treatment may outweigh the fairness benefit. In general, when a fairness constraint directly conflicts with patient safety, legal liability, or fraud prevention, the domain expert and legal counsel must make the trade-off decision — not the ML team unilaterally.

**Q6: How do you present the fairness-accuracy trade-off to a non-technical stakeholder?**
A: Frame it as a business decision, not a technical constraint. Show a two-axis chart: x-axis = fairness gap (demographic parity ratio or equalized odds gap), y-axis = overall model accuracy. Plot the Pareto frontier of achievable combinations. Mark the current model, the fairest achievable model, and the most accurate achievable model. Explain what the fairness gap means in concrete terms: "at this point on the curve, 1 in 8 qualified female applicants is rejected while an equivalent male applicant would be accepted." Let stakeholders choose the operating point on the frontier with full information. Document the chosen point and the rationale.

**Q7: Your fairness metrics pass at launch but start drifting after four months of production. How do you handle this?**
A: This is the "fairness drift" problem — it happens when the input distribution shifts in ways that affect groups differently. The response has three phases: (1) Alert — the monthly fairness monitoring detects the drift and triggers an investigation. (2) Diagnose — determine whether the drift is from a distribution shift in the incoming population, a change in the feature pipeline, or a change in ground-truth labels. (3) Remediate — if it is distribution shift, evaluate whether retraining on recent data with fairness constraints fixes the gap. If the ground-truth labels have shifted, the labeling process needs to be audited before retraining.

**Q8: Regulators ask you to demonstrate that your model does not discriminate. What artifacts do you provide?**
A: A complete fairness audit package includes: (1) The model card with the fairness metrics at each training and evaluation stage. (2) The evaluation dataset demographics (sample counts, label distributions, and protected attribute distributions). (3) The fairness metric time series over the model's production lifetime — showing that metrics were monitored continuously, not just at launch. (4) The rationale for the chosen fairness metric (why equalized odds rather than demographic parity, with reference to the domain regulatory standard). (5) The decision log for any fairness-accuracy trade-offs made (who decided, what options were considered, what the chosen operating point is). (6) The bias audit report from the pre-deployment review, with sign-off. In the EU AI Act context, this documentation package is legally required for high-risk AI systems.

---

## Cost & Resource Analysis

**Fairness audit:** 8-16 hours per quarter = $4-8K per year.
**Fairness-aware model retraining:** 1-2 weeks if constraint added = $5-10K per iteration.
**Legal review of fairness metrics:** 40 hours law firm = $10K.
**Regulatory compliance (hiring, lending):** dedicated team = $100K+/year.

**Cost of fairness violations:** Lawsuit $500K-5M+, reputational damage, regulatory fines 1-4% revenue.
**ROI:** Proactive fairness work $10-100K/year. Prevents incidents worth $100K-5M. Break-even: 1 incident per year.

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Fairness metrics compute | $0.001/eval | 10K evaluations/week | $40 |
| Ground-truth labeling for fairness eval | $0.15/sample | 5K samples/month | $750 |
| Fairness dashboard (build, one-time) | $200/hr | 20 hr total | $0 after build |
| Annual fairness report (engineer time) | $200/hr | 40 hr/year amortized | $666/month |
| **Total (steady-state)** | | | **~$1,456/month** |

The ground-truth labeling cost ($750/month, ~52% of total) is the central ongoing cost, and it is unavoidable for meaningful fairness monitoring — without labeled subgroup data, you cannot compute per-group precision and recall. The compute cost for fairness metrics is negligible at $40/month. The fairness dashboard is a one-time investment that pays for itself in every subsequent audit cycle. The key trade-off to communicate to stakeholders: the $1,456/month cost of proactive fairness monitoring prevents regulatory fines (1-4% of annual revenue in EU AI Act context) and litigation ($500K-5M per incident). The break-even point is roughly one month of monitoring to prevent one medium-severity regulatory finding.

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

## Interview Quick-Reference
| Use Case | Best Metric | Legal Standard |
|----------|---|---------|
| Hiring | Equalized odds | EEOC disparate impact |
| Lending | Equalized odds | Fair Housing Act, ECOA |
| Medical diagnosis | Equalized odds | No formal standard yet |
| Content ranking | Demographic parity | No formal standard |
| Criminal justice | Calibration | Contested; varies by jurisdiction |

## Related Topics
- [Bias Detection](24-bias-detection.md)
- [Interpretability](19-interpretability.md)

## Resources
- [Fairness and Machine Learning Book](https://fairmlbook.org/)
