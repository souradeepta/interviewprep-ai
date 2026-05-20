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

## Detailed Trade-off Analysis

| Fairness Metric | Definition | When to Use | Drawback |
|-----------------|-----------|-----------|----------|
| Demographic parity | Approval rate equal across groups | Low-stakes (recommendations) | Ignores base rates |
| Equalized odds | TPR equal across groups | Medium-stakes (hiring, loans) | May reduce overall accuracy |
| Predictive parity | Precision equal across groups | Different base rates OK | Complex to enforce |
| Calibration | P(correct\|score) same across groups | Critical (criminal justice) | Requires ground truth |
| Individual fairness | Similar decisions for similar people | Legal defense needed | Vague definition |

**Decision:** Loans/hiring → equalized odds. Recommendations → demographic parity. Criminal justice → calibration.

---

## Production Failure Scenarios

**Scenario 1: Fairness metric improved, but unfair outcomes increased**
- Enforce demographic parity (equal approval rates). Males 50% approval, Females 50%. But male threshold lower (30% confidence), female threshold higher (70% confidence).
- Females need higher confidence to get approved = unfair.
- Prevention: Use equalized odds (equal TPR). Or: calibrate thresholds, don't just enforce rates.

**Scenario 2: Proxy variables bypass fairness constraint**
- Remove gender feature. Model learns gender from zip code (correlated). Gender gap persists.
- Fix: Identify and remove all proxy variables. Or: use adversarial debiasing (train model to NOT predict gender from features).

**Scenario 3: Fairness audit only quarterly, bias introduced in between**
- Monthly data drift changes cohort balance. Fairness gap increases 5% each month, discovered quarterly (15% gap).
- Fix: Monthly fairness audits. Real-time monitoring of cohort-specific metrics.

**Scenario 4: Fairness trade-off with overall accuracy**
- Enforce fairness, overall accuracy drops 5%. Business complains about reduced performance.
- Root cause: bias was helping overall accuracy (overrepresented group had easier patterns).
- Fix: Separate models per cohort. Or: accept accuracy drop as cost of fairness.

---

## Implementation Guidance

**Wrong:** Check fairness once before deployment. Assume it stays fair.
**Right:** Monthly fairness audits. Monitor cohort-specific metrics continuously. Alert on fairness gap >5%.

**Wrong:** Use single fairness metric (e.g., demographic parity). Sufficient.
**Right:** Use multiple metrics (demographic parity + equalized odds). Understand trade-offs.

---

## Sophisticated Interview Q&A

**Q1: Model has 8% fairness gap (male 95%, female 87%). How fix?**
A: (1) Root cause: is it data imbalance or biased features? (2) If data: collect more female examples, retrain. (3) If features: identify biased features (e.g., name → gender), remove or decorrelate. (4) If model: use fairness-aware training (add fairness constraint to loss). (5) Measure: gap should decrease to <2%.

**Q2: Fairness metric says "fair" but qualitative feedback says "unfair". Trust metric?**
A: Metrics can be gamed or context-insensitive. (1) Investigate: what does "unfair" mean to complainants? (2) Check if metric captures that dimension. (3) Use multiple metrics. (4) Involve stakeholders in defining fairness. (5) Metrics + human judgment required.

**Q3: Equalized odds says "not fair" (10% TPR gap). But base rates different—should we force equal TPR?**
A: Equalized odds assumes equal base rates is OK. If base rates truly different (e.g., male and female fraud rates differ), forcing equal TPR might be unfair. (1) Verify base rates are not due to past bias. (2) If legitimate, accept TPR difference. (3) Monitor: does lower TPR group notice unfair treatment?

**Q4: Two fairness metrics contradict (demographic parity says fair, equalized odds says not). Which to enforce?**
A: Context-dependent. (1) Loans: use equalized odds (care about equal opportunity). (2) Recommendations: demographic parity OK (no high stakes). (3) Criminal justice: calibration (accuracy across all groups). (4) If conflict, prioritize metric that matches legal/stakeholder expectations. Document trade-off.

---

## Cost & Resource Analysis

**Fairness audit:** Manual slice analysis 8 hours/month = $4K. Automated tools (Fairness Indicators, AI Fairness 360): $500-2K/month.
**Retraining for fairness:** 1-2 weeks engineer time = $5-10K. May reduce overall accuracy.
**Litigation cost if bias discovered:** $100K-10M depending on domain and damage.

**ROI:** Proactive fairness audit cost $500-4K/month. Prevents litigation worth $100K+. Break-even: 1 lawsuit prevented per year.

---

## Monitoring & Observability

**Key metrics:** Fairness gap (max cohort difference), TPR/FPR by cohort, precision by cohort, demographic parity ratio, proxy variable correlation, audit completion rate

**Alerts:** Fairness gap exceeds 5%, TPR drops for any cohort, new proxy variable discovered (correlated with protected attribute), fairness audit overdue (>30 days)

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
