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

**Failure 1: Aggregate Metrics Hide Subgroup Disparity**
- **Symptom:** Model reports overall accuracy of 92% in the pre-launch audit. After deployment, a regulatory audit discovers that accuracy for the subgroup (female, age 65+) is 71% — a 21-point gap that triggers a compliance finding.
- **Root cause:** The pre-launch audit reported only aggregate metrics. No one broke down accuracy by protected attribute combinations.
- **Detection:** Break down all accuracy and recall metrics by protected attribute combinations (age × gender × race). Establish a minimum per-subgroup threshold (e.g., no subgroup should fall below 80% of the overall metric) and make it a deployment gate.
- **Fix:** Never report only aggregate metrics for any model in a regulated or high-stakes context. Require per-subgroup breakdown as part of every model card. Gate deployment on the minimum-subgroup threshold.

**Failure 2: Proxy Discrimination Undetected**
- **Symptom:** The model does not use any protected attributes as inputs. The bias audit passes. Two years post-launch, a journalist's analysis reveals that zip code is the strongest predictor and correlates 0.71 with race, producing a redlining effect.
- **Root cause:** Protected attributes were removed but highly correlated proxy features (zip code, census tract, last name) were retained and heavily used.
- **Detection:** Compute the Pearson correlation between every input feature and every protected attribute. Flag any feature where |r| > 0.3 for manual review.
- **Fix:** Drop or transform features with high protected-attribute correlation. Document the decision. For features that cannot be removed (e.g., zip code is genuinely predictive of default risk), use a fairness-aware training objective that penalizes reliance on proxy features.

**Failure 3: Historical Bias in Labels**
- **Symptom:** A loan default model trained on 10 years of historical data produces lower approval rates for minority applicants. The model has no explicit protected attribute features and passes all standard fairness audits.
- **Root cause:** The training labels (historical defaults) reflect past discriminatory lending practices. Minority applicants were systematically denied loans at higher rates historically, creating a feedback loop where limited credit history generates biased labels.
- **Detection:** Analyze the label distribution by protected group and compare to population base rates. If historical approval rates for a group are far below their base-rate default risk, the labels are likely biased.
- **Fix:** Use counterfactual fairness techniques, importance reweighting to correct for historical underrepresentation, or recalibrate labels using external audit data. Document the limitation prominently in the model card.

**Failure 4: Bias Introduced by Data Augmentation**
- **Symptom:** A skin condition detection model achieves 91% accuracy overall but only 74% accuracy on darker skin tones (Fitzpatrick Scale V-VI) after a new augmentation pipeline is introduced to improve generalization.
- **Root cause:** The new augmentation (skin tone normalization) was calibrated on the majority skin tone in the training set. The normalization degrades the signal for darker tones rather than normalizing it.
- **Detection:** Run accuracy by Fitzpatrick Scale subcategory before and after any augmentation pipeline change. Treat per-skin-tone accuracy as a gating metric.
- **Fix:** Test the impact of each augmentation on every demographic subgroup before applying globally. If an augmentation improves aggregate accuracy but reduces per-subgroup accuracy below the minimum threshold, reject it.

---

## Implementation Guidance

**Wrong:** Check fairness once before deployment. Assume it stays fair.
**Right:** Monthly fairness audits. Monitor cohort-specific metrics continuously. Alert on fairness gap >5%.

**Wrong:** Use single fairness metric (e.g., demographic parity). Sufficient.
**Right:** Use multiple metrics (demographic parity + equalized odds). Understand trade-offs.

---

## Interview Q&A

**Q1: Your model shows an 8% fairness gap (male 95%, female 87% accuracy). How do you approach fixing it?**
A: Diagnose before fixing. The gap could be from data imbalance (fewer female training examples), biased features (name or title encoding gender), or a biased label generation process. Check each: (1) Compare class-conditional sample counts for male vs. female in the training set. (2) Compute the correlation between model features and gender. (3) Analyze whether labels themselves show historical bias. Once you identify the root cause, fix accordingly: oversample or synthetically augment underrepresented examples, remove or decorrelate biased features, or apply importance reweighting. Measure the gap after each intervention — the target is < 2%, not "better than before."

**Q2: The fairness metric reports the model is fair, but affected groups report systematic unfairness. Which do you trust?**
A: Both signals matter. Metrics measure what they measure — demographic parity says approval rates are equal, but it says nothing about whether the threshold is calibrated fairly, whether the feature distribution differs between groups, or whether the error types differ. The qualitative feedback from affected groups tells you what dimension of fairness your metrics aren't capturing. Investigate the gap: ask what specifically feels unfair, map that to a quantitative metric, and check whether that metric shows a problem. Metrics are necessary but not sufficient; stakeholder feedback provides the hypothesis space for which metrics to compute.

**Q3: Base rates are genuinely different between groups — male and female fraud rates differ by 12%. Should you enforce equal TPR (equalized odds)?**
A: Verify whether the base rate difference is legitimate or an artifact of historical bias before deciding. If the difference reflects genuine behavioral differences (confirmed by multiple independent data sources and domain experts), forcing equal TPR is mathematically incorrect — it requires different decision thresholds per group, which means different standards for the same outcome. If the difference is an artifact of past discriminatory enforcement (e.g., one group was over-policed), the base rate itself is biased and equalized odds may still be appropriate. Document your reasoning either way.

**Q4: Two fairness metrics contradict each other — demographic parity passes but equalized odds fails. Which do you enforce?**
A: The metric should match the legal and ethical definition of fairness for your specific use case. Lending and hiring: use equalized odds (equal opportunity — same TPR across groups is the legal standard in most jurisdictions). Recommendations and content: demographic parity is sufficient (equal outcome rate, no legal standard for equal opportunity). Criminal justice risk scoring: calibration (the probability score must be equally accurate across groups). When metrics conflict, the tie-breaker is what the domain's regulatory framework requires, not which metric your model performs better on.

**Q5: When would you NOT run a bias detection audit before deployment?**
A: For a model serving only internal users (employees), where the model cannot affect protected groups' access to services, credit, employment, or housing. For a model that produces no individualized decisions (e.g., a model that estimates total inventory demand for a warehouse — the output is not applied to individuals). For a pure research prototype with a documented two-week lifecycle and no production pathway. In all other cases — any model that affects the public or makes decisions about individuals — the audit is mandatory.

**Q6: How do you detect bias that emerges after deployment (not at launch)?**
A: Continuous monitoring with per-cohort metrics. The implementation has three components: (1) Segment all incoming prediction requests and outcomes by protected attribute (or a reasonable proxy). (2) Compute fairness gap metrics weekly on the rolling 30-day window of live predictions. (3) Alert if any fairness metric changes by more than 5% month-over-month. The fairness gap most commonly grows due to distribution shift (new cohorts entering the population), model retraining on biased new data, or feature pipeline changes that affect groups differently. The alert tells you to investigate; the weekly monitoring data tells you when the shift started.

**Q7: Your retraining pipeline inadvertently introduced bias — accuracy for one group dropped 8% but the model passed all aggregate quality gates. How do you prevent this?**
A: The root cause is that aggregate gates don't catch subgroup regressions. The fix is to add per-subgroup accuracy to the deployment gate: for every protected attribute, the new model must not regress more than 3% from the previous model's per-subgroup accuracy. This requires labeled subgroup ground truth in your evaluation set — without it, you cannot compute per-subgroup metrics. The evaluation set must oversample minority groups to have enough statistical power to detect a 3% shift.

**Q8: You're asked to build a bias detection system for 20 models across three business lines. How do you architect it?**
A: Centralize the computation and standardize the contract. (1) A shared fairness monitoring service that all models send predictions to (along with demographic metadata where legally permissible). (2) A standard fairness metric report format: demographic parity ratio, equalized odds gap, and at least two intersectional combinations. (3) A model-specific configuration layer that maps each model's protected attributes, the appropriate fairness metrics for its domain, and the alert thresholds. (4) A quarterly audit report that aggregates across all models and identifies systemic patterns. The key architectural decision is where demographic data lives — in some jurisdictions, storing protected attributes alongside predictions is restricted, so the fairness monitoring service may need to compute metrics at collection time and discard the raw data.

---

## Cost & Resource Analysis

**Fairness audit:** Manual slice analysis 8 hours/month = $4K. Automated tools (Fairness Indicators, AI Fairness 360): $500-2K/month.
**Retraining for fairness:** 1-2 weeks engineer time = $5-10K. May reduce overall accuracy.
**Litigation cost if bias discovered:** $100K-10M depending on domain and damage.

**ROI:** Proactive fairness audit cost $500-4K/month. Prevents litigation worth $100K+. Break-even: 1 lawsuit prevented per year.

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Bias audit (external consultant) | $10K/audit | 1 per quarter | $3,333 amortized |
| Subgroup ground-truth labeling | $0.10/sample | 50K samples | $5,000 |
| Fairness monitoring compute | $0.50/hr | 20 hr/month | $10 |
| Legal/compliance review | $300/hr | 4 hr/month | $1,200 |
| **Total** | | | **~$9,543/month** |

The labeling cost dominates at $5,000/month (~52% of total). This is the cost of maintaining labeled subgroup ground truth for ongoing fairness monitoring. It is non-negotiable for regulated contexts (lending, hiring, healthcare) — without labeled subgroup data, fairness monitoring degenerates to proxy metrics. The legal review at $1,200/month is a fixed cost that grows with the number of models in production. Organizations can reduce the external audit cost from $3,333/month by building internal fairness audit capability, but this requires a dedicated fairness engineer (~$20K/month loaded). The break-even point is roughly 6 models in production: at that scale, internal capability is cheaper than quarterly external audits.

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

## Interview Quick-Reference
| Metric | Definition | Best For |
|--------|---|---------|
| Demographic parity | Approval rate same for all groups | Recommendations, low-stakes |
| Equalized odds | TPR same for all groups | Hiring, lending, healthcare |
| Predictive parity | Precision same for all groups | Different base rates |
| Calibration | P(correct\|score) same across groups | Criminal justice, critical decisions |

## Related Topics
- [Fairness Metrics](25-fairness-metrics.md)
- [Model Debugging](17-model-debugging.md)

## Resources
- [Fairness and Machine Learning](https://fairmlbook.org/)
