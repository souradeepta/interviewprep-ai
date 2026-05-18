# Fairness metrics

## TL;DR
Core ML system design pattern for production.

## Core Intuition
[Intuitive explanation]

## How It Works
[Technical details]

## Key Properties / Trade-offs
- Property 1
- Property 2

## Common Mistakes / Gotchas
- Mistake 1
- Mistake 2

## Best Practices
- Measure fairness across multiple metrics simultaneously — no single metric captures all notions of fairness
- Use disaggregated evaluation by subgroup before and after model training
- Apply fairness constraints during training when post-processing is insufficient
- Document which fairness definition was optimized and why
- Use Aequitas or Fairlearn libraries for systematic fairness auditing
- Monitor fairness metrics in production — distribution shift can re-introduce bias
- Involve domain experts and affected communities in defining fairness criteria

## Interview Q&A

**Q: When is demographic parity the appropriate fairness metric to use?**
A: Demographic parity (equal positive prediction rates across groups) is appropriate when: the selection process should be equal-opportunity regardless of group-specific base rates (e.g., hiring from historically underrepresented groups), the harm of under-selection is severe, and when you're trying to correct historical imbalances. It's inappropriate when: there are genuine group differences in the underlying construct being measured (e.g., demographic parity in medical diagnosis would mean treating equal numbers of sick and healthy people from each group regardless of disease rates).

**Q: How do you choose between equalized odds and equal opportunity as fairness criteria?**
A: Equal opportunity: only requires TPR (sensitivity) to be equal across groups—acceptable false positive rate disparity. Use when: false positives are low-harm (showing an ad to someone not interested). Equalized odds: requires both TPR and FPR to be equal—stricter, ensures errors are equally distributed. Use when: both false positives and false negatives have significant consequences (credit decisions, criminal justice). In practice, perfect equalized odds is impossible without equal base rates—choose which error type is more important and enforce equality on that.

**Q: How do you handle intersectionality in fairness evaluation?**
A: Intersectionality: disparities may exist for combinations of protected characteristics (e.g., Black women may face bias that doesn't appear when analyzing race or gender separately). Test: measure fairness metrics for all intersectional subgroups in your data. Challenge: small sample sizes for intersection subgroups reduce statistical power. Mitigation: combine rare intersection groups thoughtfully (don't merge to the point of obscuring real disparities), use bootstrap confidence intervals to communicate uncertainty, and prioritize reporting subgroups with enough data for reliable measurement.

**Q: What is calibration fairness and why does it matter?**
A: Calibration fairness: the model's predicted probabilities match actual rates equally well across groups. A model is calibrated for group A if "70% confident predictions" turn out to be correct 70% of the time for group A—same requirement for all groups. Calibration parity is important for: any application where the model's confidence score is used for decision-making (risk scores, medical probabilities), because a model that is well-calibrated overall but poorly calibrated for specific groups systematically misleads decisions for those groups.

**Q: How do you communicate fairness metrics to business stakeholders who aren't familiar with the technical definitions?**
A: Use concrete language: "The model approves loan applications from Group A at 68% and Group B at 55%—a 19% difference" rather than "Demographic parity difference is 0.13." Show impact: "This gap means approximately 3,000 additional Group B applicants per year would be approved if approval rates were equal." Frame in terms of regulatory risk: "Disparate impact ratios below 0.8 (currently 0.81) trigger regulatory scrutiny." Use comparison to human baseline: "Before the model, the human underwriter gap was 25%—the model reduced it to 19%."

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
