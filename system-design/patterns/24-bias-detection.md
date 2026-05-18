# Bias detection

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
- Define protected attributes and fairness criteria before building models
- Measure bias on held-out test set, not training data
- Disaggregate metrics by subgroup combinations (intersectionality), not just single attributes
- Use statistical tests to determine if performance differences are significant
- Run bias evaluation as part of CI/CD pipeline — not one-time audits
- Monitor bias metrics in production — distribution shift causes bias to reappear
- Involve domain experts in interpreting bias metrics — not all disparities are problematic

## Interview Q&A

**Q: What is the difference between disparate impact and disparate treatment in ML models?**
A: Disparate treatment: using a protected characteristic (race, gender) directly as a model feature—illegal in most jurisdictions for consequential decisions. Disparate impact: the model doesn't use protected characteristics directly but produces significantly different outcomes for protected groups—can also be illegal even without discriminatory intent. Test for both: check which features the model uses, and separately measure outcomes across demographic groups. A model can have disparate impact even with no disparate treatment if proxy variables correlate with protected characteristics.

**Q: How do you measure bias in a model when you don't have demographic labels?**
A: Proxy inference: use name-based ethnicity inference (Bayesian Improved Surname Geocoding), zip code as a proxy for race/income, gender inference from name. These proxies are imperfect but can detect gross disparities. Audit vendor: hire a third-party audit firm with specialized bias detection tools. Analyze proxy features: if the model heavily weights zip code, investigate whether that creates disparate impact. Test with synthetic data: create matched pairs that differ only in demographic-correlated attributes and measure prediction differences.

**Q: What are the fundamental trade-offs between different fairness metrics?**
A: It is mathematically proven that you cannot simultaneously satisfy demographic parity (equal positive prediction rates across groups), equalized odds (equal TPR and FPR across groups), and predictive parity (equal precision across groups) unless base rates are equal across groups (which they rarely are). Choose the fairness metric that aligns with your use case: criminal justice—prefer equalized odds (equal error rates). Hiring—prefer demographic parity (equal opportunity). Medical diagnosis—prefer predictive parity (equal reliability of positive prediction).

**Q: How do you implement a bias monitoring system for production models?**
A: Compute fairness metrics (demographic parity ratio, equalized odds difference, disparate impact) on a rolling window of recent predictions. Compare against: legal thresholds (4/5 rule: adverse impact ratio <0.8 triggers investigation), historical baselines, and peer models. Alert when: fairness metrics degrade significantly, prediction volume for specific groups changes (may indicate distribution shift), or outcomes for groups diverge. Store all bias metrics with the same rigor as accuracy metrics—they're equally important for responsible deployment.

**Q: What interventions can you use to mitigate bias in a deployed model?**
A: Pre-processing: rebalance training data, remove or transform biased features. In-processing: add fairness constraints to the loss function (adversarial debiasing, fairness regularization). Post-processing: adjust decision thresholds per demographic group to equalize outcomes. Monitoring: implement feedback loops to detect bias creep after retraining. Choosing between them: post-processing is fastest to implement but may reduce overall accuracy; pre-processing addresses root causes but requires new training. Start with post-processing to demonstrate the fix is feasible, then address root causes through data and training changes.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
