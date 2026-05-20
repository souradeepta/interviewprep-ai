# A/B Testing

## TL;DR
Run two model versions (A=baseline, B=treatment) on subset of real users. Measure: accuracy, latency, user satisfaction. Statistical significance matters—need 1000+ users minimum.

## Core Intuition
Labs show model B is 5% better. Real users show 1% better (noise). A/B test with real users determines if improvement is real (p<0.05) or noise.

## How It Works

1. Allocate: 50% users → model A, 50% → model B
2. Log: for each user, which model they saw, outcome
3. Measure: accuracy, CTR, conversion, latency
4. Analyze: significance test (chi-square, t-test)
5. Decision: if B is statistically better, promote B

| Metric | Model A | Model B | Lift | P-value |
|--------|---------|---------|------|---------|
| Accuracy | 0.923 | 0.928 | +0.54% | 0.08 |
| CTR | 3.2% | 3.1% | -0.31% | 0.42 |

p-value > 0.05 = no significant difference. Need longer test.

## Key Properties / Trade-offs
- Duration: longer = more reliable (but slow to ship)
- Sample size: bigger = more power (but cost increases)
- Multi-armed bandits: adaptive allocation (vs fixed 50/50)

## Detailed Trade-off Analysis

| Aspect | A/B Test (Fixed) | Multi-Armed Bandit | Canary Deployment | Shadow Mode |
|--------|-----------------|------------------|-------------------|-------------|
| Sample size per variant | 1000+ | Adaptive (200-500) | N/A | N/A |
| Duration | 7-30 days | 1-3 days | Gradual hours | Continuous |
| Revenue loss (bad variant) | 50% of users × loss% | 10% of users × loss% | 5% of users × loss% | 0% (offline) |
| Statistical rigor | Highest | Medium | Low | N/A |
| Infrastructure | Simple event logging | Adaptive allocation | Traffic splitting | Parallel models |

**Decision:** Need statistical proof → A/B test. Confident model → canary. Risky changes → shadow. Fast iteration → MAB.

---

## Production Failure Scenarios

**Scenario 1: P-hacking (multiple comparisons bias)**
- Run 20 metrics, find 1 with p<0.05 by chance. Report as significant.
- Fix: Pre-register metrics before test. Bonferroni correction (p<0.05/N).

**Scenario 2: Sample ratio mismatch (SRM)**
- Expected 50/50 split, actual is 49/51. Confounds results.
- Fix: Validate allocation before analyzing. Use Chi-square SRM test.

**Scenario 3: Network effects (spillover)**
- Model B users share results with model A users. Control group contaminated.
- Fix: Isolate test cohorts geographically or by user network.

**Scenario 4: Temporal confounding**
- Test runs Mon-Fri (high activity) but ends on weekend. Day-of-week effect confounds lift.
- Fix: Run full weeks, account for seasonality in analysis.

---

## Implementation Guidance

**Wrong:** Run test until B looks good, stop early, report results.
**Right:** Pre-register sample size and duration. Analyze at end. Use sequential testing if continuous monitoring needed.

**Wrong:** Track 50 metrics, report the significant ones.
**Right:** Pre-register primary and secondary metrics. Apply multiple-comparisons correction.

---

## Sophisticated Interview Q&A

**Q1: Sample size 500 per variant shows B better. Can you ship?**
A: Depends on effect size and baseline. If baseline accuracy 80%, 500 users gives ~10% power for 1% lift. Need simulation to verify. Typical: 1000+ per variant for 5% lift detection.

**Q2: A/B test duration: when to stop early?**
A: Use sequential testing (Wald's SPRT) if continuous monitoring required. Otherwise, ignore temptation. Early stopping raises Type I error. Pre-registered duration non-negotiable.

**Q3: Two variants perform equally. How extend to find winner?**
A: Double duration (2x users). If still equal after doubling, both are equivalent (good!). Ship either. Or switch to MAB to explore further while exploiting best.

**Q4: A/B test on 10K users shows 0.5% lift (p=0.04). Real impact?**
A: Statistical significance ≠ practical significance. 0.5% lift on 10M users = 50K additional conversions. Revenue impact: 50K × $100 = $5M/year. Practical yes, but effect is small—watch for maintenance burden.

---

## Cost & Resource Analysis

**Sample size planning:**
- Small effect (0.5% lift): 10K users per variant, 30 days
- Medium effect (2% lift): 2.5K users per variant, 14 days
- Large effect (5% lift): 1K users per variant, 7 days

**Revenue impact (opportunity cost):**
- Bad variant serving 50% of traffic × cost of delay = significant loss
- Example: $1M/day revenue, 2% accuracy drop, 30-day test = $600K loss
- A/B testing investment justified if variants differ by >1%

**Infrastructure cost:** Minimal (event logging + analytics). ~$1K/month for analysis infrastructure.

---

## Monitoring & Observability

**Key metrics:** Sample ratio (should be 50/50), CTR by variant, accuracy by variant, p-value trajectory, confidence interval width

**Alerts:** SRM detected (>2% deviation), p-value inflates (Type I error), variant crashes (error rate >1%)

## Common Mistakes / Gotchas
- p-hacking: run many tests, report only significant ones (false positive)
- Sample size too small: 100 users → wide confidence intervals
- Confounding variables: day-of-week effect masked by model difference
- Early stopping: stop test early if B looks good (skews p-value)

## Best Practices
- **Pre-register:** decide sample size, duration, metric before test
- **Multiple metrics:** track both lift + safety (false positive rate)
- **Minimum sample:** 1000 users per variant (30-day test typical)
- **Ignore early results:** don't stop at day 7 of 30-day test
- **Holdout group:** always keep model A for baseline monitoring

## Code Example
```python
from scipy.stats import chi2_contingency

def ab_test_significance(model_a_correct, model_a_total,
                         model_b_correct, model_b_total):
    # Contingency table
    contingency = [
        [model_a_correct, model_a_total - model_a_correct],
        [model_b_correct, model_b_total - model_b_correct]
    ]
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    a_accuracy = model_a_correct / model_a_total
    b_accuracy = model_b_correct / model_b_total
    lift = (b_accuracy - a_accuracy) / a_accuracy
    
    return {
        "a_accuracy": a_accuracy,
        "b_accuracy": b_accuracy,
        "lift": lift,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
```

## Interview Q&A
**Q: Model B shows 5% improvement in lab but p-value=0.15 in A/B test. Ship B?**
A: No. 15% chance this difference is noise. Keep running test or use model A. Statistical significance is more important than magnitude of lift in real-world decisions.

**Q: A/B test duration: 7 days shows B is better. Stop test?**
A: No. Early stopping biases results. Pre-registered duration is 30 days. Continue for full duration before deciding. Temptation to stop early is the biggest A/B testing mistake.

## Interview Quick-Reference
| Sample Size | Duration | Confidence |
|-------------|----------|-----------|
| 100 per variant | 1 day | Low (wide CI) |
| 1000 per variant | 7 days | Medium |
| 10K per variant | 30 days | High |

## Related Topics
- [Canary Deployment](12-canary-deployment.md)
- [Model Serving](05-model-serving.md)

## Resources
- [A/B Testing Best Practices](https://en.wikipedia.org/wiki/A/B_testing)
