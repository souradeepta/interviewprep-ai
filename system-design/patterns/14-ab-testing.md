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
