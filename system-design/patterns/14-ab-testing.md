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

## Failure Scenarios

### Failure 1: Sample Ratio Mismatch (SRM)
**Symptom:** Treatment group is 60% of users instead of the expected 50%; chi-square test on group sizes yields p < 0.001.
**Root Cause:** Bug in cookie/session assignment logic, bot traffic inflating one bucket, or lossy event logging that drops control events more than treatment events.
**Detection:** Run a daily chi-square test on assignment counts. Flag any deviation where |actual_ratio - 0.5| > 0.02 for two consecutive days.
**Fix:** Invalidate the test immediately. Fix the assignment logic (hash by user_id modulo 2, not by session). Re-run with a clean launch. Add an SRM dashboard check to the experiment launch checklist.

### Failure 2: Novelty Effect
**Symptom:** Treatment wins convincingly in week 1 (CTR +4%, p = 0.01), then by week 4 the effect shrinks to +0.3% and is no longer significant.
**Root Cause:** Users engage more with any UI change simply because it is new. The behavior returns to baseline as the novelty fades.
**Detection:** Track week-over-week metric stability. Require that the lift standard deviation across weekly cohorts does not exceed 50% of the mean lift before calling significance.
**Fix:** Always run experiments for a minimum of two full weeks. Pre-register a "stability requirement": the effect must be significant AND stable (week-on-week std/mean < 0.5) before shipping.

### Failure 3: Multiple Testing (p-Hacking)
**Symptom:** You track 20 secondary metrics; one shows p < 0.05 and the team ships the treatment based on that single metric.
**Root Cause:** With 20 independent tests at alpha = 0.05, you expect one false positive by chance even if there is no real effect.
**Detection:** Monitor the ratio of significant metrics to total metrics tested. If it exceeds 2× the nominal alpha rate (e.g., > 10% of 20 metrics are significant), suspect p-hacking.
**Fix:** Pre-register exactly one primary metric before launching. Apply Bonferroni correction (alpha / n) for all secondary metrics. Only ship if the primary metric is significant.

### Failure 4: Interference Effects (SUTVA Violation)
**Symptom:** Control group metrics improve alongside treatment group metrics — the treatment effect appears larger than it really is.
**Root Cause:** Social or network effects: user A in treatment shares a recommendation or link with user B in control, contaminating the control.
**Detection:** Run a SUTVA violation test: compare users at the border of treatment/control clusters against users in the interior. If border control users perform closer to treatment users, interference exists.
**Fix:** Switch to cluster randomization — assign entire households, geographic regions, or social-graph communities to a single variant rather than individual users.

### Failure 5: Temporal Confounding
**Symptom:** Test starts Monday and "finishes" Friday. Results show strong lift, but the lift disappears in the following week.
**Root Cause:** Weekday behavior (high intent, higher conversion) favored the treatment which was exposed to more Monday-Wednesday traffic when the test was under-sampled.
**Detection:** Segment results by day-of-week. If lift is concentrated in specific days, confounding is present.
**Fix:** Always run tests for complete calendar weeks (7-day multiples). Consider stratifying assignment by day-of-week for experiments that run less than 14 days.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Event logging (S3) | $0.023/GB | 100 GB/day | $69 |
| Experiment infrastructure (EC2) | $0.10/hr | 720 hr/month | $72 |
| Statistical analysis compute | $0.05/run | 10 runs/day | $15 |
| Experiment analyst time | $200/hr | 4 hr/experiment | $800 (per experiment) |
| Engineer setup and launch time | $200/hr | 8 hr/experiment | $1,600 (one-time) |
| **Total per experiment** | | | **~$2,556** |

The dominant cost is engineer time, not infrastructure. At $2,500 per experiment, running 10 experiments per month costs roughly $25K — justified only if the expected lift generates more revenue than that. Use effect-size estimation and power analysis to avoid running experiments with less than 80% power to detect the business-relevant minimum detectable effect (MDE).

---

## Interview Q&A

**Q1: Sample size 500 per variant shows B better. Can you ship?**
A: Depends on effect size and baseline. If baseline accuracy 80%, 500 users gives approximately 10% power for a 1% lift. Need simulation to verify. Typical requirement: 1,000+ per variant for 5% lift detection with 80% power.

**Q2: A/B test duration — when to stop early?**
A: Use sequential testing (Wald's SPRT) if continuous monitoring is required. Otherwise, ignore early results. Early stopping inflates Type I error. The pre-registered duration is non-negotiable unless a safety issue arises.

**Q3: Two variants perform equally. How do you find a winner?**
A: Double the duration (2× users). If still equivalent after doubling, declare both variants equivalent and ship either. Alternatively, switch to a Multi-Armed Bandit to keep exploring while exploiting the best observed variant.

**Q4: A/B test on 10K users shows 0.5% lift (p = 0.04). Real impact?**
A: Statistical significance does not equal practical significance. 0.5% lift on 10M users = 50K additional conversions. At $100 revenue per conversion that is $5M/year. Practical — but the small effect size means it is fragile and could vanish with a minor product change.

**Q5: When would you NOT use a traditional A/B test?**
A: When you cannot accept the revenue loss from exposing 50% of users to a potentially worse variant (use canary at 5% instead), or when the metric takes months to observe (long-horizon outcomes like 90-day retention), or when social/network effects make user-level randomization invalid (use cluster randomization or a switchback design).

**Q6: What breaks first when your A/B testing system scales to 10× traffic?**
A: Event logging infrastructure — at 10× volume you may hit S3 PUT throttling limits or Kafka lag. Also, the statistical analysis jobs may time out if they scan the full event table rather than pre-aggregated summaries. Pre-aggregate metrics daily per variant; never re-scan raw events for reporting.

**Q7: How would you debug a situation where both variants show the same lift over the holdout?**
A: Check for a broken holdout — if holdout users are leaking into the test groups (e.g., a/b assignment logic applied post-holdout exclusion), both variants look better than holdout by identical amounts. Verify holdout assignment is applied at the very first entry point (user acquisition/cookie creation), not downstream.

**Q8: What is the trade-off between a fixed-horizon A/B test and a Bayesian adaptive design?**
A: Fixed-horizon guarantees frequentist error control (alpha, power) at a pre-registered sample size, but you must wait until the end. Bayesian adaptive designs allow early stopping with lower regret and can reallocate traffic toward the better arm, but require stronger modeling assumptions and are harder to audit for regulatory purposes. Use fixed-horizon when the test result will face external scrutiny; Bayesian when speed and regret minimization matter more.

**Q9: Model B shows 5% improvement in the lab but p-value = 0.15 in the A/B test. Ship B?**
A: No. A 15% probability this difference is noise is too high. Continue running the test to reach the pre-registered sample size, or perform a power analysis to determine if the lab effect size was overestimated.

**Q10: A/B test duration: 7 days shows B is better. Stop the test?**
A: No. Early stopping biases results. Pre-registered duration is 30 days. Continue for the full duration before deciding. The temptation to stop early is the single most common A/B testing mistake and inflates the false positive rate substantially.

## Related Topics
- [Canary Deployment](12-canary-deployment.md)
- [Model Serving](05-model-serving.md)

## Resources
- [A/B Testing Best Practices](https://en.wikipedia.org/wiki/A/B_testing)
