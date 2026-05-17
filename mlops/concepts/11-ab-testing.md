# A/B Testing & Experimentation: Validating Models with Statistical Rigor

## Definition & Why It Matters

A/B testing is a controlled experiment comparing two (or more) model versions on real production traffic. Unlike lab evaluation, A/B testing measures actual business impact with statistical rigor, accounting for natural variation.

**Why it's critical:**
- **Lab evaluation is wrong**: A model with 2% higher accuracy in testing might hurt user experience (users hate change). Lab results don't capture real-world behavior.
- **Guardrails**: A/B testing ensures you don't ship regressions. "Model A wins on accuracy, but users engage less." Catch it before full deployment.
- **Business metrics matter**: Model may be more accurate but slower (latency hurts engagement). Tests measure both.
- **Randomization prevents bias**: Non-randomized comparisons are unreliable. (If you give new model to power users only, of course it performs better.)

Netflix found that most models that won in lab evaluations hurt engagement in production. A/B testing is how they avoid shipping those.

---

## How It Works

### The Experimentation Process

```
Design experiment
    ↓
Run hypothesis test (power analysis)
    ↓
Randomize users into control/treatment
    ↓
Run experiment for N days (enough data)
    ↓
Compute p-value & confidence interval
    ↓
Make decision (ship, iterate, or discard)
```

### Key Components

**1. Hypothesis**
- Null: new model ≠ baseline (no difference)
- Alternative: new model > baseline (directional: must improve)
- Importance: guides sample size calculation, decision rules

**2. Sample Size Calculation**
- Need enough samples to detect effect if it exists
- Formula: depends on baseline metric, expected lift, confidence level
- Example: "Detect 2% improvement in CTR with 95% confidence: need 5M impressions"

**3. Randomization**
- Users/sessions randomly assigned to control/treatment
- Ensures differences aren't confounding (e.g., new model didn't just get lucky with good users)
- Methods: cookie-based, user-based, or session-based

**4. Metrics**
- **Primary metric**: what are you optimizing? (CTR, conversion, engagement)
- **Secondary metrics**: guardrails (latency, load)
- **Guardrail metrics**: don't ship if any degrades >1% (e.g., don't increase latency)

**5. Duration**
- Must run long enough to collect sample size
- Accounts for day-of-week effects (run ≥7 days)
- Seasonal effects (may need 2+ weeks)

**6. Analysis**
- Compute p-value (is difference statistically significant?)
- Compute 95% CI (expected improvement range)
- Success: p < 0.05 and CI doesn't include 0

---

## Interview Q&A: A/B Testing

### Q1: "When should you A/B test vs deploy immediately?"
**Answer outline:** Deploy immediately only for:
1. **Obviously better (deterministic)**: latency reduction, crash fix, bug fix
2. **Internal tooling**: backend changes users don't interact with
3. **Rollback tested**: known good previous version available if needed

A/B test for:
1. **Model changes**: new model architecture, retraining approach
2. **Feature changes**: new features affecting user experience
3. **Uncertain business impact**: model better in lab, but real-world impact unknown

Example: Deploy tokenizer fix immediately (no risk). A/B test new recommendation model (risky, unknown business impact).

### Q2: "You need 10M samples to detect 2% lift. Users = 50M/day. How long?"
**Answer outline:** Sample size calculation:
- 50M users/day × 50% in treatment = 25M treatment samples/day
- Need 10M samples → 10M / 25M = 0.4 days ≈ 10 hours

But wait: account for day-of-week effects (running only 10 hours misses weekly patterns). Solution: run ≥7 days minimum, even if sample size achieved earlier.

Example: 50M users/day, 7 days = 350M samples, easily >10M. Statistical significance reached after day 2, but run full 7 days to account for seasonality.

### Q3: "New model wins on primary metric (+3% CTR), but latency increased by 10%. Ship?"
**Answer outline:** Depends on guardrail thresholds:
1. **Guardrail for latency**: "Don't ship if latency increases >5%"
   → 10% increase violates guardrail. Don't ship.
2. **If guardrail was "don't ship if latency increases >15%"**:
   → 10% is acceptable. Ship, but monitor.

Trade-off analysis:
- 3% more clicks (good) vs 10ms slower (acceptable for most use cases)
- If users care about speed, latency increase hurts engagement even if CTR higher
- Solution: optimize model for latency (quantization, distillation) before shipping

Example: LinkedIn tested notification ranking, won on CTR, but engagement latency increased. Users disliked slower notifications. Didn't ship until optimized.

### Q4: "A/B test shows p=0.08 (not quite significant). What do you do?"
**Answer outline:** p=0.08 means: 8% chance you'd see this result if models were actually equal (threshold is 5%). Options:
1. **Extend test duration**: Collect more samples, increase power to reject null hypothesis
2. **Accept it's inconclusive**: Model probably not better, retry next month with improvements
3. **Dig deeper**: Maybe works better for specific user cohorts. Analyze subgroups.

Temptation: "Almost significant, let's ship anyway." Don't. p=0.08 is not statistically significant. Requires >0.05 iterations to ship.

Example: Email redesign shows p=0.08 improvement. Don't ship yet. Run for another week. If p-value improves to <0.05, ship. If not, revisit design.

### Q5: "Design A/B test for model serving latency optimization. 100K pred/sec."
**Answer outline:** Latency optimization (e.g., model quantization) requires careful testing:
1. **Primary metric**: p99 latency (what users experience)
2. **Secondary metrics**: accuracy (ensure quantization doesn't hurt too much), throughput (queries/sec)
3. **Guardrails**: don't ship if accuracy drops >0.1%
4. **Sample size**: latency is noisy. Need 24 hours to capture peak traffic patterns
5. **Analysis**: p99 latency should improve (e.g., 120ms → 95ms), accuracy should match

Challenge: latency A/B tests are hard because different times have different baselines. Solution: stratify by time-of-day, or run continuous test (all traffic sees both models in rotation).

Example: Quantized model shows 95ms latency vs 120ms baseline, accuracy 95.2% vs 95.8% (acceptable). Ship after 24-hour test.

---

## Best Practices

1. **Always set guardrails**: Define metrics you must not regress. Latency, error rate, engagement.

2. **Power analysis before starting**: Calculate sample size needed. Don't run "until significant."

3. **Lock analysis plan before starting**: Decide hypothesis, metrics, sample size upfront. Prevents p-hacking.

4. **Run for at least 7 days**: Accounts for day-of-week effects, seasonality.

5. **Stratify if possible**: Different user segments may have different responses. Subgroup analysis reveals this.

6. **Statistical significance ≠ business significance**: p < 0.05 means likely true, not necessarily worth shipping. 0.1% improvement in 10M users = significant but tiny ROI.

7. **Monitor post-deployment**: A/B test says "worked in lab," but monitoring catches real-world surprises (e.g., bugs specific to certain user profiles).

8. **Iterate, don't hoard improvements**: Ship winning models quickly, learn from failures.

9. **Use sequential analysis if available**: Can stop early if effect is larger than expected (save time/compute).

10. **Document all tests**: Create record of what you tried, results, learnings. Prevents duplicate work.

---

## Common Pitfalls

1. **p-hacking**: Run test, p=0.08, adjust parameters, run again, p=0.03. Multiple attempts invalidates p-value. Lock plan upfront.

2. **Peeking at results**: "Results are in, can we stop early?" Early stopping invalidates p-value. Commit to duration.

3. **Confounding variables**: Ran test during holiday weekend. Results don't generalize. Account for seasonality.

4. **Ignoring guardrails**: Model wins on CTR but latency increased. Shipped anyway. Users hated it. Don't ignore guardrails.

5. **Sample contamination**: Users in control group saw treatment, or vice versa. Results biased.

6. **No subgroup analysis**: Model works overall but fails for minority group. Didn't test subgroups. Always stratify.

7. **Shipping without follow-up monitoring**: A/B test passed. Shipped. Didn't monitor. Bugs discovered by users weeks later.

8. **Underpowered test**: Ran for 3 days. Need 7+ for statistical validity. P-value might be 0.08 (close) but invalid because duration too short.

9. **Effect size too small to matter**: P < 0.05 improvement of 0.01%. Statistically significant but business irrelevant.

10. **No holdout control**: Comparing variant to baseline, but baseline changed over time. Results confounded.

---

## Real-World Examples

### Example 1: Netflix Recommendation A/B Testing
Netflix runs 100s of recommendation experiments simultaneously.
- **Hypothesis**: New deep learning model improves watch hours
- **Sample size**: 5M users (control), 5M users (new model)
- **Duration**: 2 weeks (capture weekly patterns)
- **Primary metric**: watch hours/user (+1% = $50M/year revenue)
- **Guardrails**: don't increase abandonment rate, don't decrease diversity

Result: New model +3% watch hours, +1% abandonment, +5% diversity. Watch hours improvement worth shipping, monitor abandonment.

### Example 2: Stripe Fraud Model Testing
Stripe A/B tests fraud models carefully (false declines cost revenue).
- **Hypothesis**: New model catches more fraud without increasing false declines
- **Sample size**: 10M transactions (control), 10M (new model)
- **Duration**: 1 month (fraud patterns vary weekly)
- **Primary metric**: fraud detection rate (minimize loss)
- **Guardrails**: don't increase false decline rate >1% (hurts revenue)

Result: New model detects 2% more fraud, false declines up 0.5% (acceptable). Deploy incrementally.

### Example 3: Uber ETA Model Canary Test
Uber tests new ETA models with careful monitoring.
- **Hypothesis**: New model improves prediction accuracy
- **Canary phase**: 5% of users, 24 hours
- **Metrics**: prediction error, request latency, user complaints
- **Guardrail**: latency must stay <100ms

Result: New model MAE down 5%, latency stable, no complaints. Roll out to 25% for 3 days, then 100%.

---

## Sample Interview Case Study

**Scenario:** Netflix recommends new ranker model. Lab tests: +2% click-through rate. Design A/B test.

**Solution:**

1. **Hypothesis**: New ranker improves user engagement (watch hours)
2. **Sample size**: Netflix has 250M subscribers. Test on 10M (control), 10M (new model). Need 2 weeks to capture weekly patterns.
3. **Primary metric**: watch hours per user (main business metric)
4. **Secondary metrics**: engagement (whether recommendations are clicked), diversity (% of recommendations outside top 100 shows)
5. **Guardrails**: engagement ≥ baseline, diversity ≥ baseline
6. **Implementation**: Randomize at user level (user always sees same model, consistent experience). Cookie-based randomization.
7. **Analysis**: Compute p-value for watch hours metric. If p < 0.05, check guardrails. If all pass, deploy to 50% traffic, monitor 3 days, expand to 100%.

**Strong answer:** "Design A/B test: hypothesis is watch hours increase. Sample: 10M control vs 10M treatment for 2 weeks (capture weekly patterns). Primary metric: watch hours. Guardrails: engagement and diversity don't decrease. Randomize at user level for consistency. Analyze: p-value for watch hours, confidence interval for guardrails. Only ship if primary metric significant and all guardrails pass."

---

## Key Takeaways

A/B testing is the gold standard for model validation. Lab tests can be misleading; production A/B tests reveal true business impact.

**Experimentation pipeline:** hypothesis → sample size → randomization → run → analysis → decision → monitor

**Common interview pattern:** "Model is better in testing. Why A/B test?" → Answer: "Lab results often don't reflect production. A/B testing with randomization and statistical rigor reveals true impact on users and business."

---

## Related Concepts

- **Model Testing** (Concept 09): Lab evaluation before A/B test
- **Evaluation Metrics** (Concept 12): Metrics used in A/B tests
- **Canary Deployment** (Concept 16): Deployment strategy using A/B test framework
- **Monitoring** (Concept 18): Continuous monitoring post-A/B test deployment
