# A/B Testing

## TL;DR
Experiment framework: split traffic, run model A vs B, compare metrics. Measure impact (accuracy, revenue, latency) with statistical rigor. Essential for production ML.

## Core Intuition
Claim: "Model B is better." Proof: run on real users, measure. Can't just test on historical data (data drift, selection bias).

## How It Works
```
Traffic split: 50% old model (A), 50% new model (B)
Measure: accuracy, latency, revenue per user
Duration: 1-4 weeks (enough samples for significance)
Stats test: t-test or Chi-square
Result: reject or accept improvement
```

**Example:**
- Baseline: model A, 92% accuracy
- Candidate: model B, 93% accuracy
- Sample size: 100k users per variant
- Result: p<0.05, B significantly better
- Decision: deploy B

## Key Properties / Trade-offs
- Statistical power: more samples = confidence, but slow
- Business impact: measure what matters (revenue, not just accuracy)
- Interaction effects: some users benefit, others harmed (subgroup analysis)

## Common Mistakes / Gotchas
- **Low sample size:** high variance, false negatives
- **Peeking:** stop test early if result looks good → inflated p-values
- **Multiple comparisons:** test 10 variants → expect ~1 false positive
- **Not accounting for novelty:** users try new UI, brief boost, fades
- **Wrong metric:** optimize accuracy, break user experience. Measure real goals.

## Interview Q&A

**Q: How do you calculate the required sample size for an A/B test on a model?**
A: Sample size depends on: minimum detectable effect (how small an improvement matters to the business), baseline metric value and variance, desired statistical power (typically 80%), and significance level (typically 5%). Use a power calculator: n = (z_alpha/2 + z_beta)^2 x 2sigma^2 / delta^2 where delta is MDE and sigma is standard deviation. For binary metrics (click rate 10%, MDE 0.5%), you need ~30K users per variant. Run the power calculation before starting—underpowered tests give inconclusive results that waste time.

**Q: What is the novelty effect and how does it affect ML A/B test results?**
A: Users initially engage more with novel features or UI changes regardless of quality, inflating the new model's metrics in the first few days. This artificial boost fades after users adapt. To control for novelty: run experiments for at least 2 weeks, look for a "novelty decay" pattern in daily metric trends (high engagement early, decreasing to steady state), analyze new vs. returning users separately, and discount first-day results in your final analysis. Declaring victory based on week-1 data is a common mistake.

**Q: How do you handle network effects in A/B tests for models that affect shared resources?**
A: Network effects (one user's experience affecting another's) violate the independence assumption of standard A/B tests. Examples: recommendation models where popular items compete, ranking models where showing an item to one user affects its ranking for others. Mitigation: use cluster-based randomization (assign entire user cohorts to treatment/control, not individuals), use time-based assignments (all users on Tuesdays get treatment, Wednesdays get control), or use holdout groups (treatment: 95% of users, holdout: 5% who never see the new model as a long-term control).

**Q: When is A/B testing not appropriate for evaluating ML models?**
A: Not appropriate when: the experiment would expose some users to a degraded experience that's ethically unacceptable (medical diagnosis, safety-critical systems), the metric takes too long to manifest relative to your release cycle (lifetime value changes that take months), the sample size required exceeds available traffic, or you can't randomize users cleanly (e.g., recommendations in a social network with strong network effects). In these cases: use shadow mode, offline evaluation on historical data, or quasi-experimental designs (before/after analysis with a holdout).

**Q: How do you avoid running too many A/B tests simultaneously?**
A: Simultaneous tests cause interaction effects if the same users are in multiple experiments. Avoid by: using an experiment mutex (users can only be in one experiment at a time for the same feature area), prioritizing experiments by expected impact and run sequentially, using a factorial design when you must test multiple things together. Track experiment coverage: what fraction of users are in at least one experiment? Above 60%, interaction effects become significant. Budget experiment slots as a limited resource.

## Interview Quick-Reference
**A/B test?** Split traffic, measure impact with stats rigor, compare models. Power analysis: determine sample size upfront.

## Related Topics
- [Monitoring & Observability](16-monitoring-and-observability.md) — tracks metrics
- [Canary Deployment](12-canary-deployment.md) — gradual rollout

## Resources
- [Trustworthy Online Controlled Experiments](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264)
