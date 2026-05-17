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

## Interview Quick-Reference
**A/B test?** Split traffic, measure impact with stats rigor, compare models. Power analysis: determine sample size upfront.

## Related Topics
- [Monitoring & Observability](monitoring-and-observability.md) — tracks metrics
- [Canary Deployment](canary-deployment.md) — gradual rollout

## Resources
- [Trustworthy Online Controlled Experiments](https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264)
