# Canary Deployment

## TL;DR
Roll out to small % traffic first (5-10%), monitor metrics, gradually increase if healthy. Detect bad models early, limit blast radius.

## Core Intuition
Don't push to all users at once. Test on subset first.

## How It Works
```
Day 1: Deploy to 5% traffic
  Monitor: latency, error rate, accuracy
  If healthy → 25%
  
Day 2: 25% traffic
  Monitor for 24h
  If healthy → 50%
  
Day 3: 50%, 100%
```

**Rollback:** if metrics bad at any stage, revert to previous model.

## Key Properties / Trade-offs
- Safety: catch issues early
- Time: gradual → slower full deployment
- Cost: run 2 models in parallel (overhead)

## Common Mistakes / Gotchas
- **Bad metrics:** monitor wrong thing → miss issues
- **Too fast ramp:** 5% → 100% in 1h → catch issues too late
- **No rollback plan:** can't quickly revert

## Interview Q&A

**Q: What percentage of traffic should you route to a canary and for how long?**
A: Start with 1-5% of traffic: enough to get statistical signal, small enough to limit blast radius if the canary fails. For models with high-volume traffic (>10K req/s), 1% gives 100+ req/s to the canary—sufficient for significance in 1-2 hours. For lower-traffic models, increase to 5-10% to accumulate enough samples. Hold at each traffic level for at least 1 hour (longer for metrics that take time to manifest: session-level metrics need multiple user sessions). Stop at 50% and validate before proceeding to 100%.

**Q: How do you decide which metrics trigger an automatic canary rollback?**
A: Rollback triggers should be: latency degradation (canary P99 > baseline P99 x 1.5), error rate increase (canary error rate > baseline x 2), prediction quality degradation (primary business metric drops > threshold), or model-specific anomaly (prediction distribution shifts significantly). Set thresholds conservatively—it's better to roll back a good model unnecessarily than to leave a bad model running. Log all automatic rollbacks and review them; false rollbacks reveal metric sensitivity issues or model quality improvements that look like regressions.

**Q: How do you compare canary and baseline performance statistically?**
A: Use statistical tests appropriate for your metric type: t-test for continuous metrics (latency, revenue), proportion test for rate metrics (click-through rate, error rate), Mann-Whitney U for non-normal distributions. Determine minimum detectable effect and required sample size before the experiment (don't run until you see significance). Correct for multiple comparisons if testing many metrics. Report effect size, not just p-value—a statistically significant 0.1% improvement may not be practically significant.

**Q: What is a canary analysis framework and how does it differ from manual monitoring?**
A: Manual monitoring: engineers watch dashboards and decide when canary looks bad. Error-prone: humans miss slow degradations, have inconsistent thresholds, and don't account for time-of-day effects. Automated canary analysis (Spinnaker, Kayenta): continuously compares canary vs. baseline metrics, accounts for confounders (traffic patterns, time of day), applies statistical tests, and produces a pass/fail/inconclusive verdict. Automated analysis is more reliable, faster, and consistent—it should gate promotion to 100% traffic without requiring manual approval for routine deployments.

**Q: How do you run canary deployments for ML models with personalization?**
A: Personalized models require user-consistent routing: the same user should always hit either canary or baseline during the experiment, not alternate between them (which would contaminate both groups). Implement user-level assignment: hash user_id % 100, route <5% to canary for the duration of the experiment. This ensures: clean A/B comparison, consistent user experience, and accurate measurement of long-term behavioral changes (which require a user to experience only one model version).

## Interview Quick-Reference
**Canary?** Gradual rollout: 5% → 25% → 50% → 100%. Monitor, rollback if bad.

## Related Topics
- [Blue-Green Deployment](11-blue-green-deployment.md) — alternative
- [A/B Testing](14-ab-testing.md) — measure impact

## Resources
- [Deployment Patterns](https://martinfowler.com/bliki/CanaryRelease.html)
