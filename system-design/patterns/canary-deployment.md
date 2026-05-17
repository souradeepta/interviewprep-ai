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

## Interview Quick-Reference
**Canary?** Gradual rollout: 5% → 25% → 50% → 100%. Monitor, rollback if bad.

## Related Topics
- [Blue-Green Deployment](blue-green-deployment.md) — alternative
- [A/B Testing](ab-testing.md) — measure impact

## Resources
- [Deployment Patterns](https://martinfowler.com/bliki/CanaryRelease.html)
