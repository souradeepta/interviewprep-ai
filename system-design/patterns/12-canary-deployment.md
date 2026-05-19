# Canary Deployment

## TL;DR
Gradual rollout: deploy to 5% of users first. Monitor metrics. If good, 25% → 50% → 100%. If bad, rollback immediately. Reduces blast radius of buggy models.

## Core Intuition
Don't release to everyone at once. Release to brave few first (canaries). If they're OK, expand. If problem, only 5% affected, easy rollback.

## How It Works

1. Deploy model v2 to 5% of traffic
2. Monitor: latency, errors, prediction distribution
3. If metrics OK for 30 min: increase to 25%
4. Continue: 50%, then 100%
5. If metrics bad: rollback to v1 (instant)

| Phase | % Users | Duration | Decision |
|-------|---------|----------|----------|
| Canary | 5% | 30 min | Metrics OK? |
| Early adopters | 25% | 1 hour | Errors <0.1%? |
| Rollout | 50% | 1 hour | Latency OK? |
| General availability | 100% | Ongoing | Success! |

## Key Properties / Trade-offs
- Safety: catch issues early, affect few users
- Speed: gradual rollout is slower than instant
- Complexity: need traffic splitting, monitoring
- Risk: some users get v2, others get v1 (short period)

## Common Mistakes / Gotchas
- Not monitoring during canary (rollout succeeds but model is bad)
- Rollback too slow (wait for multiple failures before acting)
- Canary % too high (50% canary defeats the purpose)
- Not isolating canary traffic (canary sees different data than rest)

## Best Practices
- **Monitor early:** liveness (alive?), readiness (loaded?), deep (quality metrics)
- **Rollback threshold:** if error rate > 1%, rollback immediately
- **Gradual expansion:** 5% → 25% → 50% → 100% (not 5% → 100%)
- **Separate monitoring:** track canary metrics separately from baseline
- **Metrics for rollback:** accuracy, latency, error rate—not just "is it alive"

## Code Example
```python
import time

class CanaryDeployment:
    def deploy(self, model_version, stages=[0.05, 0.25, 0.5, 1.0]):
        for traffic_pct in stages:
            print(f"Routing {traffic_pct*100}% to {model_version}")
            self.set_traffic(model_version, traffic_pct)
            
            # Monitor
            time.sleep(300)  # 5 minutes per stage
            
            metrics = self.get_metrics(model_version)
            if metrics['error_rate'] > 0.01:  # >1% error
                print(f"Rollback: error rate {metrics['error_rate']}")
                self.rollback()
                return False
            
            print(f"Stage OK, error_rate={metrics['error_rate']}")
        
        print("Canary deployment successful!")
        return True
```

## Interview Q&A
**Q: Canary at 5% detects bug affecting 10% of users (edge case). Problem?**
A: Bad luck—edge case didn't trigger in first 5%. Solution: targeted canary—route canary traffic to users likely to hit edge case (e.g., specific geo, user segment). Or: longer canary duration (2 hours instead of 30 min).

**Q: How do you choose metric thresholds for rollback (error rate 1%)?**
A: Start conservative: 0.5% (half of baseline). Lower threshold → more rollbacks (false positives). Higher threshold → miss real issues. AB test thresholds: compare automatic vs manual rollback decisions. Calibrate over time.

## Interview Quick-Reference
| Stage | Traffic | Duration | Decision |
|-------|---------|----------|----------|
| Canary | 5% | 30 min | Error <1%? |
| Early | 25% | 1 hour | Latency OK? |
| Main | 50% | 1 hour | Metrics green? |
| GA | 100% | Stable | Ship it! |

## Related Topics
- [Blue-Green Deployment](11-blue-green-deployment.md)
- [A/B Testing](14-ab-testing.md)

## Resources
- [Canary Deployments](https://martinfowler.com/bliki/CanaryRelease.html)
