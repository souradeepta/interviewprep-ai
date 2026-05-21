# Canary Deployment

## Detailed Description

Gradual rollout: 5%→25%→50%→100%. Monitor, auto-rollback if errors exceed threshold. Balances risk and visibility. Slower than blue-green but cheaper.

## Core Intuition

Canary = test with small group first. 5% users get new model, 95% get old. Monitor: does new model work? Accuracy same? No errors? If good, increase to 25%. If bad, rollback to 0%. Gradual reduces risk.

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

## Detailed Trade-off Analysis

### Canary Metrics & Decision Logic

**Key metrics to monitor during canary:**
```
Error rate delta: (canary_errors - baseline_errors) / baseline_errors
  Threshold: < 1% (canary within 1% of baseline)
  
Latency delta: (canary_p99 - baseline_p99) / baseline_p99
  Threshold: < 5% (p99 latency within 5% of baseline)
  
CPU delta: (canary_cpu - baseline_cpu) / baseline_cpu
  Threshold: < 10% (resource usage within 10%)
  
Prediction distribution shift: KS-test(canary_preds vs baseline_preds)
  Threshold: KS < 0.1 (distributions similar enough)
```

**Promotion decision tree:**
```
if (error_delta < 1% AND latency_delta < 5% AND cpu_delta < 10%):
    if (KS_test < 0.1):  # distributions match
        promote_to_next_stage()
    else:
        investigate_distribution_shift()
        wait_for_more_data()
else:
    if (error_delta > 5% OR latency_delta > 10%):
        rollback_immediately()
    else:
        wait_longer_at_current_stage()
```

**Traffic ramp schedule (typical model deployment):**
```
Stage 1: 1% traffic for 15 min (100-500 QPS depending on service)
Stage 2: 5% traffic for 30 min (500-2500 QPS)
Stage 3: 10% traffic for 1 hour (1000-5000 QPS)
Stage 4: 25% traffic for 1 hour (2500-12500 QPS)
Stage 5: 50% traffic for 1 hour (5000-25000 QPS)
Stage 6: 100% traffic (full rollout)

Total time: 4-5 hours for full rollout
```

**Cost-benefit analysis:**
- Canary cost: 5% extra traffic on new model = $250/month (extra 5% GPU)
- Benefit: Catch 1 bad model per month = prevent $10K loss
- ROI: Save $10K, spend $250 = 40x return

### Deployment Strategy Comparison

| Aspect | Canary | Blue-Green | Rolling | Shadow |
|--------|--------|-----------|---------|--------|
| **Cost overhead** | 5% | 100% | 0% | 50% |
| **Rollback speed** | 10-20 min (gradual) | 30 sec (instant) | 5-10 min | N/A |
| **Risk level** | Medium (5% exposed) | Low (full test first) | High (live traffic) | Lowest (shadow) |
| **Rollout duration** | 4-5 hours | 30 seconds | 10-30 minutes | N/A |
| **Ideal for** | Safe model updates | Critical systems | Hotfixes | Risky ML experiments |
| **Example use case** | New recommendation model | Payment system update | Bug patch | ML model A/B test |

**Decision matrix by change risk:**
- Low risk (bugfix, tuning): Rolling (fast, no extra cost)
- Medium risk (new model): Canary (4h, 5% cost, catch issues)
- High risk (new system): Blue-green (30s, 100% cost, guaranteed safety)
- Very risky (experimental): Shadow (validate offline first)

---

## Production Failure Scenarios

**Scenario 1: Canary metrics hidden by noise**
- Canary error 1%, baseline 0.9%. Promote. At 100%, errors 10x.
- Fix: Higher confidence threshold (delta>1%, not >0.1%).

**Scenario 2: Slow rollout, SLA urgent**
- Canary takes 2 hours. SLA needed now.
- Fix: Use blue-green (instant switch) instead.

**Scenario 3: Canary and stable serve same user**
- Session routes between canary/stable. Inconsistent behavior.
- Fix: Sticky routing (user+version).

---

## Implementation Guidance

**Wrong:** Canary 1% for 5 minutes. Misses slow issues.
**Right:** Canary 5% for 30 minutes. Detect issues early.

---

## Sophisticated Interview Q&A

**Q1: Canary % and duration?**
A: 1-5%, 30 min minimum. Monitor error rate, latency, prediction distribution closely.

**Q2: Metric threshold for promotion?**
A: Canary error within 10% of baseline (prevents promoting on noise).

**Q3: Slow rollout conflicts with SLA?**
A: Use blue-green instead (instant).

---

## Cost & Resource Analysis

5% compute overhead. ROI: catch 1 bad model/week.

---

## Monitoring & Observability

Metrics: canary_error_rate, canary_latency_p99, canary_pred_distribution. Alerts: error>threshold, latency>SLA.

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

Q: How do you decide 5% vs 10% vs 50%?
A: A: Depends on confidence and risk tolerance. High confidence (extensive testing): 50%. Medium (some tests): 10%. Low (risky change): 5%. Also depends on error impact: fraud = lower %, recommendations = higher %.

Q: Auto-rollback if error rate >5%?
A: A: Yes. Monitor error rate continuously. If new model's error >5%, rollback automatically (humans notified). Prevents bad deploys reaching 100%.

Q: Canary conflicts with feature flags?
A: A: Complementary. Canary: infrastructure rollout (version A→B). Feature flags: code logic toggle. Use both: canary deploy new code version, feature flags control which users see new logic.

Q: Time per stage: 5%→25%→50%?
A: A: Depends on traffic and confidence. 1M DAU: 5% = 50K users (sufficient data). 1 hour at each stage = 3 hours total. High-traffic: faster. Low-traffic: slower (need more time for statistical significance).

Q: Gradual rollout but need ASAP deployment?
A: A: Trade-off. Can accelerate: 5%→50%→100% (skip 25%). Or skip canary, blue-green deploy (instant but risky).

Q: Two models differ significantly (different accuracy)?
A: A: Stay at 5% canary longer. Collect more data. Compare 95th percentile metrics (not just mean). If confident, increase. If not, investigate why models differ.

Q: Canary breaks for 1% of traffic, 99% happy?
A: A: Depends on impact. 1% errors acceptable? If issue only for power users (1%), might worth fixing before expanding. If random (any user could hit), unacceptable.

Q: How do you measure 'good enough' to expand?
A: A: Define metrics upfront. Accuracy >94%, latency <100ms, errors <1%. Compare canary vs baseline. If canary meets thresholds, expand. Otherwise, investigate.
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

