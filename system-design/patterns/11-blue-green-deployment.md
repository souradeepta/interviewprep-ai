# Blue-Green Deployment

## Detailed Description

Two identical production environments: blue (old), green (new). Deploy to green, test, switch traffic instantly. Zero downtime, instant rollback. No gradual rollout.

## Core Intuition

Blue-green = duplicate infrastructure. Blue = running (old model). Green = staging (new model). Deploy to green without touching blue. Test green fully. Flip switch: all traffic to green. If issues, flip back to blue (instant rollback, seconds).

## How It Works

1. Blue (production): serving all traffic with model v1.0.0
2. Green (staging): deploy model v2.0.0, test thoroughly
3. Validation: run synthetic tests, smoke tests, canary with 1% traffic
4. Switch: update load balancer config to route to Green
5. Rollback (if needed): switch back to Blue (config change, instant)

| Phase | Blue | Green | Traffic |
|-------|------|-------|---------|
| Initial | v1.0.0 running | Idle | 100% → Blue |
| Deployment | v1.0.0 running | v2.0.0 running | 100% → Blue |
| Validation | v1.0.0 running | v2.0.0 tested | 100% → Blue |
| Switch | v1.0.0 idle | v2.0.0 running | 100% → Green |
| Rollback | v1.0.0 running | v2.0.0 idle | 100% → Blue |

## Key Properties / Trade-offs
- Cost: 2x infrastructure (both Blue and Green running)
- Speed: instant switch (config change)
- Safety: full rollback, no downtime
- Complexity: manage two environments

## Common Mistakes / Gotchas
- Forgetting to warm cache on Green (cold start latency)
- Different infrastructure on Blue/Green (breaks reproducibility)
- Not testing on Green thoroughly before switch
- Rollback not working (Blue degraded during switch window)

## Detailed Trade-off Analysis

| Aspect | Blue-Green | Canary | Shadow |
|--------|-----------|--------|--------|
| Cost | 100% overhead | 5% overhead | 50% overhead |
| Rollback time | 1 min | 5 min | N/A |
| Risk | Low (full test) | Medium | Lowest |
| Complexity | High | Medium | Medium |
| Data readiness | Must migrate | N/A | N/A |

**Decision:** Critical services→blue-green (zero downtime). Safe changes→canary (lower cost). Validation→shadow.

---

## Production Failure Scenarios

**Scenario 1: Green not ready before switch**
- DNS change incomplete, some clients see old IP. Inconsistent.
- Fix: Use LB config change (instant) not DNS.

**Scenario 2: Database schema incompatible**
- Blue uses schema v1, green needs v2. Can't switch without migration.
- Fix: Dual-write before flip. Migrate data. Then flip.

**Scenario 3: Blue still gets traffic after flip**
- LB not fully switched. Both blue and green serving = inconsistent.
- Fix: Verify LB config change complete. Monitor both.

---

## Implementation Guidance

**Wrong:** DNS switch (5min propagation). Inconsistent states.
**Right:** LB config change (instant). Atomic switch.

---

## Sophisticated Interview Q&A

**Q1: Cost $2K/month (100% overhead). Worth it?**
A: If downtime costs $10K/hour, zero downtime saves $10K per incident. Justify if 1 incident/month prevented.

**Q2: Blue-green vs canary?**
A: Blue-green: instant switch, high cost. Canary: gradual, low cost, higher risk. Choose based on risk tolerance.

**Q3: Data schema changes. How switch?**
A: Dual-write (both v1+v2), backfill, migrate, then flip. Or accept schema migration downtime (use canary instead).

---

## Cost & Resource Analysis

2x infrastructure = 100% cost. ROI: prevents 1 incident/month.

---

## Monitoring & Observability

Metrics: error_rate_post_switch, latency_post_switch, version_distribution. Alerts: error_rate spike, latency spike post-switch.

## Best Practices
- **Pre-warm Green:** run synthetic traffic to populate caches
- **Identical infrastructure:** Blue and Green must be clones
- **Test on Green:** run full test suite before switch
- **Gradual switch:** 10% → 50% → 100% rather than instant
- **Keep Blue warm:** don't turn off old environment for 30 min after switch

## Code Example
```python
import subprocess

class BlueGreenDeployment:
    def deploy(self, model_version):
        # 1. Deploy to Green
        print("Deploying to Green...")
        subprocess.run(["kubectl", "apply", "-f", "green-deployment.yaml"])
        
        # 2. Wait for Green readiness
        subprocess.run(["kubectl", "wait", "--for=condition=ready", "pod", "-l", "app=green"])
        
        # 3. Test Green
        print("Running tests on Green...")
        test_result = subprocess.run(["python", "test_model.py", "--endpoint=green.local"])
        
        if test_result.returncode != 0:
            print("Tests failed, not switching")
            return False
        
        # 4. Switch traffic to Green
        print("Switching traffic to Green...")
        subprocess.run(["kubectl", "patch", "service", "model-api", 
                       "-p", '{"spec":{"selector":{"version":"green"}}}'])
        
        # 5. Monitor
        print("Monitoring Green for 5 minutes...")
        time.sleep(300)
        
        print("Deployment complete!")
        return True
```

## Interview Q&A

Q: Costs double: maintain 2 envs?
A: A: Yes, overhead. Justified for critical systems (can't afford downtime). Cheaper: canary (5% traffic, cheaper). Blue-green: cost vs zero downtime trade-off.

Q: Data consistency: blue and green diverge?
A: A: Use shared database. Both read/write same database. Or use data replication to sync. Ensure transactional consistency.

Q: Feature flagging vs blue-green?
A: A: Feature flags: code in blue talks to new feature (can toggle). Blue-green: completely separate env. Different trade-offs. Feature flags: can't test infrastructure changes. Blue-green: pure infrastructure test.

Q: Switch traffic: how long?
A: A: DNS change (propagation ~5min) or LB config change (instant). For instant, use LB (switch in seconds). For production, test switch time.

Q: Green fails during testing?
A: A: Don't flip. Fix green (redeploy) or rollback to blue. Blue still healthy, no customer impact.

Q: Databases incompatible (schema change)?
A: A: Must migrate data before flip. Blue uses old schema, green uses new. Migration: dual-write (write to both), backfill, then flip. Complex.

Q: When to flip: wait for full testing or ASAP?
A: A: Depends on confidence. Heavy QA: wait days. CI/CD confidence: flip after tests pass (hours). Startup: flip after smoke test (minutes).

Q: Monitoring: how do you catch issues post-flip?
A: A: Compare metrics: blue vs green. Accuracy, latency, error rate. Alert if green degrades. Rollback if needed.
## Interview Quick-Reference
| Phase | Blue | Green | Traffic |
|-------|------|-------|---------|
| Initial | v1 (prod) | - | 100% |
| Deploy | v1 (prod) | v2 (test) | 100% |
| Switch | - | v2 (prod) | 100% |

## Related Topics
- [Canary Deployment](12-canary-deployment.md)
- [Model Serving](05-model-serving.md)

## Resources
- [Blue-Green Deployment Guide](https://martinfowler.com/bliki/BlueGreenDeployment.html)

