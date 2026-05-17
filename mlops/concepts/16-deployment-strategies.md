# Deployment Strategies: Strategies for Rolling Out Models Safely

## Definition & Why It Matters

Deployment strategy determines how you transition from old model to new model. Not all strategies are equal: some are risky (deploy to 100% users immediately), others are safe (test 1% first).

**The deployment risk:** New model has hidden bugs. Deploying to 100% traffic immediately = all users affected, incident. Solution: gradually roll out, monitor continuously, rollback instantly if problems detected.

**Why strategies matter:**
- **Risk management**: Limit blast radius. 1% affected is better than 100%.
- **Monitoring**: Catch issues early (blue-green), with small user impact (canary).
- **Rollback speed**: Identify issues, decide to rollback, execute within minutes.
- **Zero downtime**: Users don't experience interruption during deployment.

Netflix uses canary deployment for all models. Stripe uses blue-green for fraud models. Every production system needs a deployment strategy.

---

## How It Works

### Deployment Strategy Comparison

**1. Canary Deployment** (recommended)
```
Deploy new model to 5% traffic (1% of users)
    ↓
Monitor: latency, accuracy, errors
    ↓
If good (24h): expand to 25%
    ↓
If good (24h): expand to 50%
    ↓
If good (24h): expand to 100%
    ↓
If bad (any time): rollback to previous model instantly
```

Advantages: Low risk (5% affected), catches issues, fast rollback
Disadvantages: Takes 3+ days for full deployment

**2. Blue-Green Deployment** (safe but slower)
```
Blue (current model): receives all traffic
    ↓
Green (new model): deployed, tested in shadow, all ready
    ↓
Switch: router instantly sends ALL traffic from Blue → Green
    ↓
If good: keep Green, decommission Blue
    ↓
If bad: instant rollback (switch traffic back to Blue)
```

Advantages: Instant rollout, instant rollback
Disadvantages: Requires double infrastructure (run both models simultaneously), resource cost

**3. Rolling Deployment** (fast but risky)
```
Deploy to 1 replica, wait for it to start
    ↓
Remove 1 old replica, add 1 new replica
    ↓
Repeat: remove old, add new
    ↓
Eventually: all replicas are new model
```

Advantages: Gradual transition, no double infrastructure
Disadvantages: Hard to rollback (old replicas gone)

**4. Shadow Deployment** (validation only, no traffic)
```
New model deployed, runs in parallel
    ↓
New model predictions logged but UNUSED
    ↓
Compare: new predictions vs baseline
    ↓
If good: ready for canary
    ↓
If bad: iterate before canary
```

Advantages: Zero risk to users (predictions not used)
Disadvantages: Can't measure real business impact (users not engaging)

### Canary Deployment Deep Dive

```
Deploy new model (version 1.1.0) to Kubernetes
    ↓
Configure traffic split: 95% → 1.0.0 (old), 5% → 1.1.0 (new)
    ↓
24-hour monitoring:
  - Latency: p99 should match baseline
  - Accuracy: compared via A/B test
  - Error rate: should be <1%
  - Business metrics: engagement, revenue
    ↓
Decision threshold:
  - All metrics pass? → Expand to 25%
  - Any metric failed? → Rollback to 1.0.0
    ↓
Rollback (if needed):
  - Update router: 100% → 1.0.0
  - Kill new model (1.1.0) pod
  - Done (2 seconds)
```

---

## Interview Q&A: Deployment Strategies

### Q1: "New model deployed yesterday. 10% of users now have errors. How do you respond?"
**Answer outline:** Canary deployment enables fast response:
1. **Detect**: Monitoring alerts on error rate spike
2. **Investigate**: Is it model issue or infrastructure? (Usually visible in logs)
3. **Decide**: Rollback (risk of more errors) or debug (users still affected)?
4. **Execute**: Rollback immediately
   ```
   kubectl set image deployment/model model=registry/model:v1.0.0
   ```
5. **Result**: Error rate returns to <1% in <30 seconds

Without canary: Error hits 100% users before someone notices. Outage.

### Q2: "Deploying new model. How do you ensure zero downtime?"
**Answer outline:** Strategy depends on approach:

**Blue-green**: Zero downtime (switch is instant)
- Current model (Blue) handles traffic
- New model (Green) deployed, warmed up, tested
- Switch router to Green (instant, no requests dropped)
- If problem: switch back to Blue (instant)

**Canary**: Near-zero downtime
- New model deployed, handles 5% traffic
- Old model handles 95% (running normally)
- No downtime, but small cohort sees new model

**Rolling**: Gradual transition
- New pod starts, handles requests
- Old pod stops gracefully (drains connections)
- Requests in-flight don't drop

All three can achieve zero-downtime, but mechanisms differ.

### Q3: "Feature took 3 days to deploy using canary (5% → 25% → 50% → 100%). Too slow. Options?"
**Answer outline:** Speed up without increasing risk:
1. **Faster canary**: Expand canary if metrics pass at 6h (not 24h)
   - Pros: Faster deployment
   - Cons: May miss time-of-day effects
   - When: For low-risk changes (bug fix, latency optimization)

2. **Bigger canary**: Start at 25% (not 5%)
   - Pros: Deploy faster
   - Cons: More users affected if issue
   - When: For highly-confident changes (tested extensively in shadow)

3. **Blue-green**: Deploy fully to Green, switch traffic
   - Pros: Instant deployment
   - Cons: Need to run both models (cost)
   - When: For critical models, sufficient infrastructure

4. **Shadow test longer**: Validate thoroughly before canary (reduce risk)
   - Pros: More confident in canary
   - Cons: Slower time-to-value
   - When: When canary issues keep occurring

Trade-off: Speed vs safety. Choose based on risk tolerance and confidence.

### Q4: "Model A deployed. Then Model B needs urgent deployment (bug fix). How?"
**Answer outline:** Parallel canary deployments:
- **Keep Model A canary running**: 25% traffic
- **New canary for Model B**: Start at 5%, expand if good
- **Monitor both**: Separate metrics for each model

If Model B is bug fix and urgent:
- Start at 10-25% (not 5%)
- Expand faster (6h windows, not 24h)
- Keep Model A → A' transition running in parallel

Orchestration: Kubernetes Flagger or Istio handles traffic splitting, auto-rollback.

### Q5: "Design deployment strategy for 100 models across company."
**Answer outline:** Standardized strategy for all:

**Baseline: Canary deployment**
- Default for all models: 5% → 25% → 50% → 100%
- Reduces risk, catches issues
- Takes 3 days

**Variations:**
- **Low-risk** (bug fix, optimization): Canary 10% → 50% → 100% (2 days)
- **High-risk** (major model change): Shadow test 2 weeks, then canary 5% (4 days)
- **Critical models** (fraud, safety): Blue-green (instant, double infra cost)

**Infrastructure:**
- Kubernetes for orchestration
- Flagger or Istio for traffic splitting + auto-rollback
- Monitoring: alert on latency, error rate, business metrics

**Approval:**
- Low-risk: auto-approved canary
- High-risk: manual approval for each stage
- Critical: manual approval + on-call engineer monitoring

Result: 100 models, safe, standardized, automated.

---

## Best Practices

1. **Always start small**: 5% is not "we're already deploying," it's "we're testing." Expand deliberately.

2. **Monitor before deciding to expand**: Don't expand on schedule (5% → 25% on day 2 if bad). Expand only if metrics good.

3. **Define rollback criteria upfront**: "Rollback if latency increases >10% or error rate increases >1%." Decide before deployment.

4. **Fast rollback is essential**: Canary only works if you can rollback in <1 minute. Test rollback procedure.

5. **Monitor different metrics for different models**: Recommendation (engagement), fraud (detection rate), ETA (latency). Not all models care about accuracy.

6. **Keep rollback version available**: Previous model version must be running (or quickly deployable). Can't rollback if old version is gone.

7. **Communicate deployment to team**: When you're deploying what model? So everyone knows. Prevents confusion if issues appear.

8. **Test rollback during canary**: Don't just theoretically know rollback works. Actually test rolling back from canary (before full deployment).

9. **Separate canary traffic**: If possible, canary users shouldn't know they're testing (no UX impact). Use random assignment, not specific user cohort.

10. **Monitor post-deployment**: Even after 100% deployment, watch metrics. Issues can appear with more traffic (concurrency bugs, cascade failures).

---

## Common Pitfalls

1. **Deploy to 100% immediately**: "Tests passed, ship it." Hidden bugs affect all users. Should canary first.

2. **No rollback plan**: Deployed bad version. Previous version deployed but taking 2 hours to roll back. Too slow.

3. **Monitoring fails**: Alert set to wrong threshold. Model degrades 20%, alert doesn't fire. Discovered by users.

4. **All traffic switches at once**: Blue-green switch happens, 1% of traffic gets errors (DNS cache, edge cases). More complex than expected.

5. **Canary cohort gets worse experience**: 5% users on new model, experience consistently worse (latency, bugs). They churn.

6. **No test of rollback**: Assume rollback works. When needed, it's broken. Always test rollback procedure.

7. **Canary period too short**: Expand to 25% after 2 hours. Miss time-of-day effects (night traffic behaves differently).

8. **Previous model deleted**: Deployed new model, deleted old model to save space. Now can't rollback.

9. **Canary doesn't catch the issue**: New model okay on 5% traffic, breaks on 50%. Indicates weak monitoring or unusual traffic.

10. **Slow deployment process**: Takes 5 days to deploy. Business can't iterate fast. Consider faster strategies (blue-green) for non-critical.

---

## Real-World Examples

### Example 1: Netflix Canary Deployment
Netflix uses canary for all recommendation model updates:
- **Canary**: 5% members for 24h
- **Monitor**: Watch hours, engagement, complaints
- **Expand**: 25% (24h) → 50% (24h) → 100%
- **Frequency**: Deploy ~50 models/day using this strategy

### Example 2: Stripe Fraud Model Blue-Green
Stripe uses blue-green for critical fraud model (false deployment risk too high):
- **Blue**: Current fraud model (100% traffic)
- **Green**: New model deployed, tested extensively in shadow
- **Switch**: Instant traffic switch to Green
- **Monitoring**: 24h watch for fraud loss changes
- **Rollback**: If fraud loss increased, instant switch back to Blue

### Example 3: Uber ETA Deployment
Uber uses canary for ETA model (less critical than fraud):
- **Canary**: 1% of requests get new model
- **Monitor**: p99 latency, MAE (mean absolute error)
- **Threshold**: Rollback if latency > 150ms or MAE worse than baseline
- **Expand**: 1% → 5% → 25% → 50% → 100% as metrics pass

---

## Sample Interview Case Study

**Scenario:** Airbnb price prediction model update. Current model: 85% MAE. New model: 84% MAE (1% improvement).

**Deployment strategy:**

1. **Pre-deployment validation**:
   - Unit tests passed
   - Shadow test on 2 weeks production data: MAE 84% (matches lab)
   - Latency: same as current model (50ms)
   - Fairness: performance across 100+ cities acceptable

2. **Canary phase** (24h):
   - Deploy to 5% of listing searches
   - Monitor: prediction MAE, latency, complaints
   - Success criteria: MAE ≥ 84%, latency ≤ 50ms
   - Result: Metrics good, expand

3. **Gradual rollout**:
   - 25% traffic (24h) → metrics still good → 50% traffic (24h) → metrics still good → 100% traffic
   - Total: 3 days from canary to full deployment

4. **Post-deployment monitoring**:
   - Continue monitoring MAE, latency
   - Set alert: if MAE > 85%, page on-call
   - Weekly review: is 1% improvement translating to business impact?

5. **Rollback plan**:
   - If MAE degrades to 85%+ at any point, instant rollback to previous version
   - Rollback takes <1 minute
   - Alert team of unexpected degradation

**Result:** Safe 3-day deployment, catches issues early, fast rollback if needed.

**Strong answer:** "Use canary deployment: 5% traffic for 24h (monitor MAE and latency), expand 25% → 50% → 100% as metrics hold. Auto-rollback if MAE degrades. Total 3 days to full deployment. Enables safe rollout and fast feedback."

---

## Key Takeaways

Deployment strategy is not "just deploy," it's "test gradually, monitor continuously, rollback instantly if needed."

**Strategy spectrum:**
- **Safe but slow**: Blue-green (instant switch, double cost)
- **Balanced**: Canary (5% → 25% → 50% → 100%, 3 days)
- **Fast but risky**: Rolling (no double cost, hard to rollback)

**Common interview pattern:** "How do you deploy new model?" → Answer: "Canary: 5% traffic 24h, monitor metrics, expand if good. 3 days to full deployment. Auto-rollback if degradation."

---

## Related Concepts

- **Model Serving** (Concept 14): Infrastructure that runs models
- **Model Registry & CI/CD** (Concept 15): Pipeline that builds deployment
- **Monitoring** (Concept 18): Monitors during deployment
- **Incident Management** (Concept 22): Handles deployment failures
