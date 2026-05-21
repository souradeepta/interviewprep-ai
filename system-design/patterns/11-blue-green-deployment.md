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

### Deployment Strategy Comparison

| Aspect | Blue-Green | Canary | Rolling | Shadow |
|--------|-----------|--------|---------|--------|
| **Cost overhead** | 100% (2x infra) | 5% (small cohort) | 0% (in-place) | 50% (duplicate compute) |
| **Rollback time** | 30-60 sec (LB flip) | 5-10 min (gradual shift) | 2-5 min (image pull) | N/A (shadow only) |
| **Risk level** | Low (full validation) | Medium (limited exposure) | High (live exposure) | Lowest (shadow only) |
| **Complexity** | High (2 systems) | Medium (gradual logic) | Low (standard) | Medium (mirroring) |
| **Data migration needed** | Yes | No | No | No |
| **Time to deploy** | 10-30 min (setup) | 30-60 min (gradual) | 5-10 min (quick) | 15-20 min |
| **Best for** | Critical services | Safe changes | routine updates | risky changes |
| **Example** | Payment system | New model v2 | Bug fix hotpatch | ML experiment |

### Cost Model (10 server cluster, $5K/month baseline)

**Blue-Green (100% overhead):**
- Blue: 10 servers = $5K/month
- Green: 10 servers = $5K/month (running, tested, waiting to deploy)
- Total: $10K/month
- Benefit: zero downtime, 1-minute rollback
- ROI: if prevents 1 outage/month ($50K loss), saves $40K/month

**Canary (5% overhead):**
- Production: 10 servers = $5K/month
- Canary: 1-2 servers (5-10%) = $250-500/month
- Total: $5.25-5.5K/month
- Benefit: low cost, detects issues before full rollout
- Trade-off: 30-60 minute rollout time vs instant

**Rolling (0% overhead, in-place):**
- Production: 10 servers = $5K/month (redeploy each)
- No extra infrastructure
- Risk: live traffic during rollout, no instant rollback

**Shadow (50% overhead):**
- Production: 10 servers = $5K/month
- Shadow (non-prod): 5-10 servers mirroring traffic = $2.5-5K/month
- Total: $7.5-10K/month
- Benefit: validate changes before they hit real traffic
- Trade-off: higher cost than canary, no actual traffic data

### Decision Matrix by Risk Level

| Service Type | Recommended | Reason | Cost |
|--------------|-------------|--------|------|
| **Critical (payments, auth)** | Blue-Green | Zero-downtime essential, can afford 2x | $10K |
| **User-facing (recommendations)** | Canary | Lower cost, gradual validation | $5.3K |
| **Background jobs (batch processing)** | Rolling | No user impact, in-place OK | $5K |
| **Experimental (new ML feature)** | Shadow | Validate before exposing to users | $7.5K |
| **Routine hotfix** | Rolling | Low risk, speed matters | $5K |

---

## Production Failure Scenarios

**Scenario 1: Green Infrastructure Cold (Cache Empty, Connections Unstable)**

**What breaks:** Green infrastructure is ready, tested with synthetic traffic. Switch happens at 2pm. Immediately after:
- Database connection pool cold (needs 30 seconds to warm up)
- Redis cache empty (cache hit rate drops from 95% to 5%)
- Request latency spikes from 50ms to 500ms for 2 minutes
- Users see timeout errors, queue builds up

**Why it happens:**
- Green was tested but not warmed up with real traffic patterns
- Cold start effect: connection pools, caches need warmup
- Assumption: "if tested, it's ready" but warmup is different from correctness

**Detection:**
```
Alert: if (latency_p99 > baseline * 5) → CRITICAL
Monitor: cache_hit_rate (drops from 95% to 5% at switch time)
```

**Recovery:**
1. Immediate: Users see slow responses for 2 minutes, automatic retry succeeds
2. Monitor: latency gradually normalizes as cache warms
3. Next time: pre-warm cache before switch

**Prevention:**
- Warmup before flip: send 1% of traffic to Green for 5 minutes
- Pre-populate cache: load top-1K keys into Redis on Green
- Connection pooling: pre-create DB connections on startup
- Smoke test: verify latency on Green matches Blue before switch

---

**Scenario 2: Database Schema Incompatibility**

**What breaks:** Model v2 in Green requires new database schema (extra column `user_engagement`). Blue still uses old schema. Switch happens, Green tries to write to old schema, INSERT fails with column-not-found error.

**Why it happens:**
- Assumption: "switch is just code"
- Database schema is shared between Blue and Green
- Schema changes aren't backward compatible

**Detection:**
```
Alert: if (error_rate > 1%) → CRITICAL
Monitor: specific error "column user_engagement not found"
```

**Recovery:**
1. Detect: Green errors immediately after switch
2. Rollback: flip traffic back to Blue (instant)
3. Migrate: update database schema to add column
4. Retry: switch to Green again

**Prevention:**
- Dual-write before flip: Blue writes to both old and new schema
  - Old columns: keep for Blue
  - New columns: pre-populate on Green
- Backward compatibility: Green code reads old columns if new ones missing
- Pre-deploy: schema migration runs 1 hour before flip

---

**Scenario 3: Traffic Split Between Blue and Green (Inconsistent State)**

**What breaks:** LB config updated to flip to Green. Update is atomic, but during transition window (0.5 seconds), some clients still hit Blue, others hit Green. User A makes purchase on Green, User B checks order status on Blue (different database view). Inconsistency.

**Why it happens:**
- Atomic LB switch takes <1 second, but not instantaneous
- In-flight requests bridge old and new state
- Clients retry, may hit different replica

**Detection:**
```
Alert: if (both_blue_and_green_error_rate > 0.1%) → WARN
Monitor: version_distribution (should be 0-100%, not 50-50% after 2 sec)
```

**Recovery:**
1. Detect: error logs show some errors on Blue, some on Green during transition
2. Verify: check LB config completed, new traffic only goes to Green
3. Rerun: if issues persist, flip back to Blue and retry later

**Prevention:**
- Connection draining: on Blue, stop accepting new connections (flush in-flight)
- LB verification: check LB config deployed before declaring switch complete
- Monitor post-switch: verify 100% traffic to Green (no Blue traffic)
- Grace period: wait 30 seconds after switch before declaring success

---

## Implementation Guidance

**Wrong:** DNS switch (5min propagation). Inconsistent states.
**Right:** LB config change (instant). Atomic switch.

---

## Sophisticated Interview Q&A

**Q1: Blue-Green costs 2x infrastructure ($10K vs $5K). When is this worth it?**

A: Calculate ROI based on downtime cost:

If 1 outage/month prevented:
- Cost of downtime: 30 minutes × $10K/hour = $5K loss
- Cost of blue-green: $5K/month infrastructure overhead
- Break-even: if prevent 1 outage/month, infrastructure cost is recovered

Different answers for different services:
- **Payment system:** $10K loss per incident → Blue-green is essential
- **Analytics dashboard:** $100 loss per incident → Blue-green overkill, use canary
- **Background batch job:** $0 loss (users don't see it) → Blue-green wasteful, use rolling

**Answer:** Worth it if (downtime_cost × incidents_per_month) > infrastructure_cost

---

**Q2: Green infrastructure ready. Ready to flip traffic?**

A: Not yet. Must check:

1. **Cache warmup:** Send 1% traffic to Green for 5 minutes
   - Monitor latency: if p99 > baseline, cache still warming
   - Only flip when latency normalizes

2. **Database readiness:** Can Green write to database?
   - Schema compatible?
   - New columns exist or nullable?
   - Run dry-run: write a test record to verify

3. **Dependencies:** Are upstream services aware of Green's IP?
   - DNS updated?
   - Service mesh config deployed?
   - Internal service discovery ready?

4. **Health checks:** Does Green pass all checks?
   - Application health endpoint responding?
   - Database connection pool size OK?
   - Load generation test (1000 QPS synthetic)?

Only flip when ALL checks pass. Rushing causes cold-start issues.

---

**Q3: Flip to Green, latency spikes to 500ms. Immediate action?**

A: **Immediate (first 10 seconds):**
- Don't panic, observe for 30 seconds
- If trend is: 500ms → 400ms → 300ms, it's warming up, stay on Green
- If stable at 500ms or increasing, rollback immediately

**Rollback (if needed):**
```
LB config: "route to Green" → "route to Blue"
Time: 30-60 seconds to flip back
Impact: users see latency return to normal
```

**Analysis:**
- Why spike? (A) Cache warming, (B) Connection pool cold, (C) Bug in v2
- Distinguish: if latency recovers → (A) or (B), acceptable. If stays high → (C) rollback.

---

**Q4: Database migration needed for Green. How avoid outage?**

A: **Pre-flip strategy (Dual-write):**
1. Deploy v2 code to Green (not-yet-live)
2. Blue still running (old code, old schema)
3. Update Blue code to dual-write: write to BOTH old schema + new schema
4. Run data migration: copy existing data to new columns
5. Wait 1 hour: ensure dual-write is working, new data populated
6. Flip to Green: now it can use new schema

**Result:** Zero downtime, schema migration completed before flip

**Alternative if no time:**
- Accept schema mismatch temporarily
- Green code: writes new columns but reads old columns as fallback
- Green: handles both old and new schema
- Next week: clean up old schema

---

**Q5: Blue-Green vs Canary vs Rolling. How choose?**

A: Decision tree:

```
Is downtime acceptable?
├─ No (critical service) → Blue-Green ($10K, instant rollback)
├─ Yes
    ├─ Can afford slow rollout? (30-60 min)
    │  ├─ Yes → Canary ($5.3K, gradual exposure)
    │  └─ No → Rolling ($5K, quick but live exposure)
    └─ Want to validate before exposing users?
       ├─ Yes → Shadow ($7.5K, non-prod validation)
       └─ No → Rolling/Canary
```

**Examples:**
- Payment system: Blue-Green (downtime = revenue loss)
- New ML model: Shadow (want to validate on real traffic patterns first)
- Routine hotfix: Rolling (low risk, speed matters)
- Daily deployment: Canary (safe change, gradual rollout)

---

**Q6: Rollback from Green back to Blue. What if Blue is broken?**

A: Scenario: During the window when Blue is offline (Green serving all traffic), Blue's replica crashes. Blue can't be restarted. Rollback blocked.

**Prevention:**
- Keep Blue running while Green is being tested
- Never shut down Blue until Green is fully validated
- Have third environment (staging) as backup if needed

**If it happens:**
1. Can't rollback (Blue unavailable)
2. Must stabilize Green instead
3. Fix issue on Green, re-validate, stay on Green
4. Restart Blue infrastructure in background

**Key lesson:** Blue-Green safety depends on maintaining both environments in good state.

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

