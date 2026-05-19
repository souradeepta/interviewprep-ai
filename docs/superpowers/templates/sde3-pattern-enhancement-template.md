# SDE3 Pattern Enhancement Template

This template defines what "SDE3-level" content looks like for each of the 6 new sections added to system design patterns. Use this as your reference when enhancing each pattern.

---

## 1. Detailed Trade-off Analysis Section

**Purpose:** Help senior engineers understand quantitative trade-offs between approaches within the pattern.

**Length:** 300-400 words including table

**Structure:**
- Metrics comparison table (latency, cost, complexity, scalability, failure rate)
- Cost breakdown by component
- Scalability characteristics
- Decision tree or matrix (when to use each approach)
- Real production metrics (not estimates)

### Example: For a hypothetical pattern with 3 approaches

```markdown
## Detailed Trade-off Analysis

### Metrics Comparison

| Aspect | Approach A | Approach B | Approach C |
|--------|-----------|-----------|-----------|
| Latency | 5-10ms | 50-100ms | 200-500ms |
| Cost (baseline) | $500/month | $100/month | $5,000/month |
| Operational overhead | 2 hours/week | 30 min/week | 20 hours/week |
| Scalability (max QPS) | 1,000 | 100 | 10,000 |
| Failure rate (weekly) | 0.01% | 0.5% | 0.001% |
| Infrastructure complexity | High | Low | Medium |

### Cost Breakdown

**Approach A:** $500/month
- Compute (2 servers): $300
- Storage (100GB): $50
- Monitoring/alerting: $100
- Personnel (ops overhead ~2h/week): $50/month (scaled)

**Approach B:** $100/month
- Compute (shared resource): $50
- Storage (10GB): $10
- Monitoring: $40
- Personnel: negligible

**Approach C:** $5,000/month
- Compute (10x servers for throughput): $3,000
- Real-time storage (Redis): $1,000
- Monitoring (24/7 required): $500
- Personnel (ops overhead ~20h/week): $500/month

### When to Use Each

**Use Approach A when:**
- <100ms latency required
- <10K QPS
- Can afford ops overhead
- Correctness > cost

**Use Approach B when:**
- <1 second acceptable
- <1K QPS
- Cost-sensitive
- Simple is better

**Use Approach C when:**
- >10K QPS required
- Must handle spikes
- Can afford operational complexity

### Real Production Metrics

From LinkedIn/Netflix/Uber systems:
- Well-optimized Approach A handles 95th percentile latency <15ms at 10K QPS
- Approach B costs are typically 5-10x cheaper but with 10-100x higher latency
- Approach C requires 5-10 engineers to operate at scale
```

**Expectations:**
- Use real numbers from known production systems (cite source when possible)
- Tables should compare 3+ approaches within the pattern
- Cost should break down by component (not opaque total)
- Decision criteria should be concrete (not "it depends")

---

## 2. Production Failure Scenarios Section

**Purpose:** Prepare engineers for things that actually break in production. Help them reason about detection and recovery.

**Length:** 400-500 words total

**Structure:**
- 4-6 realistic failure scenarios per pattern
- For each scenario:
  - What breaks (user-visible symptom)
  - Why (root cause)
  - How you detect it (signals, metrics, alerts)
  - How you recover (step-by-step procedure)
  - How you prevent it (preventive controls)

### Example: For a hypothetical deployment pattern

```markdown
## Production Failure Scenarios

### Scenario 1: New deployment completes, old system still running, inconsistent traffic routed

**What breaks:** Users see inconsistent responses (feature X works in request 1, fails in request 2).

**Why it happens:** 
- Deployment switched some servers to new version but not others
- Load balancer has stale cache (old server list)
- Network partition: some clients see old IPs, some see new IPs

**How you detect it:**
- Error rate spikes 5x (new code + old infrastructure incompatibility)
- Server-side error metrics show split by version (old vs new)
- Client-side: users report "worked a moment ago, now broken"
- Alert: if (error_rate > 5% AND servers_mixed_versions == true) then ALERT

**How you recover:**
1. Immediately: Identify which servers are on which version: `kubectl get pods -o wide | grep <app>`
2. Option A (rollback new): `kubectl rollout undo deployment/<app>` (takes 2-5 min)
3. Option B (complete new): `kubectl rollout resume deployment/<app> --force` (takes 2-5 min)
4. Verify: Error rate drops below 1% within 5 minutes
5. Incident review: Why did deployment get stuck in mixed state?

**How you prevent it:**
- Readiness gate: Block traffic switching until all old servers shutdown
- Monitoring: Alert if mixed versions detected (should never happen)
- Testing: Chaos test with failures mid-deployment (are old/new compatible?)
- Automation: Don't allow manual traffic switching, fully automate choreography
```

**Expectations:**
- Scenarios should be realistic (based on known incidents or documented gotchas)
- Root causes should be specific (not "integration issues")
- Recovery procedures should be step-by-step (operator can follow them at 3am)
- Prevention should be concrete (test strategy, alerting threshold, architectural fix)

---

## 3. Implementation Guidance & Gotchas Section

**Purpose:** Provide practical code-level guidance to engineers actually building this pattern.

**Length:** 400-500 words

**Structure:**
- Common mistakes with concrete "wrong vs right" code examples
- Edge cases with solutions
- Performance bottlenecks specific to the pattern
- Testing strategies (unit, integration, chaos)

### Example: For a hypothetical caching pattern

```markdown
## Implementation Guidance & Gotchas

### Common Mistakes

❌ **Wrong: Cache without TTL (stale data forever)**
```python
cache[key] = value  # No expiration
```

✅ **Right: Cache with TTL (data refreshes)**
```python
cache.set(key, value, ttl=3600)  # Expires in 1 hour
```

**Why it matters:**
- Without TTL, data stays in cache indefinitely
- If source updates, cache never reflects update (stale forever)
- Easy bug to miss in testing (works in 1-hour test, fails in production at 24h)

---

### Edge Cases

**Edge case 1: What if cache fills up?**
- Problem: Memory leak if cache has no size limit
- Solution: Set max_size with LRU eviction policy
```python
from cachetools import LRUCache
cache = LRUCache(maxsize=10000)  # Evicts least-recently-used when full
```

**Edge case 2: Multiple processes cache same data**
- Problem: If you have 10 workers, each caches independently (10x memory, cache misses)
- Solution: Use shared cache (Redis) or implement single-writer pattern
```python
# Wrong: Process-local cache (in-memory dict)
local_cache = {}

# Right: Shared cache with Redis
redis_cache = redis.Redis(host='localhost')
```

---

### Performance Bottlenecks

**Bottleneck 1: Cache lookup slower than computing**
- Symptom: Adding cache actually makes system slower
- Cause: Network round-trip to Redis (10-50ms) > local compute (1-5ms)
- Solution: Only cache expensive operations, batch multiple lookups
```python
# Batch lookups (get 100 keys in 1 round-trip instead of 100)
values = redis_cache.mget([key1, key2, ..., key100])
```

**Bottleneck 2: Thundering herd**
- Symptom: Cache expires, 1000 requests all recompute same data
- Cause: No coordination, all requests see cache miss simultaneously
- Solution: Probabilistic early refresh
```python
if cache_age > ttl * 0.9:  # Refresh at 90% of TTL
    async_refresh(key)  # Refresh in background
    return stale_value  # Serve stale while refreshing
```

---

### Testing Strategies

**Unit test:**
```python
def test_cache_hit():
    cache.set('key', 'value')
    assert cache.get('key') == 'value'

def test_cache_expiry():
    cache.set('key', 'value', ttl=1)
    sleep(1.1)
    assert cache.get('key') is None
```

**Integration test (with Redis):**
```python
def test_cache_with_redis():
    redis_conn = redis.Redis()
    cache = RedisCache(redis_conn)
    cache.set('key', 'value')
    assert cache.get('key') == 'value'
    redis_conn.delete('key')
    assert cache.get('key') is None
```

**Chaos test (simulate failures):**
```python
def test_cache_fallback_when_redis_down():
    with mock.patch('redis.Redis.get', side_effect=ConnectionError):
        result = get_with_cache('key', fallback_fn=expensive_compute)
        assert result == expensive_compute('key')
```
```

**Expectations:**
- Wrong/right examples should be code, not descriptions
- Edge cases should show actual failure modes (not hypothetical)
- Performance bottlenecks should have quantitative impact (not vague)
- Tests should be runnable and realistic

---

## 4. Sophisticated Interview Q&A Section

**Purpose:** Test architectural judgment at SDE3+ level (when, why, trade-offs) not pattern definitions.

**Length:** 500-600 words total

**Structure:**
- 8-12 questions per pattern
- Mix of:
  - Scenario-based questions with multiple valid approaches
  - Questions testing when NOT to use the pattern
  - Follow-up questions that dig deeper into trade-offs
  - Edge cases and handling strategies
  - Questions about judgment calls (when to violate the pattern)

### Example: For a hypothetical caching pattern

```markdown
## Sophisticated Interview Q&A

**Q1: Your cache hit rate is only 10%. Reduce?**

A: Not necessarily. Depends on cost of cache miss vs cache infrastructure.
- If cache miss costs $100 (expensive recomputation) and hit rate is 10% → cache saves $10/request → good ROI
- If cache miss costs $0.01 and hit rate is 10% → cache overhead likely exceeds benefit → consider removing cache
- Better question: What's the cost per cache miss vs cost per cache infrastructure?

**Follow-up:** Your cache has 1% hit rate. Worth keeping?

A: Probably not. But context matters: if 1B requests/month, 1% hit rate = 10M requests saved = $10M savings. Keep the cache.

---

**Q2: Caching would require schema changes. Still worth it?**

A: Depends on schema change scope. Small change (<1 day work) → probably yes if payoff is large. Large change (1-2 weeks) → need to show ROI.

Rule: refactoring cost should be <10% of annual savings.

Example: Adding `cached_at` timestamp field is 2 hours. If cache saves $100K/year → 2h cost is acceptable.

---

**Q3: Multiple caches (HTTP, Redis, in-memory). Which layers?**

A: Depends on access pattern and cost.
- HTTP cache (CDN): Cheapest, highest latency tolerance. Use for public data.
- Redis (distributed): Medium cost/latency. Use for warm data accessed by multiple servers.
- In-memory (local): Fastest, limited by memory. Use for very hot data only.

Example: User profile → HTTP cache (rarely changes), Redis cache (multiple API servers), in-memory (local LRU for hot users).

---

**Q4: Cache invalidation (Phil Karlton: "two hard things in CS"). How do you handle it?**

A: Multiple strategies, each with trade-offs:
- TTL-based: Simple, eventually consistent. Risk: stale data. Benefit: no logic.
- Explicit: Write code to invalidate on updates. Risk: miss invalidation (bugs). Benefit: fresh data.
- Event-driven: Publish invalidation events (Kafka). Risk: operational complexity. Benefit: consistent.

Choose based on freshness requirements and complexity tolerance.

---

**Q5: Your cache is out-of-sync with database. Recover?**

A: Depends on duration.
- If minutes (cache TTL is 1h): Wait for TTL to expire.
- If hours: Manually invalidate cache, rebuild from database.
- If days (bug): Audit what went wrong, fix code, rebuild, add monitoring.

Prevention: Add consistency checks (periodic random samples from cache vs DB, alert if mismatch).

---

**Q6: New cache layer adds latency. Why and how to fix?**

A: Network latency dominates. Cache lookup 10-50ms, local compute 1-5ms → cache is slower.

Solutions:
1. Remove cache (don't cache cheap operations)
2. Batch lookups (reduce round-trips)
3. Use faster cache (in-memory > Redis > database)

---

**Q7: Decide: implement cache or horizontally scale servers?**

A: Cache is cheaper. 
- Cache: $1K/month, reduces load 10x
- Servers: 10 servers × $5K = $50K/month

Cache wins by far. But complexity matters. Small team → cache complexity might be too high → scale servers instead.

---

**Q8: When to NOT use caching?**

A: Don't cache when:
- Operation is cheap (< 1ms). Cache latency overhead isn't worth it.
- Data changes frequently. Cache invalidation complexity exceeds benefit.
- Consistency is critical. Every read must be absolutely fresh.
- Small dataset. All data fits in memory without cache (just load it once).
```

**Expectations:**
- Questions test judgment (when/why/how to debug) not definitions
- Answers should have nuance (depends on context, trade-offs)
- Follow-up questions should dig into trade-offs
- Edge cases should be realistic

---

## 5. Cost & Resource Analysis Section

**Purpose:** Help engineers reason about financial and operational cost of the pattern.

**Length:** 250-300 words

**Structure:**
- Infrastructure cost model with formulas
- Operational overhead (time to maintain, run, debug)
- Cost optimization strategies
- ROI analysis or break-even analysis

### Example: For a hypothetical deployment pattern

```markdown
## Cost & Resource Analysis

### Infrastructure Cost Model

**Cost formula:**
```
Monthly cost = (server_cost × num_servers) + (storage_cost × GB) + (monitoring_cost)
```

**Example: Blue-green deployment (2 environments)**
```
Baseline system: 5 servers × $400 = $2,000/month
Blue-green overhead: 5 servers × $400 = $2,000/month (100% additional)
Total: $4,000/month
```

### Operational Overhead

**Time per deployment:**
- Prepare green environment: 15 min
- Run tests on green: 30 min
- Switch traffic: 5 min
- Monitor post-switch: 30 min
- **Total: 1.5 hours per deployment**

**Time per incident:**
- Detect issue: 5 min
- Rollback decision: 5 min
- Execute rollback: 5 min (instant, just config)
- **Total: 15 min vs 1-2 hours for other strategies**

**Personnel cost:**
- 10 deployments/week × 1.5 hours = 15 hours/week
- At $150/hour = $2,250/week = $9,000/month
- Rollback savings: 1 incident/month prevented = $900/month saved
- **Net monthly cost: $2K + $9K - $900 = $10.1K/month**

### Cost Optimization Strategies

1. **Use autoscaling:** Green only runs when deploying (30 min/day). Saves 95% cost.
   - Optimized cost: $2K baseline + $100 green = $2.1K/month (5% overhead vs 100%)

2. **Reduce validation time:** Better test automation. Reduces test time 30→10 min.
   - Saves: 20 min × 10 deployments/week × $150/hour = $500/month

### ROI Analysis

**Blue-green deployment justifies cost when:**
- Downtime cost > deployment cost
- Example: E-commerce site loses $10K/hour downtime. 1 incident/quarter prevented = $40K saved/quarter = $160K/year
- Blue-green cost: $10.1K/month = $121K/year
- **ROI: $160K savings >> $121K cost** ✓ Worth it

**Blue-green is overkill when:**
- Small startup with <$1K/hour downtime cost
- Better: Canary deployment ($20K/year) with higher (but acceptable) risk
```

**Expectations:**
- Costs should use real numbers (not made up)
- Should break down by component (not opaque totals)
- Personnel cost should be included (operations is expensive)
- ROI analysis should show when pattern is justified vs when it's overkill

---

## 6. Monitoring & Observability Patterns Section

**Purpose:** Provide operational guidance for running the pattern in production.

**Length:** 300-350 words

**Structure:**
- Specific metrics to instrument (not generic)
- Alert thresholds and strategies
- Health check implementations
- Debugging approach for issues in this pattern
- Dashboard template

### Example: For a hypothetical serving pattern

```markdown
## Monitoring & Observability Patterns

### Key Metrics to Instrument

```
Pattern-Specific Metrics:
- request_latency_ms (histogram with p50, p95, p99)
- request_throughput_qps (gauge)
- error_rate_pct (gauge)
- prediction_distribution (histogram of output values)
- model_load_time_sec (gauge)
```

**Why these metrics:**
- Latency p99: Users unhappy if 1% of requests are slow
- Throughput: Is system saturating?
- Error rate: Silent failures are worse than slow
- Distribution: Detect data drift (new models produce different distributions)

---

### Alert Thresholds & Strategy

**Critical alerts (page on-call):**
```
IF request_latency_p99 > 200ms (SLA breach)
  THEN alert with severity=CRITICAL

IF error_rate > 1% (users impacted)
  THEN alert with severity=CRITICAL

IF prediction_distribution shift KS-test > 0.05
  THEN alert with severity=CRITICAL
```

**Warning alerts (slack #incidents):**
```
IF request_latency_p99 > 150ms (trending toward breach)
  THEN alert with severity=WARNING

IF error_rate > 0.1% (early warning)
  THEN alert with severity=WARNING
```

---

### Health Checks

**Liveness (is process alive?):**
```python
@app.get("/health/live")
def liveness():
    return {"status": "alive"}
```

**Readiness (is system serving?):**
```python
@app.get("/health/ready")
def readiness():
    if not model_loaded:
        return {"status": "not_ready"}, 503
    return {"status": "ready"}
```

**Deep health (all dependencies working?):**
```python
@app.get("/health/deep")
def deep_health():
    checks = {
        "model_loaded": model is not None,
        "feature_store_accessible": feature_store.ping(),
        "database_accessible": db.ping(),
    }
    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        return {"status": "unhealthy", "checks": checks}, 503
```

---

### Debugging Approach

**If latency is high (p99 > 200ms):**
1. Profile: Break down latency by component (feature fetch, inference, post-processing)
2. Check: CPU/GPU utilization. If <50%, something blocking. If >90%, saturating.
3. Action: Optimize bottleneck (quantize model, optimize feature fetch, reduce batch size)

**If error rate is high (>1%):**
1. Check: Error logs. What's failing? OOM? Type mismatch? Feature missing?
2. If OOM: Reduce batch size or use smaller model
3. Action: Roll back to previous version immediately, debug offline

---

### Dashboard Template

```
Panel 1: Request Latency (line, p50/p95/p99)
Panel 2: Throughput (line, QPS)
Panel 3: Error Rate (line, %)
Panel 4: Prediction Distribution (histogram)
Panel 5: Dependency Health (status panel)
```
```

**Expectations:**
- Metrics should be specific to the pattern (not generic)
- Alert thresholds should have context (why that threshold?)
- Health checks should test actual dependencies (not just process alive)
- Debugging approach should be actionable (not generic troubleshooting)

---

## Summary: When You Have Doubt

This template defines the depth expected for SDE3-level pattern documentation. Use it as your reference:

- **Trade-off analysis:** Quantitative metrics, not narratives
- **Failure scenarios:** Include detection and recovery procedures
- **Implementation:** Both what to do and what NOT to do
- **Interview Q&A:** Test judgment, not memorization
- **Cost analysis:** Show financial impact clearly
- **Monitoring:** Specific and actionable

If you find yourself writing something shorter or vaguer than these examples, expand it using these examples as guides.

**Quality checklist:**
- ✓ Real metrics and numbers (not estimates)
- ✓ Production scenarios based on known incidents
- ✓ Code examples are concrete and runnable
- ✓ Interview Q&A tests judgment (when/why/how)
- ✓ Cost analysis is quantitative with examples
- ✓ Monitoring guidance is specific to the pattern
