# SDE3 Pattern Enhancement Template

This template provides comprehensive examples of what each section of an SDE3-level pattern enhancement should contain. Use this as the reference for enhancing all 13 system design patterns.

---

## 1. Detailed Trade-off Analysis

### Purpose
Provide quantitative comparison of major variants/approaches with clear decision criteria.

### Example: Data Pipeline Trade-offs

| Aspect | Batch Processing | Stream Processing | Hybrid (Kappa/Lambda) |
|--------|------------------|-------------------|----------------------|
| **Latency** | Hours → Days | Milliseconds → Seconds | Minutes (stream) + fallback |
| **Throughput** | High (billions/day) | Moderate (millions/sec) | Very High (both paths) |
| **Infrastructure Cost** | Low (scheduled jobs) | Medium (always-on cluster) | High (dual infrastructure) |
| **Operational Complexity** | Low (batch jobs, easy to debug) | High (distributed, state mgmt) | Very High (two systems to manage) |
| **Data Consistency** | Strong (processing window) | Eventual (late-arriving fixes) | Strong (batch) + Eventual (stream) |
| **Debugging** | Easy (replay any time) | Hard (distributed state) | Mixed (easier for batch path) |

**Cost Breakdown (Example: 100M records/day):**

Batch Processing:
- Compute: 20 hours × 100 cores × $0.10/hour = $200/day
- Storage: 1TB logs + 100GB processed = $5/day
- Total: ~$205/day

Stream Processing:
- Kafka cluster: 3 brokers × $300/month = $10/day
- Processing: 50 containers × $0.05/hour × 24 = $60/day
- Storage: same = $5/day
- Total: ~$75/day

**Decision Matrix:**

| Scenario | Recommendation | Reasoning |
|----------|----------------|-----------|
| Analytics dashboards (hourly freshness OK) | Batch | Low cost, simple operations, accuracy over speed |
| Real-time personalization (sub-second) | Stream | Latency critical, cost secondary |
| Financial transactions (strict consistency + speed) | Hybrid + Batch | Batch for accuracy, stream for speed, reconcile nightly |
| A/B testing features (minutes-level) | Stream only | Simplicity + speed without batch overhead |

---

## 2. Production Failure Scenarios

### Purpose
Document real failures that happen in production with detection methods and recovery procedures.

### Example: Feature Store Failure Scenarios

**Scenario 1: Feature Store Completely Down**

What breaks: All feature lookups fail → Model predictions blocked → Online serving stalls

Why it happens: Unplanned database failover, disk full, network partition, or replica lag exceeds timeout

Detection:
```
Alert: "Feature store query latency p99 > 5s OR error_rate > 1%"
```

Recovery (15-30 minutes):
1. Switch to stale feature cache (up to 24h old)
2. Database team repairs primary or promotes replica (5-15 min)
3. Smoke test queries before resuming (5 min)
4. Gradual rollout: 10% → 50% → 100%

Prevention: Multi-region replication, feature caching layer, monthly DR drills, connection timeouts (5s max)

---

**Scenario 2: Stale Features Due to Indexing Lag**

What breaks: Features computed yesterday treated as today's → Accuracy drops 2-5%

Why it happens: Batch job fails silently, upstream dependency missing, delayed data arrival

Detection:
```
Alert: "Feature freshness: max(timestamp) < NOW - 25 hours"
```

Recovery (30-60 minutes):
1. Monitoring catches feature age > 1 day (5 min)
2. Page oncall (immediately)
3. Check batch logs and dependencies (10 min)
4. Manually trigger backfill (20-40 min)
5. Validate features match upstream (5 min)

Prevention: Freshness SLO 95% < 6h old, automatic dead-letter queue, redundant computation, dependency health checks

---

**Scenario 3: Offline/Online Feature Divergence**

What breaks: Training uses batch features (yesterday), serving uses cached features (1 hour old) → Train-serve skew → Accuracy drops 3-10%

Why it happens: Different code paths for batch vs online, schema change in one path, transformation bug

Detection:
```
Alert: "Batch feature != Online feature for 5%+ of keys"
```

Recovery:
- Quick (5-30 min): Revert online change, redeploy
- Medium (1-2 hours): Fix batch, recompute
- Long (4-8 hours): Unified feature library

Prevention: Unified code library, feature parity tests in CI, post-deployment monitoring for 24h, gradual rollout with parity checks

---

**Scenario 4: Feature Cache Memory Explosion**

What breaks: Cache memory 1TB (budgeted 50GB) → Hit rate drops 95% → 45% → DB load doubles → Queries timeout

Why it happens: High-cardinality feature added, cache TTL too high (24h vs 2h), no eviction policies

Detection:
```
Alert: "Cache memory > 80% of limit OR hit_rate < 70%"
```

Recovery (15-30 minutes):
1. Reduce TTL for high-cardinality features (2-3 min)
2. Increase cache size limit if budget allows (5-10 min)
3. Implement LRU eviction (1-2 hours)
4. Redesign high-cardinality feature with bucketing (day+)

Prevention: Cardinality testing before rollout, cache size budgeting: max_cache = max_QPS × avg_size × TTL, automatic TTL tuning, hit rate SLO 90%+

---

## 3. Implementation Guidance & Gotchas

### Purpose
Show common mistakes and how to avoid them with code examples.

### Example: Model Serving Pattern

**Gotcha 1: Not Handling Slow Preprocessing**

❌ Wrong: 150ms latency per request, requests queue under load

✅ Right: Pre-compute heavy features once or use async preprocessing. Latency: 100-120ms with overlap

---

**Gotcha 2: Not Handling Model Loading**

❌ Wrong: Model loads on first request → 2 seconds first request → Client timeout at 500ms

✅ Right: Load at container startup, warmup with 100 predictions to compile. All requests fast.

---

**Gotcha 3: Not Batching Requests**

❌ Wrong: batch_size=1 → ~10 req/sec throughput, 3% GPU utilization

✅ Right: Queue requests, batch size 32. Throughput ~300 req/sec, 80%+ GPU utilization. 30x improvement.

---

**Gotcha 4: Not Monitoring Model Version Mismatch**

❌ Wrong: V1 and V2 running simultaneously during rollout, no version tracking, A/B test metrics confounded

✅ Right: Track model_version in every log. Monitor: requests per version, latency per version, error rate per version, prediction distribution per version.

---

**Edge Case: Out-of-Distribution Requests**

```
Check if any feature is >5 sigma from training mean.
If yes: Return uncertainty estimate instead of prediction,
log warning, optionally route to human review.
```

**Performance Bottleneck: Serialization Overhead**

Use efficient serialization:
- protobuf: 5-10 µs, schema validation
- msgpack: 10-20 µs, flexible
- JSON: 100+ µs, avoid for high throughput

---

## 4. Sophisticated Interview Q&A

### Purpose
Test judgment, architectural thinking, and production knowledge—not memorization.

### Example Interview Questions

**Q1: Real-time personalization (sub-second latency) with strong consistency and compliance tracking. Batch, stream, or hybrid? Why?**

A: Trick question testing trade-off boundaries. Batch can't meet latency. Pure stream sacrifices consistency.

**Correct answer: Hybrid** (if latency budget 100-5000ms):
- Stream for speed
- Batch for compliance audit log and reconciliation
- Unified format for consistency tracking

If <100ms required: Stream only, use checksums to detect drift.

Why tested: Not about definitions, but understanding when consistency matters (compliance=always), when it doesn't (personalization=eventual OK), and latency under load.

---

**Q2: Feature store queries normally 50ms, spike to 2s. Diagnose: (A) DB slow, (B) network slow, (C) computation slow, or (D) lock contention?**

A: Instrument each layer with latency breakdown. Then rule out:
- **(D) Lock contention (most common):** lock_wait_time > 1s → Add replicas or lock-free reads
- **(A) DB query:** Check DB metrics (CPU %, slow log) → Add indexes, partition, cache
- **(B) Network:** Check network metrics, connection pooling, keepalive → Fix connections
- **(C) Computation:** If none above, profile code → Rare if pre-computed

Why tested: Real production debugging—what to measure, how to rule out possibilities, understanding second-order effects of fixes.

---

**Q3: Batch pipeline runs daily at 2am. Next 3 months: 10x volume (100GB → 1TB). What changes, in what order?**

A: Scaling checklist in ROI order:

1. **Optimize current job (cheapest):**
   - Parallelism (1 → 8 cores): 10x gain
   - Compression: 2-3x I/O improvement
   - Caching: Reduce redundant computation

2. **Storage layer (next cheapest):**
   - Format: Parquet vs CSV vs ORC (5-10x faster)
   - Partitioning: By date/region (10x I/O reduction)
   - Indexing: Secondary indexes if needed

3. **Infrastructure (most expensive, last resort):**
   - Cluster: 4 nodes → 16 nodes (4x cost for 10x data)
   - Disk: SSD vs HDD
   - Network: Upgrade bandwidth

**Execution order:** Month 1 optimize (3-5x), Month 2 storage (2-3x), Month 3 infra if needed (usually not).

Why tested: Where real bottlenecks are (algorithmic > storage > infra). Being pragmatic (cheapest first). Understanding cost/benefit trade-offs rather than "add more machines."

---

**Q4: Describe failure mode where pipeline works perfectly but produces wrong answers unnoticed for 6 months.**

A: Test awareness of subtle bugs (logic errors, not infra):

1. **Silent schema change:** Upstream adds column 101. Code takes first 100. Silently drops important column. Results subtly wrong (slightly different distributions). Unnoticed until revenue drops 2%.

2. **Time zone bug:** Daily aggregates don't handle DST. One day 25h, another 23h. Consistently off 4-5%. Noticed only at year-end reconciliation.

3. **Data stale for one region:** US updates daily, EU weekly (compliance). Reports show "as of yesterday" but EU is 5 days old. Visible only when EU revenue compared to actual.

4. **Aggregation function change:** mean() → median() for outliers. Results drop 2%. Nobody investigates (no errors). Noticed only comparing to external benchmark.

Prevention: Output validation vs historical ranges, schema validation, time zone tests at DST, version control transformations.

Why tested: Infra reliability ≠ correctness. Separates engineers shipping features quickly from shipping correctly.

---

## 5. Cost & Resource Analysis

### Purpose
Quantify operational and financial costs of different approaches with ROI analysis.

### Example: Model Serving Cost

**Infrastructure Cost Model:**

Batch Serving:
- Compute: 100 req/sec × 100ms = $0.0014/req
- Storage: $0.10/month
- Network: $150/month

Online Serving:
- Compute: $0.007/req (lower batch utilization)
- Server: $500/month
- Network: $150/month

**At 1 million requests/day:**
- Batch: $1,420/day (5x cheaper)
- Online: $7,000+/day

But if latency SLA <100ms required: Must use online (batch takes 5-10 seconds)

**ROI Analysis for Caching:**

Cache cost: 3 nodes × 256GB × $0.08/GB/month = $60/month

Cache benefit:
- 70% hit rate saves $200/month in compute
- Latency: 10ms (cache) vs 100ms (model) = 10x faster

**ROI: $200/month benefit / $60/month cost = 3.3x return** → Clear win

---

## 6. Monitoring & Observability Patterns

### Purpose
Define specific metrics to instrument, alert thresholds, and debugging approaches.

### Example: Feature Store Monitoring

**Key Metrics:**

1. **Latency:** query_latency_p50/p95/p99, lock_wait_time, database_time, network_time
   - Alert: p99 > 1s OR p95 > 500ms

2. **Data Quality:** feature_freshness, missing_rate, null_rate, out_of_range
   - Alert: freshness > 25h OR missing > 1% OR null > 5%

3. **Cache:** hit_rate (target >80%), eviction_rate, memory_usage
   - Alert: hit_rate < 70% OR memory > 90%

4. **Correctness:** offline_online_parity, schema_mismatch, cardinality_drift
   - Alert: parity < 95% OR schema_mismatch > 100

**Alert Thresholds & Escalation:**

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Query p99 latency | > 500ms | > 2s | Page oncall, check locks/DB |
| Feature freshness | > 12h | > 24h | Trigger manual recompute |
| Cache hit rate | < 75% | < 50% | Increase cache or debug |
| Offline/online parity | < 98% | < 95% | Halt, investigate |
| Error rate | > 0.1% | > 1% | Rollback |

**Health Check:**
- Measure p99 latency over last minute
- Verify features updated in last 6 hours
- Spot-check 100 keys for offline/online parity
- Return unhealthy if any check fails

**Debugging Common Issues:**

Issue: Queries suddenly slow
→ Check lock_wait_time, DB CPU, network latency, cache hit rate

Issue: Features stale
→ Check batch logs, upstream dependencies, trigger manual recompute

Issue: Train-serve skew (offline != online)
→ Compare code paths, test with same data, measure parity before/after deploy, use unified library

---

## Summary Checklist

When enhancing a pattern to SDE3 level:

✅ **Detailed Trade-off Analysis** — Quantitative comparison, cost breakdown, decision matrix

✅ **Production Failure Scenarios** — 4-6 real failures, detection, recovery, prevention

✅ **Implementation Guidance & Gotchas** — 4-6 mistakes with code, edge cases, bottlenecks, testing

✅ **Sophisticated Interview Q&A** — 8-12 judgment questions, scenario-based, follow-ups

✅ **Cost & Resource Analysis** — Infra model with formulas, ROI analysis, break-even points

✅ **Monitoring & Observability** — 15-20 metrics, alert thresholds, health checks, debugging approaches

Once all sections complete and pass checklist, pattern is **SDE3-ready**.
