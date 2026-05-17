# Data Validation: Quality Checks for Production Pipelines

## Comprehensive Overview

Data validation is the safety net that prevents garbage data from training models or serving predictions. A production data pipeline can ingest corrupted, incomplete, or malformed data without raising alarms—leading to silent model degradation. Data validation implements checks at each pipeline stage: schema validation (does data match expected types?), completeness checks (are required fields present?), statistical validation (does distribution match expected?), and anomaly detection (are there outliers?). Without validation, you discover problems when models degrade in production, not when data breaks.

The cost of silent data corruption is catastrophic. Netflix recommendations trained on stale data rank fewer titles (lower engagement). Uber pricing trained on missing traffic data charges wrong prices (user frustration). Stripe fraud detection trained on incomplete labels has blind spots (fraud losses). Each of these costs millions per day. Data validation catches these issues at the source, failing loud rather than silently.

Modern teams implement data contracts—explicit agreements between data producers and consumers about what data will look like. A contract specifies schema, freshness SLA, completeness guarantee, and acceptable value ranges. Producers commit to delivering data meeting contract; consumers depend on contract. Tools like Great Expectations (Python, open-source), Pandera (schema validation), and Soda (modern data testing) encode contracts as code, enabling automated validation and reporting.

The operational challenge is balancing false positives (valid data flagged as bad) against false negatives (invalid data flagged as good). Too strict and you block legitimate data; too loose and bad data slips through. The answer is layered validation: schema checks (strict), statistical checks (moderate), and anomaly detection (smart).

## How It Works

### Validation Layers

```
Raw Data
    ↓
Layer 1: Schema Validation (type, nullability, format)
    ↓
Layer 2: Completeness Checks (required fields present)
    ↓
Layer 3: Statistical Validation (distribution, outliers)
    ↓
Layer 4: Business Logic Checks (domain-specific rules)
    ↓
Validated Data
```

**Layer 1 (Schema):** Does each field have the expected type? Schema validation catches type mismatches early.

**Layer 2 (Completeness):** Are required fields present? % of nulls within acceptable range?

**Layer 3 (Statistical):** Do value distributions match historical patterns? Outlier detection flags anomalies.

**Layer 4 (Business Logic):** Do values make sense in business context? (e.g., price >= 0, shipping_days <= 30).

### Data Contract Example

```
Feature: user_7d_spend
Owner: payments_team
Schema: float (min: 0, max: 100,000)
Freshness: daily by 2am UTC
Completeness: 99%+ of users (1% acceptable nulls for new users)
Range: [0, 100,000] (values outside are errors)
Distribution: median $50, p95 $500 (historical pattern)
Alert: if <95% completeness or median >$100 (10x change)
```

## Tool Comparisons

| Tool | Approach | Strengths | Weaknesses | Best For |
|------|----------|-----------|-----------|----------|
| **Great Expectations** | Python, open-source | Flexible, large community, good docs, integrations | Can be verbose for complex checks | Python teams, flexible validation |
| **Pandera** | Schema-first | Simple schema definition, type hints, Pandas-native | Limited to Pandas, early-stage tool | Data science workflows, exploratory work |
| **Soda** | Modern, cloud-native | User-friendly YAML, SaaS with monitoring, quick setup | SaaS vendor lock-in, costs for scale | Quick validation setup, modern teams |
| **Custom (SQL/dbt)** | SQL-based | Integration with existing data warehouse, SQL-native | High maintenance, no standardization | Warehouse-native teams, SQL shops |
| **Spark Data Validation** | Spark ecosystem | Integrates with Spark, handles big data | Less mature, smaller community | Spark shops, large-scale validation |

**Decision Framework:**
- **Python teams:** Great Expectations (flexibility)
- **Modern, quick start:** Soda (user-friendly)
- **Schema-heavy:** Pandera (Pandas-native)
- **SQL shops:** dbt tests (SQL-native)

## Interview Q&A

**Q: Your data pipeline processes 1M transactions/day. How would you design a data validation strategy to catch corruption early?**

A: Layered approach: (1) Schema validation—each field matches expected type, range. (2) Completeness—required fields present >99%. (3) Statistical—value distributions match historical (e.g., median spend), flag 5x outliers. (4) Business logic—price >= 0, quantity > 0. (5) Monitoring—alert if any check fails. (6) Alerting—on-call engineer investigates within 1 hour. Automation: run validation before loading to warehouse, block bad data.

**Q: Data validation blocked 5% of today's data for anomalies. Should you allow it or investigate?**

A: Investigate first: (1) Is this real data corruption or false positive? (2) Check source—did data producer make changes? (3) Check distribution—is it 5x normal, 5.1x? (4) If real anomaly: investigate root cause. (5) If false positive: loosen validation threshold. Never silently allow anomalies; could be real corruption. Better to block and investigate than silently train on bad data.

**Q: How would you implement data contracts between data producers and consumers?**

A: Contract specifies: schema (types, ranges), freshness (update frequency), completeness (% nulls acceptable), distribution (expected ranges). Implement: (1) Version contract (like code). (2) Encode in validation tests (Great Expectations, dbt). (3) Monitor: alert if contract violated. (4) Ownership: producer commits to meeting contract, consumer depends on contract. (5) Evolution: deprecate old contracts, introduce new ones gradually.

**Q: Your model's accuracy dropped. Validation didn't catch anything. What went wrong?**

A: Validation catches schema/completeness issues, not all failures. Investigate: (1) Did data distribution shift subtly? (e.g., median spend 5% higher, still within acceptable range). (2) Did upstream transformation change? (different calculation). (3) Did labels become stale? (training data freshness). (4) Did feature became unavailable? (replaced with default). Consider: schema validation is strict (catches obvious errors), statistical validation is moderate (catches distribution shifts), anomaly detection is smarter. May need to tighten thresholds or add business logic checks.

**Q: Validation is slowing down your pipeline. Latency went from 10 min to 30 min. How do you optimize?**

A: Profile validation: which checks are slow? (1) Schema validation: usually fast. (2) Statistical validation: slow (requires computing percentiles). (3) Anomaly detection: slow (can be complex). Optimize: (1) Parallelize checks (run in parallel, not serial). (2) Sample data (validate sample instead of full dataset). (3) Remove expensive checks (is that statistical check necessary?). (4) Use fast storage (in-memory vs disk). (5) Lazy evaluation (skip checks for known-good data). (6) Async validation (validate after loading, don't block pipeline).

## Best Practices

1. **Start with Schema:** Simple schema checks catch 80% of errors. Establish schema first, add statistical checks as needed.

2. **Data Contracts:** Formalize expectations in contracts. Producers commit to delivering data meeting contract.

3. **Layered Validation:** Schema (strict) → Completeness (moderate) → Statistical (smart). Each layer catches different errors.

4. **Monitor Validation:** Track validation metrics (% pass rate, % null, distribution). Alert on anomalies.

5. **Fail Loud:** Don't silently skip bad data. Alert and block. Investigate root cause.

6. **Version Validation:** Validation rules change as data evolves. Version them, enable rollback.

7. **Balance False Positives:** Too strict and valid data is blocked; too loose and bad data slips through. Find balance.

## Common Pitfalls

1. **Validation Too Strict:** Blocking 1% of data on false positives slows pipeline. Loosen thresholds.

2. **Silent Failures:** Data fails validation but slips through anyway. Fail loud, investigate root cause.

3. **Validation Too Slow:** Validation adds 20+ min to pipeline. Optimize or move to async.

4. **No Monitoring:** Validation runs but nobody's watching. Alert on failures.

5. **Brittle Rules:** Validation rules hardcoded, break when data naturally evolves. Use flexible thresholds.

## Real-World Examples

### Netflix: Validation for Recommendation Data

Netflix validates user viewing data:
- Schema: user_id (int), title_id (int), watch_time (float 0-1), timestamp
- Completeness: 99%+ of required fields
- Statistical: watch_time distribution matches historical
- Business logic: timestamp <= now (no future data)
- Failure: alert fraud team, investigate

### Stripe: Transaction Validation

Stripe validates transaction data:
- Schema: amount (float >0), merchant_id, user_id, timestamp
- Completeness: all fields required
- Statistical: amount distribution (median $50, p99 <$10,000)
- Business logic: amount within merchant's typical range
- Failure: block transaction, investigate with merchant

### Uber: Ride Data Validation

Uber validates ride data:
- Schema: driver_id, rider_id, distance, duration, fare
- Completeness: all fields required
- Statistical: fare matches historical for distance
- Business logic: duration > distance/60 (sanity check)
- Failure: flag pricing anomaly, investigate

## Sample Interview Questions

1. "Design a data validation strategy for a financial services company processing 10M transactions/day. What checks would you implement?"

2. "Your validation is blocking 2% of data daily. Is this acceptable? How would you investigate?"

3. "How would you prevent your validation from becoming a bottleneck as data volume grows?"

## Interview Case Study

**Scenario:** You're at Capital One building validation for credit card transaction data.

**Context:** 1M+ transactions/day, fraud detection models rely on fresh, clean data.

**Challenge:** Recently, data corruption (null amounts, future timestamps) slipped through, degrading fraud models.

**Design validation that prevents recurrence:**

1. Schema validation (strict): amount >0, required fields present
2. Completeness: 99.9% of transactions have all fields
3. Temporal: timestamp <= now, no future transactions
4. Statistical: amount distribution matches historical
5. Business logic: amount within merchant's typical range
6. Monitoring: alert on any validation failure

---

## Related Concepts

- **Concept 01:** Data Pipelines — Where validation runs
- **Concept 02:** Feature Stores — Validating features before serving

## Resources

- Great Expectations: https://greatexpectations.io/
- Pandera: https://pandera.readthedocs.io/
- Soda: https://www.soda.io/
