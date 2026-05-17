# Case Study: Stripe Fraud Detection Pipeline

## Interview Scenario

**Setup:** You're at a payment company (like Stripe) processing 1M+ transactions/day. Design a data pipeline for fraud detection that must flag fraudulent transactions in <100ms.

**Constraints:**
- 1M transactions/day = ~10/second
- Must score in <100ms (real-time serving requirement)
- Fraud labels confirmed 5 days after transaction (label delay)
- Model must be retrained daily with new fraud confirmations
- Cost-sensitive (compute and storage are expensive)

**Context:**
- Historical data: 1 year of transactions + confirmed fraud labels
- Real-time data: incoming transaction events (Kafka)
- External data: transaction amounts, merchant info, user history

---

## Strong Answer Walkthrough

### 1. Architecture Overview

**Two-path design (batch + streaming):**

```
┌─────────────────────────────────────┐
│      Batch Path (Daily Training)     │
│  ────────────────────────────────── │
│ Kafka (5 days of events)             │
│     ↓                                │
│ Fraud labels (confirmed 5 days ago)  │
│     ↓                                │
│ Feature engineering (Spark)          │
│ - User transaction history           │
│ - Merchant fraud rate                │
│ - Amount anomaly                     │
│     ↓                                │
│ Train fraud detection model          │
│ - Historical features + labels       │
│ - Deploy new model version           │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│   Streaming Path (Real-Time Serving)  │
│  ────────────────────────────────── │
│ Real-time transaction events (Kafka) │
│     ↓                                │
│ Flink: compute real-time features    │
│ - Transaction velocity (5min)        │
│ - Geographic anomaly                 │
│ - Device anomaly                     │
│     ↓                                │
│ Cache in Redis (<5ms lookup)         │
│     ↓                                │
│ Score with deployed model            │
│ - Batch features (20ms)              │
│ - Real-time features (15ms)          │
│ - Scoring (10ms)                     │
│ Total: <100ms ✓                      │
└─────────────────────────────────────┘
```

### 2. Design Decisions & Rationale

**Decision 1: Separate batch and streaming**
- **Why:** Different latency requirements. Batch can afford to be slow (daily job, 10s of minutes OK). Real-time must be fast (<100ms).
- **How:** Batch uses Spark (slower but scalable), streaming uses Flink (optimized for latency).

**Decision 2: Handle label delay with 5-day lookback**
- **Why:** Fraud confirmations arrive 5 days after transaction. Can't train on today's transactions (labels unknown).
- **How:** Train daily on transactions from 5 days ago. Use unsupervised anomaly detection for real-time gap.

**Decision 3: Combine batch and real-time at serving time**
- **Why:** Different features have different freshness requirements.
- **How:** Batch features (expensive, historical) computed daily, stored in warehouse. Real-time features (cheap, streaming) computed on-demand, cached. Combine at inference.

### 3. Detailed Implementation

**Batch Pipeline (Daily Job):**
```
1. Ingest confirmed fraud labels from 5 days ago
   - Query database: transactions from 5 days ago
   - Join with fraud_labels table (confirmed on day 5)
   - Only process transactions with confirmed labels

2. Feature engineering (Spark)
   - User features: account age, transaction count (last 30 days), avg amount
   - Merchant features: typical transaction volume, fraud rate
   - Transaction features: amount relative to user baseline, category

3. Train model
   - Historical labeled data (1 year of transactions)
   - XGBoost with batch features
   - Train on 90%, validate on 10%
   - Track accuracy, precision, recall

4. Deploy
   - Serialize model to S3
   - Version: model_v123 (daily increment)
   - Health check: test on holdout validation set
   - Deploy to serving infrastructure
```

**Real-Time Scoring (Online Serving):**
```
1. Transaction arrives in Kafka
   
2. Fetch batch features (20ms)
   - User ID → lookup user account age, transaction count
   - Merchant ID → lookup merchant fraud rate
   - From cache (Redis)
   
3. Compute real-time features (15ms, Flink)
   - User transaction velocity: count in last 5 minutes
   - Geographic check: is country unusual for this user?
   - Device check: is device new to user?
   
4. Score transaction (10ms)
   - Load model from memory
   - Feed batch + real-time features
   - Output: fraud probability (0-1)
   
5. Decision
   - p > 0.8: DECLINE
   - 0.5 < p < 0.8: CHALLENGE (2FA)
   - p < 0.5: APPROVE
```

### 4. Key Decisions Explained

| Decision | Why | Trade-off |
|----------|-----|-----------|
| Batch only (no real-time) | Simpler, easier to implement | Slower, stale labels |
| Streaming only (no batch) | Real-time, fresher decisions | Can't use historical patterns |
| Batch + Streaming (chosen) | Balance: history + real-time | More complex architecture |
| Label delay (5 days) | Reality of fraud confirmation | Can't train on today's data |
| Cache batch features | Fast lookup (<5ms) | Staleness (1-day old) |
| Unsupervised for real-time | No labeled data available | Less accurate than supervised |
| Probabilistic scoring | Enables dynamic thresholds | Can't guarantee zero fraud |

### 5. Monitoring & Alerting

**Daily Metrics:**
- Model accuracy: should stay >95%
- Precision: % of flagged transactions actually fraud
- Recall: % of actual frauds caught
- Latency: p99 should be <100ms

**Alerts:**
- Accuracy drops >5%: trigger retraining investigation
- Latency p99 > 200ms: alert ops team
- Cache hit rate <90%: performance degradation

### 6. Failure Scenarios & Responses

| Scenario | Response |
|----------|----------|
| Model accuracy drops to 90% | Investigate: retrain on recent data? Feature distribution changed? Revert to v(n-1) if better. |
| Real-time features unavailable | Fallback: use batch features only, adjust thresholds. |
| Kafka lag accumulating | Increase Flink parallelism, add compute resources. |
| Redis cache fills up | Implement LRU eviction, reduce TTL, prioritize hot features. |
| Fraud pattern changes | Unsupervised anomaly detection flags new patterns, retrain. |

---

## Strong vs Weak Answers

### STRONG Answer
"I'd design a two-path system. **Batch:** Daily Spark job ingests fraud labels (5 days delayed), computes user/merchant/transaction features, trains fraud model. Stores in warehouse with model versioning. **Real-time:** Kafka streams feed Flink, which computes velocity, geographic, device anomalies. Cache in Redis. At inference: batch features (20ms) + real-time (15ms) + scoring (10ms) = <100ms. Handles label delay by training on confirmed labels only. Falls back to unsupervised anomaly detection for real-time. Monitor: track accuracy, latency, cache hit rate. Daily retraining ensures model stays fresh."

**Why this is strong:**
- ✓ Addresses label delay explicitly
- ✓ Explains latency budget breakdown
- ✓ Handles batch + streaming separately (correct architecture)
- ✓ Discusses versioning and monitoring
- ✓ Shows understanding of trade-offs

### WEAK Answer
"I'd build a feature store with Feast, train an XGBoost model, and deploy it."

**Why this is weak:**
- ✗ Doesn't address 5-day label delay (critical constraint)
- ✗ Names tools without explaining architecture
- ✗ No discussion of batch vs streaming
- ✗ No latency analysis (critical requirement: <100ms)
- ✗ No mention of real-time features or fallbacks
- ✗ Ignores cost implications

---

## Follow-Up Questions

**Q: Model accuracy degraded from 97% to 92%. How would you investigate?**

A: (1) Check data: did fraud patterns change? (2) Retrain on last month's data: if accuracy returns to 97%, it's data. (3) Compare current features vs historical: are distributions different? (4) Check feature pipeline: did upstream data source change? (5) If real pattern shift: update thresholds or add new features. (6) If code issue: check recent deployments, rollback if needed.

**Q: Real-time latency went from 50ms to 200ms. Debug it.**

A: Profile each stage: (1) Batch feature fetch: check Redis latency and hit rate. (2) Real-time compute: check Flink lag. (3) Model loading: check memory, preload vs lazy load. (4) Network: check bandwidth. Once bottleneck identified, optimize. Example: if Redis slow, increase replication, add more instances. If Flink lagging, increase parallelism.

**Q: Cost is 10x higher than expected. How would you optimize?**

A: (1) Reduce batch job frequency: can we train every 2 days instead of daily? (2) Sample transactions: validate on 10% instead of 100% during training. (3) Compress features: store as int16 instead of float32. (4) Reduce cache TTL: accept slight staleness (1 hour instead of 24). (5) Prune features: which features have highest importance? Remove low-impact ones.

**Q: How do you handle weekend/holiday patterns?**

A: Fraud patterns differ: weekends have different velocity profiles, holidays spike certain categories. Solution: (1) Include day-of-week feature. (2) Train separate models for weekday/weekend. (3) Detect holidays and use different thresholds. (4) Use seasonality-aware features (compare to same day last year, not yesterday).

---

## Key Takeaways

**Concepts Demonstrated:**
- ✓ Batch vs streaming trade-offs
- ✓ Handling label delays
- ✓ Real-time latency budgeting
- ✓ Feature versioning and governance
- ✓ Monitoring and alerting
- ✓ Failure scenarios and recovery

**Why This Design Wins:**
- Addresses all constraints explicitly (label delay, <100ms, cost)
- Separates concerns (batch training, real-time serving)
- Shows understanding of production challenges
- Includes monitoring, alerting, and failure handling
- Explains trade-offs clearly

**Next Steps in Interview:**
- Interviewer will likely ask about:
  - Handling the label delay (this answer covers it)
  - Debugging latency (this answer covers it)
  - Cost optimization (this answer covers it)
  - Scaling to 100M transactions/day (scale batch to 1000s of machines, Flink to 100s)
  - New fraud patterns (unsupervised anomaly detection + retraining)
