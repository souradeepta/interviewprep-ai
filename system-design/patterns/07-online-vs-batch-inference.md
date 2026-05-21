# Online vs Batch Inference

## Detailed Description

Two serving modes: Batch (offline, precompute all predictions), Online (real-time, compute on-demand). Batch is cheap but stale. Online is fresh but expensive. Choose based on: latency requirement, cost tolerance, freshness need.

## Core Intuition

Batch = precompute all predictions overnight, cache results, serve instantly. Online = compute per-request in <100ms. Batch cheap (one compute pass), stale (24h old). Online fresh (real-time), expensive (per-request GPU time). Hybrid: batch for base predictions, online for personalization.

## How It Works

**Batch Inference:**
- Nightly: compute predictions for all 1M users
- Store in database
- App: instant lookup in <1ms

**Online Inference:**
- User arrives → extract features → run inference → response
- Latency: <100ms end-to-end
- Fresh: always using current user data

| Aspect | Batch | Online |
|--------|-------|--------|
| Latency | <1ms lookup | <100ms compute |
| Cost | Low (compute once/day) | High (per-request) |
| Freshness | 24h old | Real-time |
| Scalability | Scales with users | Scales with requests |
| Personalization | Limited | Full |

## Key Properties / Trade-offs
- Batch: cheap, stale, high storage cost
- Online: expensive, fresh, low storage
- Hybrid: batch base + online personalization (best of both)

## Detailed Trade-off Analysis

| Aspect | Batch (Offline) | Online (Real-time) | Hybrid |
|--------|-----------------|-------------------|--------|
| **Latency** | 1-24 hours | 50-200ms | 1-24h batch + 50ms online |
| **Cost/prediction** | $0.001 | $0.01 | $0.002 (mostly batch) |
| **Freshness** | Hours to 1 day old | Real-time (current) | Mixed: base 24h old, delta real-time |
| **Personalization** | Limited (from yesterday's data) | Full (current user state) | Partial (batch base + online boost) |
| **Scalability** | Scales with users (volume) | Scales with QPS (requests/sec) | Best of both |
| **Failure mode** | Stale predictions available | Complete outage if model/GPU down | Batch fallback for failures |
| **Implementation complexity** | Low (scheduled job) | High (real-time serving, GPUs) | Medium (two systems) |
| **Storage cost** | High (1M users × 1KB = 1GB) | Low (no storage) | Low (only batch) |
| **Model update lag** | If batch is daily, updates lag 24h | Immediate (next inference) | Batch lags, stream immediate |

### Cost Model (1M active users, 5 predictions/user/day)

**Batch Inference (nightly):**
- Compute: 1M users × 5K features = 5M tokens (GPU inference)
  - 1 GPU hour × $0.50/hour = $50/day = $1,500/month
- Storage: 1M × 1KB prediction = 1GB (S3: $0.023/GB) = $0.02/month
- Personnel (monitoring): 2 hours/month = $300/month
- **Total: $1,800/month**

**Online Inference (per-request):**
- QPS: 5M predictions/day / 86,400 seconds = ~58 QPS peak
- GPU inference: 58 QPS × 50ms = 3 GPU hours/day
- Cost: 3 GPU hours × $0.50/hour × 30 days = $4,500/month
- Feature extraction: 58 QPS × async CPU = ~$500/month
- Personnel (on-call, optimization): 10 hours/month = $1,500/month
- **Total: $6,500/month** (3.6x more expensive than batch)

**Hybrid (Recommended):**
- Batch: 90% of predictions ($1,620/month)
- Online: 10% of predictions ($450/month)
- De-duplication: If same user predicted both batch+online, cache result ($50/month)
- **Total: $2,120/month** (1.2x batch, 32% of pure online)

### Decision Matrix

| Scenario | Recommendation | Reasoning | Cost/Month |
|----------|----------------|-----------|-----------|
| **E-commerce recommendations** | Batch or Hybrid | Users check daily/hourly. 24h old OK for most. Cost-sensitive. | $1.8K batch / $2.1K hybrid |
| **Real-time fraud detection** | Online | Per-transaction (sub-second latency critical). Cost secondary. | $4-6K |
| **Email campaigns** | Batch | Send once/day. 24h stale acceptable. High volume (1M+ emails). | $1.8K |
| **Mobile app personalization** | Hybrid | Batch base (fast load), online personalization (current session). | $2.1K |
| **Search ranking** | Online | Per-query, user expects fresh results immediately. | $5-10K depending on QPS |
| **Churn prediction** | Batch | Run weekly. Update models daily. User notification batch (daily email). | $500 (weekly batch) |

---

## Production Failure Scenarios

**Scenario 1: Batch Job Misses SLA (Data Not Ready)**

**What breaks:** Batch job scheduled for 2am, completes at 8am. Users request recommendations at 6am, get yesterday's cached predictions (48h old instead of 24h). For volatile data (trending products), recommendations are stale.

**Detection:**
```
Alert: if (batch_completion_time > 7am) → WARN (SLA breach)
Check: SELECT MAX(computed_at) FROM predictions WHERE date=today
```

**Recovery:**
- Immediate (5 min): Serve yesterday's cached predictions (already available)
- Short-term (2-4 hours): Investigate batch bottleneck, restart if transient
- Fallback: Switch to hybrid (use online inference for subset of users)

**Prevention:** Monitor batch job trending toward SLA. Increase resource allocation if growing.

---

**Scenario 2: Online Inference Too Slow (Model Inference Timeout)**

**What breaks:** Online model takes 500ms to run (was 50ms). Requests timeout after 200ms. Users get errors or fallback predictions. Affects all real-time requests.

**Why it happens:**
- Model grew larger (fine-tuning added parameters)
- GPU memory pressure (sharing with batch job)
- Feature extraction slow (network latency to feature store)

**Detection:**
```
Metric: online_inference_latency_p99
Alert: if (p99 > 200ms) → WARN
```

**Recovery:**
1. Profile: Is it model inference slow or feature extraction?
2. If model slow:
   - Quantize model (FP32 → INT8): 4-5x speedup
   - Batch requests: accumulate 10 requests, compute together
   - Use smaller model: trade accuracy for speed
3. If feature extraction slow: add caching, query feature store in batch

**Prevention:** Latency SLA testing before deploy. Load test with real QPS.

---

**Scenario 3: Batch and Online Predictions Diverge**

**What breaks:** User sees recommendation A in batch email (sent 2am), then logs in at 10am and sees recommendation B in online (real-time). Both should be similar (same ranking logic), but differ by 20% (different feature versions).

**Why it happens:**
- Batch uses old feature computation logic (v1.0)
- Online uses new feature computation logic (v1.1)
- Same model, different inputs = different outputs

**Detection:**
```
Alert: if (divergence_rate > 5%) → WARN
Check: Compare batch vs online predictions for same user
```

**Recovery:**
1. Identify divergence: Which features differ?
2. Align: Update online to use same feature version as batch
3. Validate: Run both on same data, verify now match

**Prevention:** Shared preprocessing library used by both batch and online. Test parity in CI/CD.

---

**Scenario 4: Hybrid Reconciliation Failure**

**What breaks:** Hybrid system: batch computes base score, online computes boost (personalization). But if boost fails, users get no prediction (instead of fallback to batch base).

**Detection:**
```
Alert: if (no_prediction_returned) → CRITICAL
Check: Did batch score exist? Did online boost fail?
```

**Recovery:**
- Return batch score alone (without online boost)
- Log that boost unavailable, but don't fail request

**Prevention:** Implement fallback: if online boost fails within 50ms, serve batch score

---

## Implementation Guidance

**Wrong:** Different preprocessing for batch vs online (produces divergence).
**Right:** Shared preprocessing function used by both.

---

## Sophisticated Interview Q&A

**Q1: 100M predictions/day?** Batch (100M GPU requests too expensive).
**Q2: Latency SLA <5min?** Online (batch=24h latency).
**Q3: Hybrid cost-benefit?** 90% batch (cheap) + 10% online (fresh) = balanced cost.
**Q4: Batch fails?** Use yesterday's cache or fallback to online.

---

## Cost & Resource Analysis

- Batch: $600/mo
- Online: $10K+/mo
- Hybrid: $400/mo

---

## Monitoring & Observability Patterns

Metrics: completion_time, latency_p99, divergence. Alerts: batch_late, online_slow, divergence>5%.

## Common Mistakes / Gotchas
- Batch: using yesterday's features for today's users (stale)
- Online: no caching, recompute same features repeatedly
- Batch storage: don't clean old predictions → disk fills up
- Online latency: no timeout → requests hang indefinitely

## Best Practices
- **Hybrid approach:** batch for recommendations, online for real-time re-ranking
- **Cache features:** in online serving, cache extracted features
- **Request batching:** accumulate requests, compute batch of 32
- **Scheduled refresh:** update batch predictions during off-peak
- **Fallback:** if online fails, serve last batch prediction

## Code Example
```python
# Batch: nightly precompute
def batch_inference_job():
    users = fetch_all_users()
    features = extract_features_batch(users)
    predictions = model.predict(features)
    database.save_predictions(zip(users, predictions))

# Online: per-request
async def online_inference(user_id):
    features = extract_features_live(user_id)
    prediction = model.predict(features)
    return prediction

# Hybrid: batch + online
async def hybrid_inference(user_id):
    batch_score = database.get_cached_prediction(user_id)
    current_features = extract_live_features(user_id)
    personalization_boost = compute_personalization(user_id, current_features)
    return batch_score + personalization_boost
```

## Interview Q&A

Q: When use batch vs online?
A: A: Batch: recommendations (pre-ranked list), batch scores (daily user churn). Online: fraud (per-transaction), real-time ranking. Consider: latency SLA, cost, data freshness.

Q: Cost comparison: 1M users, 10 features?
A: A: Batch: 1M × 10 × $0.001 = $10/day. Online: 1M × 5 req/day × 10 × $0.001 = $50/day. Batch 5x cheaper. But if online improves CTR 5%, CTR→revenue, online pays for itself.

Q: Batch every day but model updates hourly?
A: A: Old batch data. Solution: compute batch more frequently (6 hours), or hybrid (batch + online delta).

Q: 100M predictions/day. Batch or online?
A: A: Batch. Online = 100M GPU requests = expensive. Batch = load model once, 100M predictions = cheap.

Q: Latency SLA <5min. Batch or online?
A: A: Online. Batch latency = wait until next batch run (up to 24h). Online = sub-second.

Q: How do you measure batch stale­ness impact?
A: A: Compare: batch prediction (old), fresh online prediction (real-time). How much differ? If <1% CTR impact, staleness acceptable.

Q: Batch fails, missing predictions for 100K users. Fallback?
A: A: Fall back to online (slower, but works). Or use yesterday's cached batch. Have graceful degradation.

Q: Hybrid strategy: batch + online?
A: A: Batch: base recommendations (quick, cheap). Online: personalization (add user-specific signal). Combine: batch results + online boost = fresh + fast.
## Interview Quick-Reference
| Use Case | Best Approach |
|----------|---------------|
| Recommendations | Batch (pre-ranked) or Hybrid |
| Fraud | Online (per-transaction) |
| Personalization | Online (real-time) |
| Reports | Batch (daily snapshot) |

## Related Topics
- [Model Serving](05-model-serving.md)
- [Request Batching](09-request-batching.md)

## Resources
- [Batch vs Online Serving](https://valohai.com/batch-vs-online-machine-learning-inference/)

