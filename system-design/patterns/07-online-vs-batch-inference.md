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

**Decision criteria:**
- **Batch:** <1h latency acceptable, cost-sensitive, volume >100M/day
- **Online:** <100ms critical, personalization needed, lower volume
- **Hybrid:** 90% batch (cheap) + 10% online (fresh)

**Cost model (1M predictions/day):**
- Batch: $600/mo
- Online: $1K+/mo
- Hybrid: $400/mo

---

## Production Failure Scenarios

Scenario 1: Batch late → use yesterday's cache. Scenario 2: Online slow → quantize model. Scenario 3: Divergence → use shared preprocessing.

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

