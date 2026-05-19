# Online vs Batch Inference

## TL;DR
Two serving modes: Batch (precompute all predictions nightly, serve cached), Online (compute per-request, <100ms latency). Trade-off: batch is cheap but stale (24h old), online is fresh but expensive (per-request cost).

## Core Intuition
Batch = precompute all users' scores tonight, serve instantly tomorrow (cheap). Online = compute per-user on-demand (fresh, personalizable, expensive).

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
**Q: E-commerce: recommendations. Batch or online?**
A: Start batch (cheap, good enough). If CTR improves with personalization (A/B test), switch to online or hybrid. Hybrid: batch top-100 products, online re-rank by user.

**Q: Cost comparison: 1M users, 5 features, daily batch vs online?**
A: Batch: 1M × 5 features × $0.001 = $5/day. Online: 1M × 10 requests/day × 5 features × $0.001 = $50/day. Batch 10x cheaper. But if online improves revenue by 5%, it pays for itself.

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
