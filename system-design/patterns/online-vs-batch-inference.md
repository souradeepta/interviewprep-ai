# Online vs Batch Inference

## TL;DR
Online: serve predictions per-request, real-time, high latency, per-request cost. Batch: precompute predictions offline, low latency retrieval, high throughput, cheaper. Choose based on SLA and use case.

## Core Intuition
For each user, compute immediately (online) or precomputed this morning (batch)?

## How It Works
**Online:** User → Request → Model → Predict → Response (100ms)
**Batch:** Precompute tonight, store, user queries cache (1ms lookup)

**When to use:**
- Online: personalization changes per request, user-specific context
- Batch: recommendations, scoring (can be daily), static predictions

## Key Properties / Trade-offs
| Aspect | Online | Batch |
|--------|--------|-------|
| Latency | 100-500ms | 1-10ms |
| Freshness | Real-time | Stale (lag) |
| Cost | High (per-request) | Low |
| Scalability | Limited (sync) | High |
| Personalization | Full | Limited |

## Common Mistakes / Gotchas
- **Batch for SLA <100ms:** can't meet latency. Need online.
- **Online for 1B users:** too expensive. Batch + cache.
- **No fallback:** online service down → no predictions. Have batch fallback.

## Interview Quick-Reference
**Online vs batch?** Online for personalized, real-time. Batch for static, precomputed. Hybrid common.

## Related Topics
- [Model Serving](model-serving.md) — online serving architecture
- [Inference Caching](inference-caching.md) — cache batch results

## Resources
- [Clipper: A Low-Latency Online Prediction Serving System](https://arxiv.org/abs/1611.08613)
