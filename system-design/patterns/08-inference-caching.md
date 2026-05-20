# Inference Caching

## Detailed Description

Cache model predictions from previous identical requests. If same input, return cached output without rerunning model. 20-40% hit rate saves 30-50% cost and latency. Types: exact-match (identical input), semantic (similar input).

## Core Intuition

Cache = remember previous predictions. Request 'user=123' → model runs, prediction cached. Next request 'user=123' → return from cache instantly. If 30% of requests are repeats, 30% cost savings. Also huge latency win: cache lookup (<5ms) vs model inference (100ms).

## How It Works

**Two caching strategies:**

1. **Exact-match (deterministic):**
   - Hash(prompt) → cached response
   - Works when same input appears again
   - Hit rate: 20-30% for typical LLM workloads
   - Cost savings: 20-30%

2. **Semantic (fuzzy matching):**
   - Embed query, find similar cached queries
   - Cosine similarity > 0.95 → serve cached response
   - Works for rephrased questions with same intent
   - Hit rate: 40-50% combined with exact-match
   - Cost: embedding inference (cheap) vs cache lookup gain

## Key Properties / Trade-offs

| Aspect | No Cache | Exact Match | Semantic |
|--------|----------|-------------|----------|
| Cost | Baseline | -25% | -40% |
| Latency | 1000ms | 10ms | 50ms |
| Hit rate | N/A | 25% | 50% |
| Complexity | Low | Medium | High |
| Freshness | Always fresh | Potential staleness | Potential staleness |

## Production Failure Scenarios

### Scenario 1: Cache stampede
**What breaks:** Cache expires. 100 concurrent requests see miss. All hit model simultaneously = 100x traffic spike.
**Prevent:** Probabilistic early refresh (at 90% TTL, one request recomputes, others get stale).

### Scenario 2: Stale model cached
**What breaks:** Model updated v1→v2. Cache still serves v1 responses.
**Prevent:** Include model version in cache key. Auto-invalidate on update.

### Scenario 3: Non-deterministic caching
**What breaks:** LLM with temperature=0.5 cached. Returns same response always (wrong, should be varied).
**Prevent:** Only cache temp=0 or deterministic outputs.

---

## Implementation Guidance

**Wrong:** Cache without model version (serves wrong version after update).
**Right:** Include model version in cache key (v2.0_modelid_query).

**Edge case:** Semantic similarity threshold must be high (~0.95, not 0.8 or 0.99).

---

## Sophisticated Interview Q&A

**Q1: 30% cache hit rate. Worth it?**
A: 30% saves 30% compute cost. At $1K/month, save $300/month. Cache overhead <$50 = 6:1 ROI. Yes.

**Q2: Semantic cache threshold?**
A: 0.95. Below: too different (wrong answers). Above 0.98: too strict (few hits). 0.95 sweet spot.

**Q3: Model update. Invalidate cache?**
A: Include model version in key. Update version → old cache never served.

**Q4: Cache stampede?**
A: Probabilistic early refresh. At 90% TTL, one request recomputes. Others get stale. Prevents spike.

---

## Cost & Resource Analysis

30% hit rate = 30% cost savings. ROI: 6:1 (savings >> overhead).

---

## Monitoring & Observability

Metrics: hit_rate (target 20-30%), cache_size, eviction_rate. Alerts: hit_rate dropping, stale_responses > 0.

## Common Mistakes / Gotchas
- Cache LLM responses at temp > 0 (random): different responses for same input
- Cache with no TTL: stale responses served long-term
- Cache stampede: many requests miss simultaneously → all compute at once (traffic spike)
- No cache key versioning: model v1 and v2 share cache → wrong version served

## Best Practices

- **Hash-based keys for exact match:** hash(model_id + prompt + temperature=0)
- **TTL based on freshness needs:** FAQ (24h), user profiles (1h), real-time (no cache)
- **Semantic cache with confidence:** only serve if similarity >0.95, log lower-confidence hits
- **Cache warming:** pre-populate common queries (FAQ) before peak hours
- **Monitoring:** track hit rate, cost savings, cache size. Alert if hit rate drops.

## Code Example

```python
import hashlib, json
from redis import Redis

class InferenceCache:
    def __init__(self, llm_client, redis_client, ttl_seconds=3600):
        self.llm = llm_client
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    def predict(self, prompt, model="gpt-4"):
        # Create cache key from prompt hash
        key = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()
        
        # Check cache
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        # Miss: compute
        response = self.llm.create_completion(prompt, model=model)
        
        # Store in cache with TTL
        self.redis.setex(key, self.ttl, json.dumps(response))
        return response
```

## Interview Q&A

Q: Exact-match caching overhead?
A: A: Cache key = hash(input). Lookup hash table (<1ms). If miss, compute. If hit, return. Overhead negligible. ROI: any hit rate >5% saves money.

Q: Cache invalidation: when do you evict?
A: A: TTL (time-to-live): cache for 1 hour, then discard (fresh data). Or: event-based (model retrains, invalidate all). Or LRU (least recently used, budget fixed cache size).

Q: Semantic caching: similar inputs → same output?
A: A: Hash features + embedding distance. If 'user=123' cached and 'user=124' similar (same cohort), return cached prediction. Reduces cache misses, but risk of wrong prediction (users are different).

Q: Cache cold start: new user, no history?
A: A: Cache misses for new users. Expected. Cache improves over time as repeat users accrue.

Q: Cache consistency: model updated, old predictions stale?
A: A: Invalidate cache on model update. Or tag predictions with model version (model v1.2.0's prediction is different from v1.2.1, don't confuse).

Q: Cache poisoning: what if incorrect prediction is cached?
A: A: Validation: before caching, check prediction passes sanity check (within expected range). Log cache hits vs misses (detect poison). Monitor predictions quality.

Q: Distributed cache: users on different servers?
A: A: Use Redis/Memcached (shared). All servers query same cache. Eliminates redundant computation across servers.

Q: Cache 10M predictions. Storage?
A: A: Redis: 10M × (10 bytes input + 10 bytes output + overhead) = ~200MB (fits in memory). Cost negligible.
## Interview Quick-Reference

| Strategy | Hit Rate | Cost Savings | Complexity |
|----------|----------|--------------|------------|
| Exact match | 25% | 25% | Low |
| Semantic | 40% | 40% | High |
| Combined | 45% | 45% | High |

## Related Topics
- [Model Serving](05-model-serving.md) - serves from cache
- [LLM API Gateway](03-llm-api-gateway.md) - caches responses

## Resources
- [Redis Caching Best Practices](https://redis.io/docs/management/eviction/)
- [LLM Prompt Caching](https://openai.com/blog/prompt-caching/)

