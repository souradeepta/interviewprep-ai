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

**Scenario 1: Cache Stampede (Thundering Herd)**

**What breaks:** Popular cache entry expires at exactly 10:00am. 1000 concurrent requests all see cache miss. All hit GPU model simultaneously = 1000x traffic spike. GPU overloads, requests timeout, users see errors.

**Why it happens:**
- Cache TTL = 1 hour, exact expiration time
- Popular queries (trending topic) generate high volume
- All requests expire at same time

**Detection:**
```
Alert: if (model_latency spike > 3x normal OR error_rate > 5%) → CRITICAL
Monitor: concurrent_requests_to_model
```

**Recovery:**
1. Detect: Logs show 1000 requests hitting model at 10:00:10
2. Increase cache timeout temporarily (1 hour → 2 hours)
3. Add load shedding: reject requests if queue > threshold
4. Manual cache refresh: precompute popular entries before expiry

**Prevention:** Probabilistic early refresh
```python
if (time_to_ttl < 5_minutes and random.rand() < 0.01):
    # 1% of requests trigger refresh before expiry
    refresh_cache_entry(key)
else:
    return cached_value  # 99% get instant cache hit
```

---

**Scenario 2: Stale Model Cached (Version Mismatch)**

**What breaks:** Model updated from v1.0 to v2.0 (better accuracy). Cache still serves old v1.0 predictions. Users get inferior predictions until cache naturally expires (could be hours/days).

**Why it happens:**
- Cache key: hash(query) — no model version
- Deploy v2.0 model, but old predictions still cached under same key
- Cache invalidation "is one of the hardest things in CS"

**Detection:**
```
Alert: if (model deployed but latency didn't change) → WARN
  (Usually new model has slight latency change due to size difference)
Check: Compare predictions before/after deploy on same query
```

**Recovery:**
1. Detect: Query same input, compare v1 vs v2 output — they differ
2. Identify: Cache is serving old version
3. Invalidate: Clear cache or wait for TTL expiry
4. Redeploy: Next prediction uses v2

**Prevention:** Include model version in cache key
```python
cache_key = f"v2.0_{hash(query)}"  # model version embedded
# When deploy v2.1, all keys are v2.0_* and never served
```

---

**Scenario 3: Cache Inconsistency (Non-Deterministic Model)**

**What breaks:** LLM with temperature=0.7 (randomness enabled) is cached. First user gets response "A", second user with same query gets response "A" (cached, not re-rolled). But LLM should generate varied responses. Both users expect diversity.

**Why it happens:**
- Cache doesn't know about randomness parameter
- Same input → cached output (ignores temperature)
- Model configuration not part of cache key

**Detection:**
```
Alert: if (users report repetitive responses with temperature > 0) → WARN
```

**Recovery:**
- Disable cache for temperature > 0 (only cache deterministic outputs)
- Include temperature in cache key: `f"temp0.7_{hash(query)}"`

**Prevention:** Only cache deterministic outputs (temperature=0 or non-probabilistic models)
```python
if temperature == 0:
    cache_result(query, response)  # OK to cache
else:
    skip_cache(response)  # Don't cache randomness
```

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

### Cost Model (1M queries/day)

**Without Cache:**
- GPU inference: 1M queries × 100ms = 100K GPU seconds = ~$50/day = $1,500/month

**With 25% Cache Hit Rate (exact-match):**
- GPU inference: 750K queries (75% miss) × 100ms = $1,125/month
- Cache lookup: 250K queries (25% hit) × 5ms on Redis = negligible
- Redis cluster: 3 nodes × $0.07/GB/hour for 10GB = ~$50/month
- **Total: $1,175/month** (saves $325/month = 22%)

**With 50% Cache Hit Rate (exact + semantic):**
- GPU inference: 500K queries × 100ms = $750/month
- Embedding lookup: 500K queries × semantic search = ~$100/month
- Cache infrastructure: $50/month
- **Total: $900/month** (saves $600/month = 40%)

### ROI Analysis

**Scenario 1: E-commerce recommendations (high repeat rate)**
- Cache hit rate: 40-50% (users re-check products)
- Cost savings: 40% × $1,500/month = $600/month
- Cache infrastructure cost: $50/month
- **ROI: ($600 - $50) / $50 = 11x return**

**Scenario 2: LLM chatbot (variable user behavior)**
- Cache hit rate: 20-30% (less repetition)
- Cost savings: 25% × $5,000/month = $1,250/month
- Cache infrastructure: $100/month
- **ROI: ($1,250 - $100) / $100 = 12.5x return**

**Scenario 3: Image generation (rarely exact match)**
- Cache hit rate: 5-10% (images unique)
- Cost savings: 7.5% × $10,000/month = $750/month
- Cache infrastructure: $200/month
- **ROI: ($750 - $200) / $200 = 2.75x return** (still worth it)

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

