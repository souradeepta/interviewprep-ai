# Inference Caching

## TL;DR
Cache LLM/model responses to eliminate redundant inference. Strategies: exact-match (hash prompt), semantic (embedding similarity). Hit rate 20-40% saves 20-40% cost + latency.

## Core Intuition
Same question asked twice? Don't run inference twice. First time: compute and cache. Second time: serve from cache. Simple but powerful.

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

**Q: Cache hit rate is 15%. Worth it?**
A: No. Overhead of cache management (lookup, storage, invalidation) outweighs 15% savings. Typical breakeven: 20-25% hit rate. Focus on raising hit rate first: (1) identify top 20% of queries, (2) pre-populate cache, (3) add semantic caching for similar queries. Retarget once hit rate > 25%.

**Q: How do you handle non-deterministic models (temperature > 0)?**
A: Don't cache full responses. Cache components: (1) embeddings (deterministic), (2) top-k candidates (deterministic), (3) final ranking (deterministic). Personalization happens post-cache. Example: cache top-10 products, user-specific re-ranking happens online.

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
