# Inference caching

## TL;DR
Core ML system design pattern for production.

## Core Intuition
[Intuitive explanation]

## How It Works
[Technical details]

## Key Properties / Trade-offs
- Property 1
- Property 2

## Common Mistakes / Gotchas
- Mistake 1
- Mistake 2

## Best Practices
- Use content-addressed caching (hash of input) for exact-match cache hits
- Implement semantic caching (embedding similarity) for near-duplicate queries
- Set TTL based on how frequently the underlying model or data changes
- Cache at the right granularity — response-level for full outputs, embedding-level for representations
- Monitor cache hit rate and latency reduction — validate caching is worth the complexity
- Implement cache warming for predictable request patterns
- Separate cache storage from model serving to scale independently

## Interview Q&A

**Q: What cache invalidation strategy should you use for model inference caching?**
A: TTL-based: set expiration based on how fast the underlying data changes—static lookup tables: 24 hours, personalized recommendations: 15 minutes, real-time fraud scores: never cache. Event-based: invalidate when the model is retrained or input data changes (use a cache version key that includes model version). Request-specific: for LLM responses, include a hash of the exact prompt as the cache key—any change in the prompt is a cache miss. Never cache responses from non-deterministic models at temperature >0 unless you explicitly want to freeze a specific response.

**Q: When does inference caching hurt more than it helps?**
A: Caching hurts when: the cache hit rate is <10% (overhead outweighs benefit), the cached data becomes stale quickly causing wrong predictions, memory pressure from the cache degrades other system performance, or the cache provides a false sense of capacity (real load spikes hit a cold cache). Measure: cache hit rate, latency reduction for hits vs. misses, stale cache rate (responses served after model retrain), and tail latency at cache misses (the worst case is what users experience when cache fails).

**Q: How do you implement caching for an LLM API to reduce cost and latency?**
A: Exact cache (hash match): store (prompt_hash to response), serve from cache for identical prompts. Works well for: FAQ answers, product descriptions, templated responses. Semantic cache (embedding similarity): embed the query, retrieve cached response if sufficiently similar exists. Works for: slightly rephrased questions with the same intent. Implement both with different thresholds: exact match first (free), semantic match second (cost of embedding). Track cache hit rate and cost savings; validate that semantic cache hits are actually equivalent answers to the original queries.

**Q: How do you handle personalized inference that can't be cached naively?**
A: Separate the personalization from the base computation. Cache the base computation (non-personalized model output), then apply a lightweight personalization layer (re-ranking, score adjustment) using cached user features. This way you cache the expensive part and keep the personalization layer fast. Alternatively, cache at a user segment level rather than individual level—users in the same segment get the same cached base results with segment-level adjustments. Segment-level caching has higher hit rates than individual-level.

**Q: What are the distributed caching considerations for multi-region inference serving?**
A: Cache locality: read from the nearest cache replica to minimize latency; write-through to all replicas for consistency. Replication lag: in a multi-region setup, a model retrain may not invalidate all regional caches simultaneously—implement version-aware cache keys (include model version in cache key). Cache stampede: when the cache expires and many requests simultaneously miss—use probabilistic early expiration or mutex-based single-flight to prevent all requests from computing simultaneously. Monitor cache hit rates per region independently.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
