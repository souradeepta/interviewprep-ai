# Semantic Caching

## TL;DR
Cache LLM responses by semantic similarity, not exact key match. Query similar to cached query? Return cached response. Reduces API costs 5-10x.

## Core Intuition
Users ask similar things: "What is the capital of France?" and "What's the capital city of France?" are the same. Cache both under one response.

## How It Works
```
Query: "What is Paris known for?"
  ↓ [embed]
  ↓ [search cached embeddings]
  → Hit: "information about Paris cached" 
    → return cached response
  → Miss: Call LLM, cache response
```

**Cost savings:**
- Cache hit rate: 30-50% typical
- Cost reduction: 3-10x

## Trade-offs
- Cost reduction: major
- Latency: slightly longer (similarity search)
- Staleness: cached responses may be outdated

## Interview Quick-Reference
**Semantic caching?** Cache by embedding similarity, not exact key. Reduce cost 5-10x.

## Related Topics
- [Embeddings](embeddings.md)
- [RAG](rag.md)
- [Inference Caching](../system-design/patterns/inference-caching.md)

## Resources
- [Redis + Semantic Caching](https://redis.com/blog/cache-semantic-queries-for-faster-inference/)
