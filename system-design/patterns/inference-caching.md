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

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
