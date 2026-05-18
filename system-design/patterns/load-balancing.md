# Load balancing

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
- Use health checks with warmup awareness — newly started model servers need time before accepting traffic
- Implement circuit breakers to stop routing to unhealthy backends
- Use weighted routing for gradual rollouts (canary deployments)
- Monitor P95/P99 latency per backend, not just average
- Use consistent hashing for stateful inference (session-pinned requests)
- Auto-scale backends based on queue depth, not just CPU — ML inference is memory-bound
- Test load balancer behavior during backend restarts

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
