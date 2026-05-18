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

## Interview Q&A

**Q: What load balancing strategy works best for ML model serving?**
A: Least-connections for heterogeneous models: route to the backend with fewest in-flight requests—works well when request processing times vary significantly. Round-robin for homogeneous stateless models: simple and effective when all backends have equal capacity. Latency-aware (P2C: Power of Two Choices): randomly pick 2 backends, route to the faster one—statistically approaches optimal routing with low overhead. Avoid pure round-robin for GPU serving: if one replica is saturated (slow), round-robin keeps sending it traffic, causing cascading degradation.

**Q: How do you handle sticky sessions for stateful model inference?**
A: Most ML models should be stateless (predict independently for each request). When state is needed (conversational AI, streaming generation): implement session affinity at the load balancer (route same session_id to same backend), use an external state store (Redis) that any backend can access (better—removes stickiness requirement), or use a request router that forwards state context with each request. Stateless design is strongly preferred because it simplifies scaling, failover, and deployment—use external state stores rather than in-process state.

**Q: What health check configuration prevents sending traffic to model replicas that are warming up?**
A: Configure separate liveness and readiness probes. Readiness probe: HTTP GET /health/ready with a timeout of 30s and failure threshold of 2. In the readiness endpoint, verify: model is loaded (run a synthetic inference), GPU memory is allocated, and all dependencies are reachable. Set initialDelaySeconds equal to your worst-case model loading time (e.g., 60s for a 7B model). Never route traffic to a replica that hasn't passed readiness—a partially loaded model produces unpredictable outputs.

**Q: How do you do weighted traffic routing for gradual model rollouts?**
A: Implement at the load balancer level with weighted backends: 95% traffic to stable model, 5% to new model candidate. Increment the new model's weight in stages (5% to 10% to 25% to 50% to 100%), with a hold period at each stage to validate metrics. Many load balancers (NGINX, Envoy, AWS ALB) support weighted routing natively. For A/B testing, add a request ID header to enable per-user consistency (same user always hits the same model version across requests).

**Q: What are the failure modes when a load balancer becomes a bottleneck?**
A: The load balancer itself can become a single point of failure or a throughput bottleneck. Signs: load balancer CPU usage >80%, queue depth on the load balancer increasing, latency from client to backend exceeds latency from load balancer to backend by >20ms. Mitigation: use a managed load balancer service (AWS ALB, GCP Load Balancer) that auto-scales, implement client-side load balancing for service-to-service calls (avoids the load balancer for internal traffic), and distribute traffic across multiple load balancer instances with anycast or DNS-based routing.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
