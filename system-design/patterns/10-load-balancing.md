# Load Balancing

## TL;DR
Distribute requests across multiple model serving replicas. Strategies: round-robin, least-connections, health-aware routing. Ensures no replica overloads, enables horizontal scaling, provides failover.

## Core Intuition
Without LB: one server handles all traffic → bottleneck. With LB: 3 replicas → 3x capacity + fault tolerance (one fails, others still serve).

## How It Works

**Round-robin:** Request 1 → Replica A, Request 2 → Replica B, Request 3 → Replica C, Request 4 → Replica A...

**Least-connections:** Route to replica with fewest active requests.

**Health-aware:** Check replica health every 10s, remove unhealthy replicas from rotation.

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| Round-robin | Equal replicas | Simple | Ignores replica health |
| Least-conn | Variable latency | Adaptive | Overhead tracking connections |
| Health-aware | Prod (critical) | Handles failures | Extra complexity |

## Key Properties / Trade-offs
- No LB: single point of failure
- Simple LB (round-robin): can route to dead replicas
- Complex LB (health-aware): overhead but reliable

## Common Mistakes / Gotchas
- LB without health checks: routes to dead replicas
- Sticky sessions: if all users stick to Replica A, uneven load
- Single point of failure: LB itself becomes bottleneck
- No timeout: hangs on slow replica

## Best Practices
- **Health checks:** liveness (alive?), readiness (model loaded?), deep (can reach DB?)
- **Circuit breaker:** if replica fails 3 times, remove for 30s, then retry
- **Graceful degradation:** if 2/3 replicas down, slow down clients (queue) rather than fail
- **Metrics:** latency per replica, error rate per replica. Alert if latency spikes.

## Code Example
```python
import httpx, asyncio
from collections import deque

class LoadBalancer:
    def __init__(self, replica_urls):
        self.replicas = deque(replica_urls)
        self.client = httpx.AsyncClient()
    
    async def forward(self, features):
        for _ in range(len(self.replicas)):
            replica = self.replicas[0]
            self.replicas.rotate(-1)  # Round-robin
            
            try:
                response = await self.client.post(
                    f"{replica}/predict",
                    json={"features": features},
                    timeout=2.0
                )
                return response.json()
            except httpx.RequestError:
                # Try next replica
                continue
        
        raise Exception("All replicas failed")
```

## Interview Q&A
**Q: Load balancer becomes bottleneck. Solution?**
A: Multiple LBs with DNS round-robin, or use cloud load balancer (AWS ELB). Or: embed LB logic in client (client-side load balancing). Distribute decision logic to avoid central bottleneck.

**Q: Replica A is slow (p99 latency 500ms). How LB responds?**
A: Health check detects slow response latency. Reduce traffic to Replica A (route only 20% of requests). Monitor recovery. When p99 < normal + 20%, restore to 100%.

## Interview Quick-Reference
| Replica | Status | Load |
|---------|--------|------|
| A | Healthy (50ms) | 100% |
| B | Healthy (60ms) | 100% |
| C | Slow (400ms) | 20% |
| D | Dead (timeout) | 0% |

## Related Topics
- [Model Serving](05-model-serving.md)
- [Request Batching](09-request-batching.md)

## Resources
- [NGINX Load Balancing](https://nginx.org/en/docs/http/load_balancing.html)
