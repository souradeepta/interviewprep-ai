# Load Balancing

## Detailed Description

Distribute incoming requests across multiple replicas. Prevents single server overload. Strategies: round-robin, least-connections, weighted. Ensures: no bottleneck, high availability, horizontal scaling.

## Core Intuition

LB = traffic cop. All requests hit one load balancer. LB routes: 'req1→replica1, req2→replica2, req3→replica3'. Without LB: all traffic hits one server = bottleneck. With LB: distributed = 3x capacity.

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

## Detailed Trade-off Analysis

| Algorithm | Distribution | Overhead | Sticky Sessions | Heterogeneous |
|-----------|--------------|----------|-----------------|---------------|
| Round-robin | Even | Low | No | Poor |
| Least-conn | Balanced | Medium | No | Good |
| Weighted | Custom | Low | No | Good |
| Hash-based | Consistent | Medium | Yes | Fair |

**Decision:** Heterogeneous servers→weighted/least-conn. Homogeneous→round-robin. Need sessions→hash-based.

---

## Production Failure Scenarios

**Scenario 1: Uneven distribution**
- Round-robin on [fast, fast, slow]. Slow server overloaded.
- Fix: Weighted LB or least-conn.

**Scenario 2: Unhealthy replica still gets traffic**
- Health check fails but LB still routes traffic. Cascading failures.
- Fix: Remove unhealthy from rotation. Circuit breaker.

**Scenario 3: Sticky sessions break**
- Client session pinned to replica A. Replica A dies. Session lost.
- Fix: Replicate sessions (Redis) or stateless design.

---

## Implementation Guidance

**Wrong:** Round-robin on heterogeneous hardware.
**Right:** Weighted or least-conn based on actual capacity.

---

## Sophisticated Interview Q&A

**Q1: Mix of fast and slow servers. How distribute?**
A: Weighted LB. Fast=1.0, Slow=0.5. Or empirical least-conn.

**Q2: Sticky sessions. Replica dies. Recover?**
A: Replicate sessions (Redis) or accept re-login.

**Q3: LB becomes bottleneck itself?**
A: Use DNS for geo-distribution or multi-tier LB.

---

## Cost & Resource Analysis

LB overhead ~5-10%. Benefit: distributed capacity >> cost.

---

## Monitoring & Observability

Metrics: requests_per_backend, latency_per_backend, error_rate_per_backend. Alerts: distribution skew>20%, backend latency>SLA.

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

Q: Round-robin vs least-connections?
A: A: Round-robin: cycle through replicas (simple). Least-connections: route to busiest replica with fewest active (handles variable latency). Least-connections better when requests have different durations.

Q: A replica becomes slow (p99 latency spikes)?
A: A: Health check detects (periodic test requests). If p99>threshold, mark degraded, route less traffic. Remove if latency>2x normal. Wait 30s before re-adding.

Q: Sticky sessions: keep user on same replica?
A: A: When: stateful (user context in-memory). Risk: uneven load if users cluster. Solution: distributed cache (Redis) instead of in-memory (enables easy LB).

Q: One replica fails: requests drop?
A: A: Health check detects down. Remove from rotation. Traffic auto-routes to healthy replicas. On recovery, add back. Deployment: rolling update (take down 1, update, bring back, repeat).

Q: 10 replicas, 1000 req/sec. Distribution even?
A: A: With round-robin, yes (each gets ~100 req/sec). With least-connections, yes (requests auto-balanced by queue depth). Monitor: if one replica overloaded, investigate (might have bug).

Q: Gradual rollout: 10% new model, 90% old?
A: A: Weighted routing: send 10% of traffic to replica with new model, 90% to old. Monitor: predictions differ >5%? Investigate. Increase % gradually (25%, 50%, 100%).

Q: LB itself a bottleneck?
A: A: Use HA LB: 2 LBs, clients route to nearest (DNS-based). Or distributed LB (all replicas handle LB themselves, gossip state). Or cloud-native (managed LB, auto-scales).

Q: New replica added. How does LB know?
A: A: Service discovery: replicas register with LB (Consul, Kubernetes). LB queries: what replicas exist? Auto-includes new ones.
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

