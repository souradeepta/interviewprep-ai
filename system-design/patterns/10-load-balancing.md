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

### Load Balancing Algorithm Comparison

| Algorithm | Latency Distribution | Overhead (LB CPU) | Sticky Sessions | Works with Heterogeneous | Failure Resilience |
|-----------|----------------------|-------------------|-----------------|-------------------------|-------------------|
| **Round-robin** | Even (if homogeneous) | Very low (<1%) | No | Poor (overloads slow) | Fair (doesn't detect dead) |
| **Least-connections** | Balanced (adaptive) | Medium (5-10%) | No | Good (routes by load) | Good (prefers healthy) |
| **Weighted** | Custom (via weights) | Low (2-5%) | No | Excellent (manual tuning) | Fair (weights fixed) |
| **Hash-based (sticky)** | Uneven (depends on hash) | Low (1-2%) | Yes (guaranteed) | Poor (ignores capacity) | Good if backend die rare |
| **Health-aware (dynamic)** | Best (adaptive+health) | High (20-30%) | Optional | Excellent | Excellent |

### Cost Model (3 replicas, 100 QPS, $0.50/hour each)

**Round-robin (simple):**
- 3 replicas: $36/month
- LB (lightweight, shared): $5/month
- Total: $41/month

**Least-conn (adaptive):**
- 2 replicas sufficient (better distribution): $24/month
- LB (more complex): $10/month
- Total: $34/month (17% cheaper despite more complex LB)

**Weighted (tuned for heterogeneous):**
- 1 fast + 2 standard: $26/month
- LB (moderate complexity): $8/month
- Total: $34/month

**Health-aware (production-grade):**
- 4 replicas (1 failover): $48/month
- LB (complex, health checks): $15/month
- Total: $63/month (premium for reliability)

### Decision Matrix by Scenario

| Scenario | Algorithm | Reasoning | Cost vs Benefit |
|----------|-----------|-----------|-----------------|
| **Test environment (1-2 servers)** | Round-robin | Simple to set up | $5/month |
| **Homogeneous prod cluster** | Least-conn | Adaptive, no overhead | $35/month saves replicas |
| **Heterogeneous hardware** | Weighted | Explicit capacity tuning | $34/month, manual tuning |
| **Session-heavy app (shopping cart)** | Hash-based | Sticky sessions required | $30/month, session affinity |
| **Critical high-availability** | Health-aware | Fault tolerance essential | $60+/month, premium safety |

---

## Production Failure Scenarios

**Scenario 1: Uneven Load Distribution (Heterogeneous Servers)**

**What breaks:** 3-replica setup: 2 fast (4-core GPU), 1 slow (2-core GPU). Round-robin distributes equally (33% each). Slow replica gets overloaded (handles 1/3 of load on half the resources). Queue time 500ms on slow replica vs 50ms on fast. Users hit slow replica sometimes (66% chance fair, 33% get lucky with fast).

**Why it happens:**
- Assumption: all replicas are identical
- Round-robin works only for homogeneous hardware
- Setup didn't account for hardware differences

**Detection:**
```
Metric: replica_latency_distribution (p99 per replica)
Alert: if (p99_differ > 2x) → WARN (uneven load)

Check: kubectl get nodes → see 2 GPU vs 1 GPU difference
```

**Recovery:**
1. Reconfigure LB to weighted:
   - Fast replica: weight=2.0 (gets 50% traffic)
   - Slow replica: weight=1.0 (gets 25% traffic)
2. Shift traffic gradually: 5% → 25% → 50% over 10 minutes
3. Monitor: verify slow replica latency normalizes

**Prevention:**
- Inventory: document replica capacity before deployment
- Auto-weighting: LB automatically learns capacity and adjusts weights
- Health check latency: if replica p99 > 2x others, reduce weight

---

**Scenario 2: Unhealthy Replica Still Receives Traffic**

**What breaks:** Replica B experiences database connection issue. Becomes slow (500ms latency). Health check is basic (just ping TCP port 8080). Ping still works, so health check passes. LB keeps routing traffic to Replica B. Cascading failures: clients timeout, retry, hit Replica B again.

**Why it happens:**
- Health check is shallow (TCP ping, not actual request)
- No circuit breaker (doesn't remove unhealthy from rotation)
- Assumption: "if port responds, replica is healthy" is wrong

**Detection:**
```
Alert: if (replica_error_rate > 5%) → WARN
Alert: if (replica_latency > baseline * 3) → WARN

Better: deep health check
GET /health → checks DB connectivity, returns 200 only if truly healthy
```

**Recovery:**
1. Detect: error_rate on Replica B is 20% (vs 0.1% on others)
2. Remove Replica B from LB: mark as unhealthy, route 0% traffic
3. Investigate: check database connection, logs, restart if needed
4. Restore: once fixed, gradually re-add to rotation (10% → 100%)

**Prevention:**
- Deep health checks: actually query database, not just ping TCP
- Health check frequency: every 5 seconds (catch issues quickly)
- Remove threshold: if health check fails 2x in a row, remove from rotation
- Circuit breaker: after 3 failures in 1 min, stop sending traffic

---

**Scenario 3: Sticky Sessions Break (Session Loss on Replica Failure)**

**What breaks:** E-commerce session-based (shopping cart stored in Replica A memory). Client session hashed to Replica A. User adds items to cart (stored in Replica A RAM). Replica A crashes. LB detects failure, hashes client to Replica B. Replica B has no cart data. User's cart is lost.

**Why it happens:**
- Assumption: sticky sessions are durable
- No session replication (only on one replica)
- Replica failure loses session state

**Detection:**
```
Alert: if (replica_fails AND sticky_sessions_enabled) → CRITICAL
Monitor: session_loss_count (clients with broken sessions)
```

**Recovery:**
1. Immediate: Client forced to re-login, loss of 1 session
2. Short-term (if happens often):
   - Switch to stateless: session stored in Redis (shared)
   - Or: replicate session data (write to 2 replicas)

**Prevention:**
- Session replication: after session modification, write to 2 replicas
- Use external session store (Redis): survives replica failures
- Or: stateless design (session in JWT token, signed by LB)
- Acceptable loss: if single replica failure loses <0.1% sessions, acceptable

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

