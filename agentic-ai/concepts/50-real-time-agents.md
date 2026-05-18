# Real-Time Agents

## Detailed Explanation

Real-time agents operate with strict latency constraints. Requirements: decisions must be made within milliseconds-to-seconds (not minutes). Mechanisms: minimize latency through caching (pre-compute), streaming (return partial results), parallel execution (do multiple things at once), approximate computation (fast enough answer beats perfect slow answer). Challenges: can't do complex reasoning (takes time), can't fetch external data (network latency), must be prepared (precompute what possible). Use cases: trading bots (milliseconds), recommendations (seconds), autonomous vehicles (milliseconds). Trade-offs: speed vs accuracy (fast approximation vs slow exact), planning vs reaction (predict vs respond to events).

## Core Intuition

Stock trader who must decide in milliseconds. No time for research or complex analysis. Pre-computed models, cached data, reactive logic. Speed is priority.

## How It Works

Pre-compute → Cache → Stream → Approximate → Parallel:

1. **Pre-compute** — Calculate possible responses before needed
2. **Cache** — Store computed results for fast retrieval
3. **Stream** — Return partial results immediately
4. **Approximate** — Use fast approximations instead of exact
5. **Parallelize** — Execute multiple tasks simultaneously

## Architecture / Trade-offs

**Latency Budget:** Milliseconds (trading) vs seconds (recommendations)
**Accuracy:** Exact (slow) vs approximate (fast)
**Planning:** Pre-computed (fast, rigid) vs reactive (slow, flexible)

## Interview Q&A

**Q: What are the latency requirements for different types of real-time agent applications?**
A: Streaming audio (voice assistants): <200ms end-to-end (100ms for speech recognition + 50ms LLM TTFT + 50ms TTS). Interactive chat: <1s TTFT, <50ms between tokens for streaming. Trading execution: <100ms from signal to order submission. Fraud detection: <500ms before transaction approval. Monitoring alerts: <30 seconds from event to notification. Each category requires different infrastructure choices—don't design a single real-time agent for all latency requirements.

**Q: How do you handle partial information and uncertainty in real-time decision-making?**
A: In real-time scenarios, you cannot wait for complete information before acting. Design decisions for graceful degradation: have a tiered response strategy (fast rule-based response at T=50ms, enriched LLM response at T=500ms, full analysis at T=2s). Make tentative decisions reversible where possible. Explicitly quantify uncertainty in outputs. For autonomous actions (fraud blocking, trading), set conservative thresholds and accept higher false positive rates to prevent false negative consequences.

**Q: What circuit breakers and fallbacks should a real-time agent implement?**
A: Latency circuit breaker: if model response time exceeds SLA threshold, fall back to rule-based logic. Error rate circuit breaker: if error rate exceeds 5%, stop routing to the failing component. Quality circuit breaker: if model confidence below threshold, escalate to human or use conservative default. Fallback hierarchy: LLM -> smaller faster LLM -> rule-based system -> safe default action. Test fallbacks regularly—circuit breakers that are never tested often fail when needed.

**Q: How do you manage state in a real-time agent that must process continuous event streams?**
A: Use event sourcing: log all events to an immutable append-only log (Kafka), derive current state by replaying events. Maintain a sliding window of recent context (last N events or last T seconds) to avoid unbounded memory growth. For multi-turn interactions, use an external state store (Redis) keyed by session ID. Handle late-arriving events: implement a tolerance window and reprocess recent decisions when late events arrive. Design state transitions as idempotent operations to handle duplicate events.

**Q: What monitoring is required for real-time agents to ensure they're operating correctly?**
A: P50/P95/P99 latency per operation, error rate, decision distribution (are decisions clustering suspiciously?), throughput (events per second), queue depth (is the agent keeping up?), model confidence distribution (are low-confidence cases increasing?), and business impact metrics (fraud caught vs. missed, conversion rate for recommendations). Alert on: latency P99 > 2x P50 (tail latency problem), error rate >1%, decision distribution shift >2 standard deviations. Use streaming analytics (Flink, Kafka Streams) to compute these metrics in real time.

**Q: How do you test a real-time agent before production deployment?**
A: Load testing: simulate production traffic volume and verify latency SLAs are met. Chaos testing: inject failures (model timeout, database unavailability) and verify fallbacks activate correctly. Shadow mode: run the new agent alongside the current system, comparing decisions without affecting real outcomes. Canary deployment: route 1% of traffic to the new agent, monitor for anomalies before full rollout. Record and replay: capture production events and replay them against the new agent for comparison testing. All of these must be run at scale, not just for single requests.


## Best Practices

1. Profile to find bottlenecks
2. Cache aggressively
3. Use approximations
4. Parallelize everything
5. Pre-compute possibilities
6. Stream partial results
7. Monitor latency percentiles (P99)
8. Graceful degradation (reduce quality to stay fast)

## Code Examples

### Example 1: Cached Decisions

```python
from functools import lru_cache

class RealTimeAgent:
    @lru_cache(maxsize=1000)
    def get_decision(self, state):
        # Expensive computation, cached
        return self._compute(state)

    def _compute(self, state):
        # Complex logic
        return state * 2
```

### Example 2: Parallel Execution

```python
import asyncio

async def parallel_decisions(agent, states):
    tasks = [agent.decide_async(s) for s in states]
    return await asyncio.gather(*tasks)
```

### Example 3: Streaming Results

```python
def stream_results(query):
    results = []
    for item in expensive_search(query):
        results.append(item)
        yield results  # Return partial results
```

## Related Concepts

- Latency Optimization, Observability, Performance Monitoring
