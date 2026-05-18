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
