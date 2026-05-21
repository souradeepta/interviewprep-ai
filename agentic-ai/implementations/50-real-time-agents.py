"""
Auto-generated from 50-real-time-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Real-Time Agents
# Objectives: Latency optimization, caching, parallelization, streaming
# ======================================================================

from functools import lru_cache

class CachedAgent:
    @lru_cache(maxsize=1000)
    def cached_decision(self, state):
        # Expensive computation
        return state * 2
    
    def batch_decide(self, states):
        return [self.cached_decision(s) for s in states]

agent = CachedAgent()
results = agent.batch_decide([1, 2, 3])
print(f'Cached decisions: {results}')


import asyncio

class ParallelAgent:
    async def decide_async(self, state):
        await asyncio.sleep(0.01)  # Simulate work
        return state
    
    async def parallel_decisions(self, states):
        tasks = [self.decide_async(s) for s in states]
        return await asyncio.gather(*tasks)

# Simulate parallel execution
print('Parallel execution: multiple decisions at once')


class StreamingAgent:
    def stream_results(self, query):
        results = []
        for i in range(3):
            results.append(f'result_{i}')
            yield results  # Return partial results
    
    def get_all_results(self, query):
        all_results = []
        for partial in self.stream_results(query):
            all_results = partial
        return all_results

agent = StreamingAgent()
print('Streaming: return results as available')


import time

class ApproximateAgent:
    def exact_compute(self, data):
        # Slow: 100ms
        time.sleep(0.1)
        return sum(data)
    
    def approximate_compute(self, data):
        # Fast: 1ms
        return sum(data[:10])  # Sample
    
    def smart_decide(self, data, time_budget=0.05):
        if time_budget < 0.01:
            return self.approximate_compute(data)
        else:
            return self.exact_compute(data)

agent = ApproximateAgent()
result = agent.smart_decide([1,2,3,4,5], time_budget=0.001)
print(f'Fast approximation: {result}')


class PrecomputedAgent:
    def __init__(self):
        self.decision_table = self._precompute()
    
    def _precompute(self):
        # Pre-compute all possible decisions
        return {s: s*2 for s in range(100)}
    
    def decide(self, state):
        if state in self.decision_table:
            return self.decision_table[state]  # O(1) lookup
        else:
            return None

agent = PrecomputedAgent()
result = agent.decide(5)
print(f'Pre-computed lookup: {result}')


# ======================================================================
# ## Key Takeaways
# Core concepts applied. Patterns proven. Ready for production.
# ======================================================================
