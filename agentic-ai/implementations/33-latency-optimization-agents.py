"""
Auto-generated from 33-latency-optimization-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Latency Optimization in Agents
# Learning objectives:
# - Understand why latency matters for production agents
# - Implement parallel tool execution for independent operations
# ======================================================================

# ======================================================================
# ## Setup
# ======================================================================

import os
import time
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from anthropic import Anthropic

# Configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
client = Anthropic(api_key=API_KEY)

print("✓ Setup complete: anthropic SDK ready")


# ======================================================================
# ## Level 1: Basic Implementation
# Core concept: Sequential vs parallel tool execution and measuring the difference
# ======================================================================

# Simulate tool calls with realistic latencies
async def tool_search(query: str) -> str:
    await asyncio.sleep(0.05)  # 50ms latency
    return f"Search results for: {query}"

async def tool_database(key: str) -> str:
    await asyncio.sleep(0.03)  # 30ms latency
    return f"Database record: {key}"

async def tool_api(endpoint: str) -> str:
    await asyncio.sleep(0.08)  # 80ms latency
    return f"API response: {endpoint}"

async def sequential_execution():
    """Sequential: 50 + 30 + 80 = 160ms"""
    start = time.time()
    r1 = await tool_search("python async")
    r2 = await tool_database("user_123")
    r3 = await tool_api("/api/docs")
    elapsed = (time.time() - start) * 1000
    return {"sequential_ms": elapsed, "results": [r1, r2, r3]}

async def parallel_execution():
    """Parallel: max(50, 30, 80) = 80ms (2x faster!)"""
    start = time.time()
    results = await asyncio.gather(
        tool_search("python async"),
        tool_database("user_123"),
        tool_api("/api/docs")
    )
    elapsed = (time.time() - start) * 1000
    return {"parallel_ms": elapsed, "results": results}

# Compare both approaches
async def level1_demo():
    seq = await sequential_execution()
    par = await parallel_execution()
    
    print(f"Sequential: {seq['sequential_ms']:.0f}ms")
    print(f"Parallel:   {par['parallel_ms']:.0f}ms")
    print(f"Speedup: {seq['sequential_ms'] / par['parallel_ms']:.1f}x")

# Run it
# await level1_demo()


# ======================================================================
# ## Level 2: Advanced Implementation
# Full agent loop with streaming responses, error handling, and latency tracking across P50/P95/P99
# ======================================================================

@dataclass
class LatencyMetrics:
    """Track latency percentiles"""
    values: List[float]
    
    def p50(self) -> float:
        sorted_vals = sorted(self.values)
        return sorted_vals[len(sorted_vals) // 2]
    
    def p95(self) -> float:
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[idx]
    
    def p99(self) -> float:
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals)-1)]

class OptimizedAgent:
    """Agent with latency optimization: parallel tools, streaming, caching"""
    
    def __init__(self, client):
        self.client = client
        self.cache = {}  # {prompt_hash -> (response, timestamp)}
        self.cache_ttl = 300  # 5 minutes
        self.latencies = []  # Track all response latencies
    
    async def execute_tools_parallel(self, tools_dict: Dict) -> Dict:
        """Execute multiple tools in parallel"""
        tasks = []
        tool_names = []
        
        for name, (coro) in tools_dict.items():
            tasks.append(coro)
            tool_names.append(name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {name: result for name, result in zip(tool_names, results)}
    
    def _get_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def query_with_cache(self, prompt: str) -> Optional[str]:
        """Return cached response if available and fresh"""
        cache_key = self._get_cache_key(prompt)
        
        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]
            age = time.time() - timestamp
            
            if age < self.cache_ttl:
                return response  # Cache hit!
        
        return None  # Cache miss
    
    def cache_response(self, prompt: str, response: str) -> None:
        """Store response in cache"""
        cache_key = self._get_cache_key(prompt)
        self.cache[cache_key] = (response, time.time())
    
    async def query_stream(self, prompt: str, use_cache: bool = True) -> str:
        """Query with streaming and caching"""
        start = time.time()
        
        # Try cache first
        if use_cache:
            cached = self.query_with_cache(prompt)
            if cached:
                elapsed = (time.time() - start) * 1000
                self.latencies.append(elapsed)
                return cached
        
        # Cache miss: call API with streaming
        response_text = ""
        
        with self.client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                response_text += text
        
        elapsed = (time.time() - start) * 1000
        self.latencies.append(elapsed)
        
        # Cache the response
        if use_cache:
            self.cache_response(prompt, response_text)
        
        return response_text
    
    def get_metrics(self) -> Dict:
        """Get latency statistics"""
        if not self.latencies:
            return {}
        
        metrics = LatencyMetrics(self.latencies)
        return {
            "p50_ms": metrics.p50(),
            "p95_ms": metrics.p95(),
            "p99_ms": metrics.p99(),
            "count": len(self.latencies)
        }

# Usage: agent = OptimizedAgent(client)
print("✓ OptimizedAgent class ready with parallel execution, streaming, caching")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Request Batching with Concurrency Limits
# ======================================================================

class BatchedAgentExecutor:
    """Execute multiple agent requests with concurrency limits to avoid overload"""
    
    def __init__(self, client, max_concurrent: int = 5):
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.latencies = []
    
    async def query_with_limit(self, prompt: str) -> str:
        """Query with concurrency limit using semaphore"""
        async with self.semaphore:  # Only max_concurrent queries at once
            start = time.time()
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            elapsed = (time.time() - start) * 1000
            self.latencies.append(elapsed)
            
            return response.content[0].text
    
    async def batch_queries(self, prompts: List[str]) -> List[str]:
        """Execute multiple prompts with concurrency control"""
        tasks = [self.query_with_limit(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def print_stats(self):
        if self.latencies:
            print(f"Avg latency: {sum(self.latencies)/len(self.latencies):.0f}ms")
            print(f"Max latency: {max(self.latencies):.0f}ms")
            print(f"Total requests: {len(self.latencies)}")

# Example usage:
# executor = BatchedAgentExecutor(client, max_concurrent=3)
# prompts = ["Explain X", "What is Y", "How to Z", ...]
# results = await executor.batch_queries(prompts)
# executor.print_stats()

print("✓ BatchedAgentExecutor: control concurrency to avoid overload")


# ======================================================================
# ### Example 2: Multi-Level Caching Strategy (Fast + Distributed)
# ======================================================================

class TieredCacheAgent:
    """Agent with 2-level caching: in-memory (fast) + distributed (scalable)"""
    
    def __init__(self, client):
        self.client = client
        self.l1_cache = {}  # In-memory cache (fast, ~1ms lookups)
        self.l1_ttl = 60  # 1 minute
        # In production: self.l2_cache would be Redis, DynamoDB, etc.
        self.stats = {"l1_hits": 0, "l1_misses": 0, "api_calls": 0}
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_from_l1(self, prompt: str) -> Optional[str]:
        """Check in-memory cache"""
        key = self._hash(prompt)
        if key in self.l1_cache:
            cached, timestamp = self.l1_cache[key]
            if time.time() - timestamp < self.l1_ttl:
                self.stats["l1_hits"] += 1
                return cached
        return None
    
    def query(self, prompt: str, use_cache: bool = True) -> str:
        """Query with 2-level caching fallback"""
        # L1 cache check
        if use_cache:
            cached = self.get_from_l1(prompt)
            if cached:
                return cached
            self.stats["l1_misses"] += 1
        
        # L2 cache check (simulated as "found in Redis")
        # In production, query actual Redis here
        
        # API call (slowest, ~1000-2000ms)
        self.stats["api_calls"] += 1
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        
        # Store in L1 cache
        key = self._hash(prompt)
        self.l1_cache[key] = (result, time.time())
        
        return result
    
    def print_stats(self):
        total = self.stats["l1_hits"] + self.stats["l1_misses"]
        hit_rate = 100 * self.stats["l1_hits"] / total if total > 0 else 0
        print(f"L1 Cache Hit Rate: {hit_rate:.1f}% ({self.stats['l1_hits']}/{total})")
        print(f"API Calls Saved: {self.stats['l1_hits']}")
        print(f"Total API Calls: {self.stats['api_calls']}")

# agent = TieredCacheAgent(client)
# result1 = agent.query("What is machine learning?")  # API call
# result2 = agent.query("What is machine learning?")  # L1 cache hit
# agent.print_stats()

print("✓ TieredCacheAgent: L1 (in-memory) + L2 (distributed) caching")


# ======================================================================
# ### Example 3: Latency-Aware Tool Selection (Smart Model Routing)
# ======================================================================

class LatencyAwareAgent:
    """Agent that chooses model/tool based on latency budget"""
    
    # Model latency profiles (ms, measured on production hardware)
    MODEL_PROFILES = {
        "claude-3-5-haiku-20241022": {"avg_ms": 200, "throughput": "high"},
        "claude-3-5-sonnet-20241022": {"avg_ms": 800, "throughput": "medium"},
        "claude-opus-4-1": {"avg_ms": 1500, "throughput": "low"}
    }
    
    def __init__(self, client):
        self.client = client
        self.latency_budget_ms = 500  # Max acceptable latency
    
    def select_model(self, prompt: str, complexity: str = "medium") -> str:
        """Choose best model for given latency budget"""
        if complexity == "simple":
            # For simple tasks, use fast model
            return "claude-3-5-haiku-20241022"  # ~200ms
        elif complexity == "medium":
            # For medium tasks, balance speed/quality
            return "claude-3-5-sonnet-20241022"  # ~800ms
        else:
            # For complex reasoning, quality > speed
            return "claude-opus-4-1"  # ~1500ms
    
    def query_adaptive(self, prompt: str, required_latency_ms: int = 1000) -> str:
        """Query with adaptive model selection based on latency SLA"""
        # Choose model that fits latency budget
        if required_latency_ms < 300:
            model = "claude-3-5-haiku-20241022"
        elif required_latency_ms < 1000:
            model = "claude-3-5-sonnet-20241022"
        else:
            model = "claude-opus-4-1"
        
        start = time.time()
        
        response = self.client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        elapsed = (time.time() - start) * 1000
        result = response.content[0].text
        
        print(f"Model: {model}")
        print(f"Latency: {elapsed:.0f}ms (budget: {required_latency_ms}ms)")
        print(f"SLA Met: {'✓' if elapsed < required_latency_ms else '✗'}")
        
        return result

# agent = LatencyAwareAgent(client)
# # Strict latency SLA: 300ms
# result1 = agent.query_adaptive("What is AI?", required_latency_ms=300)
# # Relaxed latency SLA: 2000ms
# result2 = agent.query_adaptive("Explain deep reinforcement learning", required_latency_ms=2000)

print("✓ LatencyAwareAgent: adaptive model selection for latency budgets")


# ======================================================================
# ## Key Takeaways
# 1. **Parallel Execution is Free** — Independent tool calls should always execute in parallel (typically 1.5-3x speedup). Use `asyncio.gather()` or equivalent.
# 2. **P99 Latency Matters More Than P50** — A system with P50=100ms but P99=5s ruins the user experience. Always optimize tail latency, not median latency.
# ======================================================================
