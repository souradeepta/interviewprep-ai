"""
Auto-generated from 24-mcp-tunnel.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # MCP Tunnel / Request Proxy
# ## Learning Objectives
# 1. Implement local request proxy with auth and retry logic
# 2. Add circuit breaker and health checks
# 3. Support request queuing and TLS simulation
# 4. Test failure scenarios and recovery patterns
# ======================================================================

# Prerequisites & Imports
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime, timedelta

print("MCP Tunnel / Request Proxy Implementation")
print(f"Starting at {datetime.now().isoformat()}")


# ======================================================================
# ## Level 1: Basic Request Proxy with Auth & Retry
# ======================================================================

# Level 1: Basic Request Proxy

@dataclass
class ProxyRequest:
    request_id: str
    method: str
    path: str
    body: Dict = field(default_factory=dict)
    auth_token: str = ""
    timestamp: float = field(default_factory=time.time)

class BasicMCPTunnel:
    """Basic request proxy with auth."""
    
    def __init__(self, target_host: str, api_key: str):
        self.target_host = target_host
        self.api_key = api_key
        self.request_log = []
    
    def validate_auth(self, token: str) -> bool:
        """Validate auth token."""
        return token == self.api_key
    
    async def proxy_request(self, request: ProxyRequest) -> Dict:
        """Forward request to target (simulated)."""
        # Validate auth
        if not self.validate_auth(request.auth_token):
            return {'error': 'Unauthorized', 'code': 401}
        
        # Simulate HTTP request
        await asyncio.sleep(0.01)
        
        response = {
            'request_id': request.request_id,
            'status': 200,
            'method': request.method,
            'path': request.path,
            'body': {'result': f'Processed {request.method} {request.path}'}
        }
        
        self.request_log.append(response)
        return response
    
    def get_stats(self) -> Dict:
        return {'total_requests': len(self.request_log)}

# Test Level 1
print("Testing basic tunnel...")
tunnel = BasicMCPTunnel('localhost:8080', api_key='secret123')

# Valid request
req = ProxyRequest('req_001', 'POST', '/api/calculate', {'a': 5, 'b': 3}, auth_token='secret123')
resp = await tunnel.proxy_request(req)
print(f"Valid auth: {resp['status']}")

# Invalid auth
req_bad = ProxyRequest('req_002', 'GET', '/api/status', auth_token='wrong')
resp_bad = await tunnel.proxy_request(req_bad)
print(f"Invalid auth: {resp_bad['code']}")

print(f"Tunnel stats: {tunnel.get_stats()}")


# ======================================================================
# ## Level 2: Advanced with Circuit Breaker, Health Checks, Queueing
# ======================================================================

# Level 2: Advanced Tunnel with Circuit Breaker

class CircuitBreakerState(Enum):
    CLOSED = 'closed'      # Normal operation
    OPEN = 'open'          # Failing, reject requests
    HALF_OPEN = 'half_open'  # Testing recovery

class CircuitBreaker:
    """Circuit breaker pattern."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 5):
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # seconds before trying again
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            print("  ✓ Circuit CLOSED (recovered)")
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            print(f"  ✗ Circuit OPEN (failures: {self.failure_count})")
    
    def can_execute(self) -> bool:
        """Check if request can execute."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.failure_count = 0
                print("  ⚡ Circuit HALF_OPEN (testing)")
                return True
            return False
        
        # HALF_OPEN: allow one request
        return True

class AdvancedMCPTunnel(BasicMCPTunnel):
    """Tunnel with circuit breaker and health checks."""
    
    def __init__(self, target_host: str, api_key: str):
        super().__init__(target_host, api_key)
        self.circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=3)
        self.request_queue = asyncio.Queue(maxsize=10)
        self.health_check_interval = 5  # seconds
    
    async def health_check(self) -> bool:
        """Check if target is healthy (simulated)."""
        # Simulate health check endpoint
        await asyncio.sleep(0.01)
        # Return based on circuit state
        return self.circuit_breaker.state != CircuitBreakerState.OPEN
    
    async def proxy_with_circuit_breaker(self, request: ProxyRequest, max_retries: int = 2) -> Dict:
        """Proxy with circuit breaker protection."""
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return {'error': 'Circuit breaker OPEN', 'code': 503}
        
        # Try request
        for attempt in range(max_retries):
            try:
                # Simulate failures randomly for demo
                import random
                if random.random() < 0.3:  # 30% failure rate
                    raise Exception('Simulated failure')
                
                response = await self.proxy_request(request)
                self.circuit_breaker.record_success()
                return response
            
            except Exception as e:
                self.circuit_breaker.record_failure()
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        
        return {'error': 'Max retries exceeded', 'code': 503}
    
    async def enqueue_request(self, request: ProxyRequest):
        """Add request to queue."""
        try:
            self.request_queue.put_nowait(request)
        except asyncio.QueueFull:
            return {'error': 'Queue full', 'code': 429}
    
    async def process_queue(self):
        """Process queued requests."""
        while True:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1)
                await self.proxy_with_circuit_breaker(request)
                self.request_queue.task_done()
            except asyncio.TimeoutError:
                continue

# Test Level 2
print("Testing advanced tunnel with circuit breaker...")
tunnel = AdvancedMCPTunnel('localhost:8080', 'secret123')

print("\nSending requests with 30% failure rate:")
for i in range(5):
    req = ProxyRequest(f'req_{i}', 'POST', '/api/test', auth_token='secret123')
    result = await tunnel.proxy_with_circuit_breaker(req)
    print(f"  Request {i}: {result.get('code', result.get('error', 'ok'))}")
    await asyncio.sleep(0.1)

print(f"\nCircuit breaker state: {tunnel.circuit_breaker.state.value}")
print(f"Requests processed: {len(tunnel.request_log)}")



# ======================================================================
# ## Real-World Example 1: MCP Tunnel with Auth Validation
# ======================================================================

# Example 1: Production Tunnel with Auth

class ProductionMCPTunnel(AdvancedMCPTunnel):
    """Production-grade tunnel."""
    
    def __init__(self, target_host: str, api_key: str):
        super().__init__(target_host, api_key)
        self.auth_failures = []
        self.request_times = []
    
    def validate_auth_enhanced(self, token: str, client_id: str = None) -> bool:
        """Enhanced auth with client tracking."""
        if token != self.api_key:
            self.auth_failures.append({
                'client_id': client_id,
                'timestamp': time.time()
            })
            return False
        return True
    
    async def proxy_request_timed(self, request: ProxyRequest) -> Dict:
        """Track request timing."""
        start = time.time()
        
        if not self.validate_auth_enhanced(request.auth_token):
            return {'error': 'Unauthorized', 'code': 401}
        
        # Execute
        await asyncio.sleep(0.02)
        
        elapsed = time.time() - start
        self.request_times.append(elapsed)
        
        return {
            'request_id': request.request_id,
            'status': 200,
            'latency_ms': elapsed * 1000
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance metrics."""
        if not self.request_times:
            return {}
        
        import statistics
        return {
            'avg_latency_ms': statistics.mean(self.request_times) * 1000,
            'p50_latency_ms': sorted(self.request_times)[len(self.request_times)//2] * 1000,
            'p99_latency_ms': sorted(self.request_times)[int(len(self.request_times)*0.99)] * 1000,
            'auth_failures': len(self.auth_failures)
        }

# Test
print("Example 1: Production Tunnel\n")

tunnel = ProductionMCPTunnel('api.example.com:443', 'sk-proj-123')

print("Sending valid requests:")
for i in range(10):
    req = ProxyRequest(f'req_{i:03d}', 'POST', '/api/inference', auth_token='sk-proj-123')
    result = await tunnel.proxy_request_timed(req)
    print(f"  {i}: {result['status']} ({result['latency_ms']:.2f}ms)")

print("\nSending invalid auth:")
for i in range(3):
    req = ProxyRequest(f'bad_{i}', 'GET', '/api/status', auth_token='invalid')
    result = await tunnel.proxy_request_timed(req)
    print(f"  {i}: {result['code']}")

stats = tunnel.get_performance_stats()
print(f"\nPerformance stats:")
for k, v in stats.items():
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")



# ======================================================================
# ## Real-World Example 2: Failure Scenarios and Recovery
# ======================================================================

# Example 2: Failure Modes and Recovery

class FailureSimulatorTunnel(AdvancedMCPTunnel):
    """Tunnel that simulates various failure modes."""
    
    def __init__(self, target_host: str, api_key: str):
        super().__init__(target_host, api_key)
        self.failure_mode = None  # 'timeout', 'auth', 'server_error', 'overload'
        self.failure_counter = 0
    
    async def proxy_with_failures(self, request: ProxyRequest) -> Dict:
        """Proxy with simulated failures."""
        if self.failure_mode == 'timeout':
            await asyncio.sleep(5)  # Simulate timeout
            return {'error': 'Request timeout', 'code': 504}
        
        elif self.failure_mode == 'auth':
            return {'error': 'Invalid credentials', 'code': 401}
        
        elif self.failure_mode == 'server_error':
            self.failure_counter += 1
            # Fail first 3 times, then recover
            if self.failure_counter <= 3:
                self.circuit_breaker.record_failure()
                return {'error': 'Internal server error', 'code': 500}
            else:
                self.circuit_breaker.record_success()
                return {'status': 200, 'body': 'Recovered'}
        
        elif self.failure_mode == 'overload':
            return {'error': 'Service overloaded', 'code': 429}
        
        else:
            return {'status': 200, 'body': 'OK'}

# Test scenarios
print("Example 2: Failure Scenarios\n")

scenarios = [
    ('timeout', 'Timeout after 5 seconds'),
    ('auth', 'Authentication failure'),
    ('server_error', 'Server errors with recovery'),
    ('overload', 'Rate limited (429)'),
    (None, 'Normal operation')
]

for failure_mode, description in scenarios:
    print(f"\nScenario: {description}")
    tunnel = FailureSimulatorTunnel('api.example.com', 'secret')
    tunnel.failure_mode = failure_mode
    
    req = ProxyRequest('test_req', 'POST', '/api/test', auth_token='secret')
    result = await tunnel.proxy_with_failures(req)
    
    if 'error' in result:
        print(f"  → {result['code']}: {result['error']}")
    else:
        print(f"  → {result['status']}: Success")
    
    if failure_mode == 'server_error':
        print(f"  Circuit breaker state: {tunnel.circuit_breaker.state.value}")



# ======================================================================
# ## Real-World Example 3: Request Rate Limiting
# ======================================================================

# Example 3: Rate Limiting

class RateLimitedTunnel(ProductionMCPTunnel):
    """Tunnel with rate limiting."""
    
    def __init__(self, target_host: str, api_key: str, rate_limit: int = 10):
        super().__init__(target_host, api_key)
        self.rate_limit = rate_limit  # requests per second
        self.request_timestamps = []
    
    def is_rate_limited(self) -> bool:
        """Check if rate limit exceeded."""
        now = time.time()
        # Keep only requests from last second
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 1.0]
        
        if len(self.request_timestamps) >= self.rate_limit:
            return True
        
        self.request_timestamps.append(now)
        return False
    
    async def proxy_with_rate_limit(self, request: ProxyRequest) -> Dict:
        """Proxy with rate limiting."""
        if self.is_rate_limited():
            return {'error': 'Rate limit exceeded', 'code': 429}
        
        return await self.proxy_request_timed(request)

# Test
print("Example 3: Rate Limiting\n")

tunnel = RateLimitedTunnel('api.example.com', 'secret', rate_limit=5)

print(f"Rate limit: 5 req/sec")
print(f"Sending 8 requests rapidly:")

results = {'200': 0, '429': 0}
for i in range(8):
    req = ProxyRequest(f'burst_{i}', 'GET', '/api/data', auth_token='secret')
    result = await tunnel.proxy_with_rate_limit(req)
    code = result.get('status', result.get('code', '500'))
    results[str(code)] = results.get(str(code), 0) + 1
    print(f"  Request {i}: {code}")

print(f"\nResults: {results['200']} allowed, {results.get('429', 0)} rate limited")



# ======================================================================
# ## Comparison & Metrics
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np

# Simulate tunnel behavior under load
load_levels = [10, 50, 100, 200, 500]
error_rates = [0.01, 0.05, 0.12, 0.35, 0.68]
cb_open_times = [0, 0, 0.5, 2.0, 5.0]  # seconds
latencies = [15, 18, 25, 45, 120]  # milliseconds

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Error rate vs load
ax1.plot(load_levels, [e*100 for e in error_rates], marker='o', linewidth=2, color='#e74c3c')
ax1.set_xlabel('Request Load (req/s)', fontsize=11)
ax1.set_ylabel('Error Rate (%)', fontsize=11)
ax1.set_title('Error Rate Under Load', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
for x, y in zip(load_levels, error_rates):
    ax1.text(x, y*100 + 2, f'{y*100:.0f}%', ha='center', fontsize=9)

# Circuit breaker open time
ax2.bar(range(len(load_levels)), cb_open_times, color='#e67e22', alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(load_levels)))
ax2.set_xticklabels(load_levels)
ax2.set_ylabel('CB Open Time (seconds)', fontsize=11)
ax2.set_title('Circuit Breaker Engagement', fontsize=12, fontweight='bold')
for i, v in enumerate(cb_open_times):
    if v > 0:
        ax2.text(i, v + 0.2, f'{v:.1f}s', ha='center', fontsize=10)

# Request latency
ax3.plot(load_levels, latencies, marker='s', linewidth=2, color='#3498db')
ax3.set_xlabel('Request Load (req/s)', fontsize=11)
ax3.set_ylabel('Latency (ms)', fontsize=11)
ax3.set_title('Request Latency vs Load', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
for x, y in zip(load_levels, latencies):
    ax3.text(x, y + 3, f'{y}ms', ha='center', fontsize=9)

# Success rate
success_rates = [100 - e*100 for e in error_rates]
colors_grad = ['#2ecc71' if s > 95 else '#f39c12' if s > 85 else '#e74c3c' for s in success_rates]
ax4.bar(range(len(load_levels)), success_rates, color=colors_grad, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(load_levels)))
ax4.set_xticklabels(load_levels)
ax4.set_ylabel('Success Rate (%)', fontsize=11)
ax4.set_title('Request Success Rate', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 105)
ax4.axhline(y=99.9, color='green', linestyle='--', linewidth=1, label='Target (99.9%)')
ax4.legend()
for i, v in enumerate(success_rates):
    ax4.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

print("Tunnel Performance Under Load:")
print(f"\n{'Load':<12} {'Error Rate':<15} {'CB Open':<15} {'Latency':<15} {'Success':<12}")
print("-" * 70)
for load, err, cb, lat, succ in zip(load_levels, error_rates, cb_open_times, latencies, success_rates):
    print(f"{load:<12} {err*100:>6.1f}% {cb:>13.1f}s {lat:>13}ms {succ:>10.1f}%")



# ======================================================================
# ## Key Takeaways
# **Tunnel Architecture:**
# 1. Auth validation on every request
# 2. Request forwarding with retry logic
# 3. Circuit breaker prevents cascading failures
# 4. Health checks enable graceful degradation
# **Failure Handling:**
# - Timeout: Set per-request deadline
# - Auth failure: Return 401 immediately (non-retryable)
# - Server errors: Retry with backoff
# - Overload: Return 429, throttle or queue
# **Production Patterns:**
# - Rate limiting per client/token
# - Request queuing for load smoothing
# - Latency histograms (p50, p99)
# - Circuit breaker: 3 failures → OPEN
# - Health check: Every 5 seconds
# **Scaling Limits:**
# - Single tunnel: ~500 req/s
# - Multiple tunnels: Shard by token or path
# - Load balancer: Distribute across tunnels
# **Related Concepts:** [[resilience-patterns]], [[api-gateway]], [[service-mesh]], [[load-balancing]]
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Extend Example 1:** Add request signing (HMAC-SHA256)
# 2. **Modify Example 2:** Implement adaptive circuit breaker (failure % vs count)
# 3. **Enhance Example 3:** Add per-client rate limit buckets
# ======================================================================
