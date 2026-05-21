"""
Auto-generated from 31-observability-for-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Observability for Agents
# Learning objectives:
# - Understand the three pillars: logs, metrics, traces
# - Implement structured JSON logging for queryable events
# ======================================================================

# ======================================================================
# ## Setup
# ======================================================================

import json
import time
import uuid
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from statistics import median

print("✓ Setup complete: logging and metrics ready")


# ======================================================================
# ## Level 1: Basic Implementation
# Core concept: Structured logging—JSON events with consistent schema
# ======================================================================

class SimpleStructuredLogger:
    """Log events as JSON for observability"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logs = []  # In production: send to Datadog/Honeycomb
    
    def log(self, event_type: str, request_id: str, **fields) -> None:
        """Log structured event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "event_type": event_type,
            "request_id": request_id,
            **fields
        }
        self.logs.append(log_entry)
        # In production: print(json.dumps(log_entry))  # stdout to log collector
    
    def query(self, event_type: str = None, request_id: str = None) -> list:
        """Query logs (in production: use Datadog query language)"""
        results = self.logs
        if event_type:
            results = [l for l in results if l["event_type"] == event_type]
        if request_id:
            results = [l for l in results if l["request_id"] == request_id]
        return results

# Usage: agent logs events
logger = SimpleStructuredLogger("agent-service")
request_id = str(uuid.uuid4())[:8]

# Agent starts processing
logger.log("request_start", request_id, user_query="What is AI?")
time.sleep(0.05)

# Agent calls a tool
logger.log("tool_call", request_id, tool="search", latency_ms=250, status="success")

# Agent finishes
logger.log("request_complete", request_id, total_latency_ms=350, status="success")

# Query logs
print(f"All logs for request {request_id}:")
for log in logger.query(request_id=request_id):
    print(f"  {log['event_type']}: {log.get('latency_ms', '')}")


# ======================================================================
# ## Level 2: Advanced Implementation
# Metrics aggregation: collect logs and compute latency percentiles, error rates, costs
# ======================================================================

class MetricsAggregator:
    """Aggregate logs into queryable metrics"""
    
    def __init__(self):
        self.tool_latencies = defaultdict(list)  # {tool_name: [latencies]}
        self.tool_errors = defaultdict(int)      # {tool_name: error_count}
        self.tool_costs = defaultdict(float)     # {tool_name: total_cost}
        self.request_count = 0
    
    def record_tool_execution(self, tool: str, latency_ms: float, 
                               status: str, cost_usd: float = 0.0) -> None:
        """Record tool execution from log"""
        self.tool_latencies[tool].append(latency_ms)
        self.tool_costs[tool] += cost_usd
        if status == "error":
            self.tool_errors[tool] += 1
        self.request_count += 1
    
    def get_percentile(self, tool: str, percentile: float) -> float:
        """Get latency percentile for tool"""
        latencies = sorted(self.tool_latencies[tool])
        if not latencies:
            return 0.0
        idx = int(len(latencies) * percentile)
        return float(latencies[min(idx, len(latencies)-1)])
    
    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Get metrics for dashboard"""
        dashboard = {
            "summary": {
                "total_requests": self.request_count,
                "total_cost_usd": sum(self.tool_costs.values())
            },
            "tools": {}
        }
        
        for tool in self.tool_latencies.keys():
            latencies = self.tool_latencies[tool]
            cost = self.tool_costs[tool]
            errors = self.tool_errors[tool]
            
            dashboard["tools"][tool] = {
                "calls": len(latencies),
                "p50_ms": round(self.get_percentile(tool, 0.50), 1),
                "p95_ms": round(self.get_percentile(tool, 0.95), 1),
                "p99_ms": round(self.get_percentile(tool, 0.99), 1),
                "error_count": errors,
                "error_rate": f"{100 * errors / len(latencies):.1f}%" if latencies else "0%",
                "total_cost_usd": round(cost, 4)
            }
        
        return dashboard

# Simulate tool executions
agg = MetricsAggregator()

# Simulate 100 search tool calls
for i in range(100):
    latency = 250 + (i % 30) * 5  # Vary latency
    status = "error" if i % 25 == 0 else "success"  # 4% error rate
    cost = 0.05
    agg.record_tool_execution("search", latency, status, cost)

# Simulate 50 api_call executions
for i in range(50):
    latency = 500 + (i % 50) * 10
    status = "error" if i % 20 == 0 else "success"
    cost = 0.10
    agg.record_tool_execution("api_call", latency, status, cost)

# Print dashboard
metrics = agg.get_metrics_dashboard()
print("=== Observability Dashboard ===")
print(f"Total requests: {metrics['summary']['total_requests']}")
print(f"Total cost: ${metrics['summary']['total_cost_usd']:.2f}\n")

print("Tool Performance:")
for tool, stats in metrics["tools"].items():
    print(f"\n{tool}:")
    print(f"  Calls: {stats['calls']}")
    print(f"  P50: {stats['p50_ms']}ms | P95: {stats['p95_ms']}ms | P99: {stats['p99_ms']}ms")
    print(f"  Errors: {stats['error_count']} ({stats['error_rate']})")
    print(f"  Cost: ${stats['total_cost_usd']}")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Distributed Tracing for Multi-Agent Requests
# ======================================================================

@dataclass
class TraceSpan:
    """Single operation in a distributed trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_ms: float
    end_ms: float
    status: str

class DistributedTracer:
    """Track request through multiple agents and services"""
    
    def __init__(self):
        self.traces = {}  # {trace_id: [spans]}
        self.start_time = time.time() * 1000  # ms
    
    def create_trace(self) -> str:
        """Start new trace"""
        trace_id = str(uuid.uuid4())[:8]
        self.traces[trace_id] = []
        return trace_id
    
    def start_span(self, trace_id: str, operation: str, service: str,
                   parent_span: Optional[str] = None) -> str:
        """Start operation span"""
        span_id = str(uuid.uuid4())[:8]
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span,
            operation_name=operation,
            service_name=service,
            start_ms=time.time() * 1000,
            end_ms=0,
            status="running"
        )
        self.traces[trace_id].append(span)
        return span_id
    
    def end_span(self, trace_id: str, span_id: str, status: str = "success") -> None:
        """End operation span"""
        for span in self.traces[trace_id]:
            if span.span_id == span_id:
                span.end_ms = time.time() * 1000
                span.status = status
                break
    
    def visualize_trace(self, trace_id: str) -> str:
        """Show trace as tree"""
        spans = self.traces[trace_id]
        
        def show_span(parent_id, indent=0):
            result = ""
            for span in spans:
                if span.parent_span_id == parent_id:
                    duration = span.end_ms - span.start_ms if span.end_ms > 0 else 0
                    indent_str = "  " * indent
                    result += f"{indent_str}├─ {span.operation_name} ({duration:.0f}ms) [{span.status}]\n"
                    result += show_span(span.span_id, indent + 1)
            return result
        
        return show_span(None)

# Simulate multi-agent request
tracer = DistributedTracer()
trace_id = tracer.create_trace()

# User request enters Agent A
span_a = tracer.start_span(trace_id, "Agent-A-Process", "agent-service")
time.sleep(0.02)

# Agent A calls search tool
span_search = tracer.start_span(trace_id, "Tool-Search", "search-service", parent_span=span_a)
time.sleep(0.05)
tracer.end_span(trace_id, span_search, "success")

# Agent A calls Agent B
span_b = tracer.start_span(trace_id, "Agent-B-Process", "agent-service", parent_span=span_a)
time.sleep(0.03)
tracer.end_span(trace_id, span_b, "success")

# Agent A completes
tracer.end_span(trace_id, span_a, "success")

print(f"Trace {trace_id}:")
print(tracer.visualize_trace(trace_id))


# ======================================================================
# ### Example 2: Sampling Strategy (Log Intelligently)
# ======================================================================

class SamplingLogger:
    """Reduce log volume with intelligent sampling"""
    
    def __init__(self, error_sample_rate: float = 1.0,
                 slow_sample_rate: float = 0.1,
                 fast_sample_rate: float = 0.01,
                 slow_threshold_ms: float = 1000):
        self.error_rate = error_sample_rate  # Always log errors
        self.slow_rate = slow_sample_rate    # 10% of slow requests
        self.fast_rate = fast_sample_rate    # 1% of fast requests
        self.slow_threshold = slow_threshold_ms
        self.stats = {"logged": 0, "skipped": 0, "total": 0}
    
    def should_log(self, latency_ms: float, is_error: bool) -> bool:
        """Decide whether to log this request"""
        import random
        
        if is_error:
            return random.random() < self.error_rate  # Always log errors
        elif latency_ms > self.slow_threshold:
            return random.random() < self.slow_rate   # 10% of slow
        else:
            return random.random() < self.fast_rate   # 1% of fast
    
    def log_request(self, latency_ms: float, is_error: bool) -> None:
        """Conditionally log request"""
        self.stats["total"] += 1
        
        if self.should_log(latency_ms, is_error):
            self.stats["logged"] += 1
            # In production: send to log service
        else:
            self.stats["skipped"] += 1
    
    def print_stats(self):
        total = self.stats["total"]
        logged = self.stats["logged"]
        rate = 100 * logged / total if total > 0 else 0
        print(f"Logged {logged}/{total} requests ({rate:.1f}%)")
        print(f"Savings: {self.stats['skipped']} logs not written")

# Simulate requests with sampling
logger = SamplingLogger(slow_threshold_ms=500)

# Simulate 1000 requests
for i in range(1000):
    if i % 100 == 0:
        latency = 800  # Slow
        is_error = False
    elif i % 500 == 0:
        latency = 250  # Fast
        is_error = True
    else:
        latency = 200 + (i % 100) * 2  # Normal
        is_error = False
    
    logger.log_request(latency, is_error)

print("Sampling Results (100% errors, 10% slow, 1% fast):")
logger.print_stats()


# ======================================================================
# ### Example 3: Alerting on SLOs (Service Level Objectives)
# ======================================================================

class SLOMonitor:
    """Alert when SLOs are violated"""
    
    def __init__(self, p99_latency_slo_ms: float = 2000,
                 error_rate_slo: float = 0.05,
                 cost_per_request_slo: float = 0.50):
        self.p99_slo = p99_latency_slo_ms
        self.error_slo = error_rate_slo
        self.cost_slo = cost_per_request_slo
        self.latencies = []
        self.errors = 0
        self.costs = []
        self.violations = []
    
    def record_request(self, latency_ms: float, is_error: bool, cost_usd: float) -> None:
        """Record request metrics"""
        self.latencies.append(latency_ms)
        if is_error:
            self.errors += 1
        self.costs.append(cost_usd)
    
    def check_slos(self) -> List[str]:
        """Check if any SLOs are violated"""
        violations = []
        n = len(self.latencies)
        
        if n == 0:
            return violations
        
        # Check P99 latency
        sorted_latencies = sorted(self.latencies)
        p99_idx = int(n * 0.99)
        p99 = sorted_latencies[min(p99_idx, n-1)]
        if p99 > self.p99_slo:
            violations.append(f"🔴 P99 latency {p99:.0f}ms exceeds SLO {self.p99_slo}ms")
        
        # Check error rate
        error_rate = self.errors / n
        if error_rate > self.error_slo:
            violations.append(f"🔴 Error rate {error_rate*100:.1f}% exceeds SLO {self.error_slo*100:.1f}%")
        
        # Check cost per request
        avg_cost = sum(self.costs) / len(self.costs) if self.costs else 0
        if avg_cost > self.cost_slo:
            violations.append(f"🔴 Avg cost ${avg_cost:.4f} exceeds SLO ${self.cost_slo:.2f}")
        
        return violations
    
    def print_status(self):
        """Print SLO status"""
        n = len(self.latencies)
        if n == 0:
            print("No requests yet")
            return
        
        sorted_latencies = sorted(self.latencies)
        p99_idx = int(n * 0.99)
        p99 = sorted_latencies[min(p99_idx, n-1)]
        
        error_rate = self.errors / n
        avg_cost = sum(self.costs) / len(self.costs)
        
        print(f"\nSLO Status:")
        print(f"  P99 Latency: {p99:.0f}ms (SLO: {self.p99_slo}ms) {'✓' if p99 <= self.p99_slo else '✗'}")
        print(f"  Error Rate: {error_rate*100:.1f}% (SLO: {self.error_slo*100:.1f}%) {'✓' if error_rate <= self.error_slo else '✗'}")
        print(f"  Avg Cost: ${avg_cost:.4f} (SLO: ${self.cost_slo:.2f}) {'✓' if avg_cost <= self.cost_slo else '✗'}")
        
        violations = self.check_slos()
        if violations:
            print(f"\n⚠️  ALERTS:")
            for v in violations:
                print(f"  {v}")
        else:
            print(f"\n✅ All SLOs met")

# Simulate requests and check SLOs
monitor = SLOMonitor(p99_latency_slo_ms=1500, error_rate_slo=0.05, cost_per_request_slo=0.10)

# Normal scenario
print("=== Scenario 1: Healthy System ===")
for i in range(100):
    latency = 250 + (i % 50) * 2
    is_error = i % 100 == 0  # 1% error rate
    cost = 0.05
    monitor.record_request(latency, is_error, cost)

monitor.print_status()

# Degraded scenario
print("\n=== Scenario 2: Degraded System ===")
monitor2 = SLOMonitor(p99_latency_slo_ms=1500, error_rate_slo=0.05, cost_per_request_slo=0.10)

for i in range(100):
    latency = 1800 + (i % 200) * 5  # Much slower
    is_error = i % 20 == 0  # 5% error rate
    cost = 0.15  # Higher cost
    monitor2.record_request(latency, is_error, cost)

monitor2.print_status()


# ======================================================================
# ## Key Takeaways
# 1. **Structured Logging is Foundation** — JSON logs with consistent schema are queryable. Free-form text logs are useless at scale.
# 2. **Three Pillars of Observability** — Logs (detailed), Metrics (aggregated), Traces (flow). All three are needed for complete visibility.
# ======================================================================
