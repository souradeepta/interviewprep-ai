"""
Auto-generated from 32-tracing-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Distributed Tracing for Agents
# Learning objectives:
# - Understand how distributed tracing tracks requests through multiple services
# - Implement trace ID propagation across agent boundaries
# ======================================================================

# ======================================================================
# ## Setup
# ======================================================================

import uuid
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

print("✓ Setup complete: tracing libraries ready")


# ======================================================================
# ## Level 1: Basic Implementation
# Core concept: Trace ID propagation—follow a request through multiple services
# ======================================================================

class SimpleTracer:
    """Basic distributed tracer"""
    
    def __init__(self):
        self.traces = {}  # {trace_id: [spans]}
    
    def create_trace(self) -> str:
        """Start new trace, return trace ID"""
        trace_id = str(uuid.uuid4())[:8]
        self.traces[trace_id] = []
        return trace_id
    
    def create_span(self, trace_id: str, operation: str, 
                    parent_id: Optional[str] = None) -> str:
        """Create span in trace"""
        span_id = str(uuid.uuid4())[:8]
        span = {
            "span_id": span_id,
            "operation": operation,
            "parent_id": parent_id,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "duration_ms": None
        }
        self.traces[trace_id].append(span)
        return span_id
    
    def end_span(self, trace_id: str, span_id: str) -> None:
        """End span"""
        for span in self.traces[trace_id]:
            if span["span_id"] == span_id:
                span["end_time"] = datetime.utcnow().isoformat()
    
    def get_trace(self, trace_id: str) -> Dict:
        """Get trace data"""
        return {"trace_id": trace_id, "spans": self.traces[trace_id]}

# Usage: trace a request
tracer = SimpleTracer()
trace_id = tracer.create_trace()
print(f"Trace ID: {trace_id}")

# Agent A starts
span_a = tracer.create_span(trace_id, "Agent-A")
time.sleep(0.02)

# Agent A calls Tool
span_tool = tracer.create_span(trace_id, "Tool-Search", parent_id=span_a)
time.sleep(0.05)
tracer.end_span(trace_id, span_tool)

# Agent A completes
tracer.end_span(trace_id, span_a)

# Show trace
trace = tracer.get_trace(trace_id)
print(f"\nTrace has {len(trace['spans'])} spans:")
for span in trace['spans']:
    print(f"  {span['operation']} (parent: {span['parent_id']})")  


# ======================================================================
# ## Level 2: Advanced Implementation
# Full tracing with timing analysis and waterfall visualization
# ======================================================================

@dataclass
class Span:
    span_id: str
    operation: str
    service: str
    parent_id: Optional[str]
    start_ms: float
    end_ms: float
    attributes: Dict[str, Any]
    
    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms if self.end_ms > 0 else 0

class FullTracer:
    """Full-featured tracer with timing and attributes"""
    
    def __init__(self):
        self.traces = {}  # {trace_id: [Span]}
        self.start_time = time.time() * 1000  # ms
    
    def _elapsed_ms(self) -> float:
        return (time.time() * 1000) - self.start_time
    
    def create_trace(self) -> str:
        trace_id = str(uuid.uuid4())[:8]
        self.traces[trace_id] = []
        return trace_id
    
    def start_span(self, trace_id: str, operation: str, service: str,
                   parent_id: Optional[str] = None) -> str:
        span_id = str(uuid.uuid4())[:8]
        span = Span(
            span_id=span_id,
            operation=operation,
            service=service,
            parent_id=parent_id,
            start_ms=self._elapsed_ms(),
            end_ms=0,
            attributes={}
        )
        self.traces[trace_id].append(span)
        return span_id
    
    def end_span(self, trace_id: str, span_id: str) -> None:
        for span in self.traces[trace_id]:
            if span.span_id == span_id:
                span.end_ms = self._elapsed_ms()
    
    def add_attribute(self, trace_id: str, span_id: str, key: str, value: Any) -> None:
        for span in self.traces[trace_id]:
            if span.span_id == span_id:
                span.attributes[key] = value
    
    def get_critical_path(self, trace_id: str) -> float:
        """Get longest chain (critical path latency)"""
        spans = {s.span_id: s for s in self.traces[trace_id]}
        
        def get_path_duration(span_id: str) -> float:
            span = spans[span_id]
            children = [s for s in self.traces[trace_id] if s.parent_id == span_id]
            if not children:
                return span.duration_ms
            max_child = max(get_path_duration(c.span_id) for c in children)
            return span.duration_ms + max_child
        
        roots = [s for s in self.traces[trace_id] if s.parent_id is None]
        return max(get_path_duration(r.span_id) for r in roots) if roots else 0
    
    def visualize_waterfall(self, trace_id: str) -> str:
        """Create ASCII waterfall"""
        spans = sorted(self.traces[trace_id], key=lambda s: s.start_ms)
        min_start = min(s.start_ms for s in spans)
        max_end = max(s.end_ms for s in spans)
        total_duration = max_end - min_start
        
        lines = []
        for span in spans:
            # Calculate position and width
            start_pos = int(40 * (span.start_ms - min_start) / total_duration) if total_duration > 0 else 0
            duration_width = int(40 * span.duration_ms / total_duration) if total_duration > 0 else 1
            duration_width = max(1, duration_width)
            
            # Build line
            line = " " * start_pos + "█" * duration_width + f" {span.operation} ({span.duration_ms:.0f}ms)"
            lines.append(line)
        
        return "\n".join(lines)

# Simulate complex request
tracer = FullTracer()
trace_id = tracer.create_trace()

# Agent A
span_a = tracer.start_span(trace_id, "Agent-A", "agent-service")
time.sleep(0.02)

# Tool Search (parallel)
span_search = tracer.start_span(trace_id, "Tool-Search", "search-service", parent_id=span_a)
tracer.add_attribute(trace_id, span_search, "query", "What is AI?")
time.sleep(0.08)
tracer.end_span(trace_id, span_search)

# Agent B (also under A)
span_b = tracer.start_span(trace_id, "Agent-B", "agent-service", parent_id=span_a)
time.sleep(0.05)
tracer.end_span(trace_id, span_b)

tracer.end_span(trace_id, span_a)

print("Trace Waterfall:")
print(tracer.visualize_waterfall(trace_id))
print(f"\nCritical path: {tracer.get_critical_path(trace_id):.0f}ms")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Multi-Agent Request Tracing
# ======================================================================

class MultiAgentTracer:
    """Trace requests across multiple agents with context propagation"""
    
    def __init__(self):
        self.traces = {}
        self.start_time = time.time() * 1000
    
    def propagate_context(self, trace_id: str, parent_span_id: str) -> Dict[str, str]:
        """Create context for passing to another agent"""
        return {"trace_id": trace_id, "parent_span_id": parent_span_id}
    
    def create_trace(self) -> str:
        trace_id = str(uuid.uuid4())[:8]
        self.traces[trace_id] = []
        return trace_id
    
    def log_span(self, trace_id: str, span_id: str, operation: str, 
                 service: str, parent_id: Optional[str], 
                 duration_ms: float) -> None:
        self.traces[trace_id].append({
            "span_id": span_id,
            "operation": operation,
            "service": service,
            "parent_id": parent_id,
            "duration_ms": duration_ms
        })
    
    def print_trace_summary(self, trace_id: str) -> None:
        spans = self.traces[trace_id]
        total = sum(s["duration_ms"] for s in spans)
        
        print(f"\nTrace {trace_id} Summary:")
        print(f"Total latency: {total:.0f}ms")
        print(f"Spans: {len(spans)}")
        
        for span in sorted(spans, key=lambda s: s["duration_ms"], reverse=True):
            pct = 100 * span["duration_ms"] / total if total > 0 else 0
            print(f"  {span['operation']:20} {span['duration_ms']:6.0f}ms ({pct:5.1f}%) [{span['service']}]")

# Trace a complex multi-agent flow
tracer = MultiAgentTracer()
trace_id = tracer.create_trace()

# Client → Agent A
span_a = str(uuid.uuid4())[:8]
tracer.log_span(trace_id, span_a, "Agent-A-Process", "agent-svc", None, 150)

# Agent A → Tool Search
span_search = str(uuid.uuid4())[:8]
tracer.log_span(trace_id, span_search, "Tool-Search", "search-svc", span_a, 300)

# Agent A → Agent B (context propagated)
context = tracer.propagate_context(trace_id, span_a)
span_b = str(uuid.uuid4())[:8]
tracer.log_span(trace_id, span_b, "Agent-B-Process", "agent-svc", span_a, 200)

# Agent B → Tool API
span_api = str(uuid.uuid4())[:8]
tracer.log_span(trace_id, span_api, "Tool-API", "api-svc", span_b, 500)

tracer.print_trace_summary(trace_id)


# ======================================================================
# ### Example 2: Identifying Bottlenecks from Trace Data
# ======================================================================

class BottleneckAnalyzer:
    """Analyze traces to identify performance bottlenecks"""
    
    @staticmethod
    def find_slowest_span(spans: List[Dict]) -> Dict:
        """Find span with highest latency"""
        return max(spans, key=lambda s: s["duration_ms"])
    
    @staticmethod
    def find_slowest_service(spans: List[Dict]) -> str:
        """Find service with highest total time"""
        service_times = {}
        for span in spans:
            svc = span.get("service", "unknown")
            service_times[svc] = service_times.get(svc, 0) + span["duration_ms"]
        return max(service_times, key=service_times.get)
    
    @staticmethod
    def get_optimization_suggestions(spans: List[Dict]) -> List[str]:
        """Suggest optimizations based on trace"""
        suggestions = []
        total = sum(s["duration_ms"] for s in spans)
        
        # Find ops taking >30% of time
        for span in spans:
            if span["duration_ms"] > 0.3 * total:
                pct = 100 * span["duration_ms"] / total
                suggestions.append(f"Optimize {span['operation']} ({pct:.0f}% of time)")
        
        return suggestions

# Analyze example trace
example_spans = [
    {"operation": "Agent-A", "service": "agent-svc", "duration_ms": 150},
    {"operation": "Tool-Search", "service": "search-svc", "duration_ms": 300},
    {"operation": "Agent-B", "service": "agent-svc", "duration_ms": 200},
    {"operation": "Tool-API", "service": "api-svc", "duration_ms": 800},
    {"operation": "Response", "service": "agent-svc", "duration_ms": 50},
]

analyzer = BottleneckAnalyzer()

print("Bottleneck Analysis:")
slowest = analyzer.find_slowest_span(example_spans)
print(f"\nSlowest operation: {slowest['operation']} ({slowest['duration_ms']:.0f}ms)")

slowest_svc = analyzer.find_slowest_service(example_spans)
print(f"Slowest service: {slowest_svc}")

suggestions = analyzer.get_optimization_suggestions(example_spans)
print(f"\nOptimization suggestions:")
for s in suggestions:
    print(f"  • {s}")


# ======================================================================
# ### Example 3: Sampling Traces for Large-Scale Systems
# ======================================================================

class SamplingTracer:
    """Trace with intelligent sampling to reduce overhead"""
    
    def __init__(self, error_sample_rate: float = 1.0,
                 slow_sample_rate: float = 0.1,
                 fast_sample_rate: float = 0.01):
        self.error_rate = error_sample_rate
        self.slow_rate = slow_sample_rate
        self.fast_rate = fast_sample_rate
        self.traces_sampled = 0
        self.traces_dropped = 0
    
    def should_sample(self, is_error: bool, latency_ms: float) -> bool:
        import random
        
        if is_error:
            return random.random() < self.error_rate
        elif latency_ms > 1000:  # slow threshold
            return random.random() < self.slow_rate
        else:
            return random.random() < self.fast_rate
    
    def process_request(self, is_error: bool, latency_ms: float) -> Optional[str]:
        """Process request and return trace ID if sampled"""
        if self.should_sample(is_error, latency_ms):
            self.traces_sampled += 1
            return str(uuid.uuid4())[:8]
        else:
            self.traces_dropped += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.traces_sampled + self.traces_dropped
        return {
            "total_requests": total,
            "traces_sampled": self.traces_sampled,
            "sampling_rate": f"{100 * self.traces_sampled / total:.1f}%" if total > 0 else "0%",
            "cost_reduction": f"{100 * self.traces_dropped / total:.0f}%" if total > 0 else "0%"
        }

# Simulate request stream with sampling
sampler = SamplingTracer(error_sample_rate=1.0, slow_sample_rate=0.1, fast_sample_rate=0.01)

for i in range(10000):
    if i % 100 == 0:
        is_error = True
        latency = 200
    elif i % 50 == 0:
        is_error = False
        latency = 1500  # slow
    else:
        is_error = False
        latency = 250
    
    sampler.process_request(is_error, latency)

print("Sampling Results (100% errors, 10% slow, 1% fast):")
stats = sampler.get_stats()
print(f"  Total requests: {stats['total_requests']:,}")
print(f"  Traces sampled: {stats['traces_sampled']:,}")
print(f"  Sampling rate: {stats['sampling_rate']}")
print(f"  Cost reduction: {stats['cost_reduction']}")


# ======================================================================
# ## Key Takeaways
# 1. **Trace ID is the Magic Thread** — Every request gets unique ID that flows through all services. Following it shows complete request path. Make trace ID mandatory in all logging/metrics.
# 2. **Context Propagation is Critical** — When Agent A calls Agent B, include trace ID and parent span ID. Without propagation, trace breaks and you lose visibility into multi-agent flows.
# ======================================================================
