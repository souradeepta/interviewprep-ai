"""
Auto-generated from 13-llm-serving-frameworks.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # LLM Serving Frameworks
# ## Learning Objectives
# 1. Understand request batching, dynamic batching, and PagedAttention concepts
# 2. Implement a mini serving engine with request handling and response buffering
# 3. Measure throughput and latency with different batching strategies
# 4. Compare serving approaches for production deployments
# ======================================================================

import numpy as np
import torch
import time
from queue import Queue, Empty
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: Basic Serving Engine with Batching
# Simple batching: collect requests until batch_size, process together, return responses.
# ======================================================================

@dataclass
class Request:
    request_id: int
    prompt: str
    arrival_time: float = 0

class SimpleBatchingServer:
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
    
    def mock_inference(self, prompts: List[str]) -> List[str]:
        # Simulate inference: time scales with batch size
        batch_sz = len(prompts)
        time.sleep(0.1 + 0.05 * batch_sz - 0.01 * batch_sz * (batch_sz - 1) / 2)
        return [f"Response to '{p[:20]}...'" for p in prompts]
    
    def serve(self, num_requests: int) -> Dict:
        requests = [Request(i, f"Prompt {i}") for i in range(num_requests)]
        latencies = []
        total_start = time.time()
        
        for batch_idx in range(0, len(requests), self.batch_size):
            batch = requests[batch_idx:batch_idx + self.batch_size]
            prompts = [r.prompt for r in batch]
            outputs = self.mock_inference(prompts)
            
            for _ in batch:
                latencies.append(time.time() * 1000 - total_start * 1000)
        
        total_time = time.time() - total_start
        
        return {
            'batch_size': self.batch_size,
            'throughput_req_per_sec': num_requests / total_time,
            'avg_latency_ms': np.mean(latencies),
            'p99_latency_ms': np.percentile(latencies, 99),
            'total_time_sec': total_time
        }

print('Testing simple batching...')
batch_results = []
for batch_size in [1, 4, 8, 16]:
    server = SimpleBatchingServer(batch_size=batch_size)
    metrics = server.serve(num_requests=32)
    batch_results.append(metrics)
    print(f'✅ Batch size {batch_size}: {metrics["throughput_req_per_sec"]:.2f} req/s, latency {metrics["avg_latency_ms"]:.2f} ms')


# ======================================================================
# ## Level 2: Advanced Dynamic Batching with PagedAttention
# Dynamic batching: wait up to max_wait_ms for more requests, PagedAttention for efficient memory use.
# ======================================================================

class DynamicBatchingServer:
    def __init__(self, batch_size: int = 8, max_wait_ms: float = 50):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.request_queue = Queue()
        self.metrics = {'batch_sizes': [], 'latencies': []}
    
    def mock_inference_paged_attention(self, batch_reqs: List[Request]) -> List[str]:
        # PagedAttention: virtual memory for KV cache
        # Allows larger batches without OOM
        batch_sz = len(batch_reqs)
        # More efficient than vanilla attention
        inference_time = 0.15 + 0.02 * batch_sz
        time.sleep(inference_time)
        return [f"PagedAttn response {i}" for i in range(batch_sz)]
    
    def get_batch_with_timeout(self, timeout_ms: float) -> List[Request]:
        batch = []
        deadline = time.time() + timeout_ms / 1000
        
        while len(batch) < self.batch_size and time.time() < deadline:
            try:
                timeout_remaining = max(0, (deadline - time.time()))
                req = self.request_queue.get(timeout=timeout_remaining)
                batch.append(req)
            except Empty:
                break
        return batch
    
    def serve_dynamic(self, num_requests: int, request_rate: float = 100) -> Dict:
        inter_arrival = 1.0 / request_rate
        request_times = np.cumsum(np.random.exponential(inter_arrival, num_requests))
        requests = [Request(i, f"Prompt {i}") for i in range(num_requests)]
        
        total_start = time.time()
        current_req_idx = 0
        latencies = []
        
        while current_req_idx < num_requests:
            current_time = time.time() - total_start
            while current_req_idx < num_requests and request_times[current_req_idx] <= current_time:
                self.request_queue.put(requests[current_req_idx])
                current_req_idx += 1
            
            batch = self.get_batch_with_timeout(self.max_wait_ms)
            if batch:
                _ = self.mock_inference_paged_attention(batch)
                self.metrics['batch_sizes'].append(len(batch))
                for _ in batch:
                    latencies.append((time.time() - total_start) * 1000)
            
            time.sleep(0.01)
        
        # Drain remaining
        while not self.request_queue.empty():
            try:
                batch = []
                while len(batch) < self.batch_size:
                    batch.append(self.request_queue.get(timeout=0.01))
                if batch:
                    _ = self.mock_inference_paged_attention(batch)
                    self.metrics['batch_sizes'].append(len(batch))
            except Empty:
                break
        
        total_time = time.time() - total_start
        
        return {
            'max_wait_ms': self.max_wait_ms,
            'throughput_req_per_sec': num_requests / total_time,
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0,
            'avg_batch_size': np.mean(self.metrics['batch_sizes']) if self.metrics['batch_sizes'] else 0,
            'total_time_sec': total_time
        }

print('\nTesting dynamic batching...')
dynamic_results = []
for max_wait in [10, 50, 100]:
    server = DynamicBatchingServer(batch_size=8, max_wait_ms=max_wait)
    metrics = server.serve_dynamic(num_requests=32, request_rate=50)
    dynamic_results.append(metrics)
    print(f'✅ Max wait {max_wait}ms: {metrics["throughput_req_per_sec"]:.2f} req/s, latency {metrics["avg_latency_ms"]:.2f} ms')


# ======================================================================
# ## Real-World Example 1: Batch Server with GPT-2
# ======================================================================

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    class GPT2BatchServer:
        def __init__(self, batch_size: int = 4):
            self.batch_size = batch_size
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
            self.model.eval()
        
        def batch_generate(self, prompts: List[str]) -> List[str]:
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids, max_length=50, attention_mask=attention_mask,
                    temperature=0.7, top_p=0.9, do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        
        def serve(self, prompts: List[str]) -> Dict:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            outputs = self.batch_generate(prompts)
            elapsed = (time.time() - start) * 1000
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            
            return {
                'throughput': len(prompts) / (elapsed / 1000),
                'latency_ms_per_prompt': elapsed / len(prompts),
                'peak_memory_mb': peak_memory,
                'outputs': outputs
            }
    
    print('\nTesting GPT-2 batch server...')
    prompts = ["The future of AI is", "Machine learning enables", "Neural networks learn"]
    server = GPT2BatchServer(batch_size=4)
    result = server.serve(prompts)
    
    print(f'✅ GPT-2 batch server:')
    print(f'  Throughput: {result["throughput"]:.2f} prompts/sec')
    print(f'  Latency: {result["latency_ms_per_prompt"]:.2f} ms/prompt')
    print(f'  Peak memory: {result["peak_memory_mb"]:.1f} MB')
    gpt2_result = result

except Exception as e:
    print(f'Note: GPT-2 requires transformers: {e}')
    gpt2_result = None


# ======================================================================
# ## Real-World Example 2: Throughput vs Batch Size
# ======================================================================

print('\nComparing batch sizes...')
batch_comparisons = []

for batch_size in [1, 2, 4, 8, 16]:
    if batch_size == 1:
        total_time = 0
        for i in range(32):
            time.sleep(0.05)
            total_time += 0.05
    else:
        server = SimpleBatchingServer(batch_size=batch_size)
        result = server.serve(num_requests=32)
        total_time = result['total_time_sec']
    
    throughput = 32 / total_time
    batch_comparisons.append({'batch_size': batch_size, 'throughput': throughput})
    print(f'✅ Batch {batch_size}: {throughput:.2f} req/s')


# ======================================================================
# ## Real-World Example 3: Continuous Batching (vLLM-style)
# ======================================================================

class ContinuousBatchingServer:
    """Continuous batching: process requests at different decode steps in parallel"""
    def __init__(self, max_batch_size: int = 8, target_tokens: int = 50):
        self.max_batch_size = max_batch_size
        self.target_tokens = target_tokens
        self.active = {}
        self.completed = []
    
    def step(self, incoming: List[Request]) -> Tuple[int, float]:
        # Add new requests
        slots = self.max_batch_size - len(self.active)
        for req in incoming[:slots]:
            self.active[req.request_id] = (0, time.time())
        
        batch_sz = len(self.active)
        if batch_sz > 0:
            step_time = 0.005 + 0.002 * batch_sz
            time.sleep(step_time)
            
            completed = []
            for req_id in list(self.active.keys()):
                tokens, arrival = self.active[req_id]
                tokens += 1
                if tokens >= self.target_tokens:
                    latency = (time.time() - arrival) * 1000
                    self.completed.append({'request_id': req_id, 'latency_ms': latency})
                    completed.append(req_id)
                else:
                    self.active[req_id] = (tokens, arrival)
            
            for req_id in completed:
                del self.active[req_id]
        else:
            step_time = 0
        
        return batch_sz, step_time
    
    def serve(self, num_requests: int, request_rate: float = 50) -> Dict:
        inter_arrival = 1.0 / request_rate
        next_arrival = inter_arrival
        req_idx = 0
        time_elapsed = 0
        batch_sizes = []
        
        while req_idx < num_requests or self.active:
            incoming = []
            while req_idx < num_requests and time_elapsed >= next_arrival:
                incoming.append(Request(req_idx, f"Prompt {req_idx}"))
                req_idx += 1
                next_arrival += inter_arrival
            
            batch_sz, step_time = self.step(incoming)
            batch_sizes.append(batch_sz)
            time_elapsed += step_time
        
        latencies = [c['latency_ms'] for c in self.completed]
        
        return {
            'throughput': num_requests / time_elapsed if time_elapsed > 0 else 0,
            'avg_latency_ms': np.mean(latencies),
            'p99_latency_ms': np.percentile(latencies, 99),
            'avg_batch_size': np.mean(batch_sizes),
            'total_time_sec': time_elapsed
        }

print('\nTesting continuous batching...')
cb_server = ContinuousBatchingServer(max_batch_size=8)
cb_result = cb_server.serve(num_requests=32, request_rate=50)
print(f'✅ Continuous batching: {cb_result["throughput"]:.2f} req/s, latency {cb_result["avg_latency_ms"]:.2f} ms')


# ======================================================================
# ## Comparison: Serving Strategies
# ======================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Simple batching throughput
batch_sizes_simple = [r['batch_size'] for r in batch_results]
throughputs = [r['throughput_req_per_sec'] for r in batch_results]
axes[0, 0].plot(batch_sizes_simple, throughputs, 'o-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Batch Size')
axes[0, 0].set_ylabel('Throughput (req/s)')
axes[0, 0].set_title('Simple Batching: Throughput vs Batch Size')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Latency vs batch size
latencies_simple = [r['avg_latency_ms'] for r in batch_results]
axes[0, 1].plot(batch_sizes_simple, latencies_simple, 's-', linewidth=2, markersize=8, color='#ff7f0e')
axes[0, 1].set_xlabel('Batch Size')
axes[0, 1].set_ylabel('Avg Latency (ms)')
axes[0, 1].set_title('Simple Batching: Latency vs Batch Size')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Dynamic batching trade-off
if dynamic_results:
    max_waits = [r['max_wait_ms'] for r in dynamic_results]
    dyn_throughputs = [r['throughput_req_per_sec'] for r in dynamic_results]
    ax3 = axes[1, 0]
    ax3.plot(max_waits, dyn_throughputs, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Max Wait Time (ms)')
    ax3.set_ylabel('Throughput (req/s)')
    ax3.set_title('Dynamic Batching: Throughput vs Wait Time')
    ax3.grid(True, alpha=0.3)

# Plot 4: Strategy comparison
strategies = ['Simple\nBatch-8', 'Dynamic\nBatch', 'Continuous\nBatch']
strategy_throughputs = [
    batch_results[3]['throughput_req_per_sec'] if len(batch_results) > 3 else 0,
    dynamic_results[1]['throughput_req_per_sec'] if len(dynamic_results) > 1 else 0,
    cb_result['throughput']
]
axes[1, 1].bar(range(len(strategies)), strategy_throughputs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
axes[1, 1].set_ylabel('Throughput (req/s)')
axes[1, 1].set_title('Serving Strategy Comparison')
axes[1, 1].set_xticks(range(len(strategies)))
axes[1, 1].set_xticklabels(strategies)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/llm_serving_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
print('✅ Comparison visualization saved')


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# LLM serving frameworks optimize throughput via batching. Key insight: larger batches improve throughput but increase P99 latency (batching trade-off).
# ### Serving Strategies
# | Strategy | Throughput | P99 Latency | Best For |
# |----------|---|---|---|
# | Simple Batching | Moderate | Low for batch | Static workloads |
# | Dynamic Batching | High | Moderate | Variable request rate |
# | Continuous Batching | Very High | Low overall | High throughput |
# ### Key Technologies
# - **PagedAttention:** Virtual memory for KV cache → larger batches without OOM
# - **Continuous Batching:** New requests join mid-decoding → 3-5x higher throughput
# - **Token-level scheduling:** Fair resource allocation across requests
# ### Common Pitfalls
# - **Blocking on full batch:** Static batching wastes time. Use dynamic batching.
# - **Memory runaway:** Track KV cache growth. Use PagedAttention or eviction.
# - **Unfair latency:** Early requests finish much faster. Use priority queues.
# ======================================================================

# ======================================================================
# ## Exercises
# 1. **Find optimal max_wait_ms:** Vary max_wait_ms to find sweet spot (throughput > 10 req/s, P99 < 200ms)
# 2. **Memory profiling:** Profile memory usage of batch_size=[1, 8, 16]. Where does memory peak?
# 3. **Request rate scaling:** At what request rate does continuous batching break down?
# ======================================================================
