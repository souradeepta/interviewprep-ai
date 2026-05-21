# Request Batching

## Detailed Description

Accumulate incoming per-request queries, batch them, run single inference pass. Process 32 requests simultaneously = 32x speedup vs single. Trade-off: latency vs throughput. Batch size: 32-256 (depends on memory). Max wait: 10-100ms.

## Core Intuition

Batching = fill a bus before driving. Single requests = driving bus with 1 passenger. Batch 32 requests = 32 passengers, same trip. Throughput 32x better. Latency: individual request waits (50ms for batch to fill). Throughput win outweighs latency cost for most systems.

## How It Works

**Batching queue:**
- Requests arrive, get queued
- When batch_size reached OR max_wait_ms elapsed → run inference
- Return responses to all queued requests

| Config | Throughput | GPU Util | Latency |
|--------|-----------|----------|---------|
| batch=1 | 20 req/s | 5% | 50ms |
| batch=32 | 600 req/s | 95% | 100ms |
| batch=256 | 2000 req/s | 99% | 200ms |

## Key Properties / Trade-offs
- Batch size 1: low latency, low throughput
- Batch size 256: high throughput, high latency
- Sweet spot: 32-64 for most models

## Common Mistakes / Gotchas
- Batch size too small (1-2): GPU idle, defeats purpose
- No wait time limit: requests queue forever waiting for batch
- Queue overflow: no backpressure → OOM
- Variable input sizes: can't batch sequences of different lengths

## Detailed Trade-off Analysis

### Batch Size Tuning Comparison

| Batch Size | Throughput | P99 Latency | GPU Util | Queue Wait | Memory | Use Case |
|------------|-----------|-------------|----------|-----------|--------|----------|
| **1** | 20/sec | 50ms | 5% | 0ms | 100MB | Testing/debug only |
| **8** | 120/sec | 75ms | 40% | 10ms | 300MB | Low-latency priority (SLA <100ms) |
| **32** | 600/sec | 100ms | 95% | 30ms | 800MB | **RECOMMENDED** (balanced) |
| **64** | 1000/sec | 130ms | 98% | 60ms | 1.2GB | High throughput |
| **256** | 2000/sec | 200ms | 99% | 100ms | 3GB | Very high throughput, latency flexible |

### Cost Model (1M requests/day, 100 QPS peak)

**Without batching (batch=1):**
- GPU needed: 100 QPS × 50ms latency = 5 GPU seconds / second = 5 GPUs at 100% util
- Cost: 5 GPUs × $0.50/hour × 24h = $60/day = $1,800/month

**With batching (batch=32):**
- GPU needed: 100 QPS / 600 req/sec per GPU = 0.17 GPUs (share 1 GPU)
- Cost: 1 GPU × $0.50/hour × 24h = $12/day = $360/month
- **Savings: $1,440/month (80% reduction!)**

**With batching (batch=256, higher latency tolerance):**
- GPU needed: 100 QPS / 2000 req/sec per GPU = 0.05 GPUs
- Cost: 1 GPU × $0.50/hour × 5% utilization = $0.60/day = $18/month
- **Savings: $1,782/month (99% reduction)**

### Decision Matrix by Constraint

| Primary Constraint | Recommended Batch Size | Reasoning | Example |
|-------------------|------------------------|-----------|---------|
| **Latency SLA <100ms** | 8-16 | Keep queue wait <20ms | Real-time fraud detection |
| **Balanced (latency + cost)** | 32-64 | 100-130ms latency, 95%+ GPU util | Recommendation systems |
| **Throughput-first** | 128-256 | Maximize requests/sec, latency flexible | Batch processing, offline jobs |
| **Memory-constrained** | 16-32 | Large model can't fit batch>32 | LLM inference (model >10GB) |
| **Variable input sizes** | 8-16 | Padding waste increases with size | NLP tasks (variable sequence length) |

---

## Production Failure Scenarios

**Scenario 1: Batch Timeout Too Long (Latency SLA Breach)**

**What breaks:** Batch size=256, max_wait=200ms. SLA requires P99<100ms. In practice, P99 latency is 150ms because requests wait up to 200ms for batch to fill. 50% of SLA budget consumed by batching alone.

**Why it happens:**
- Batch size chosen for throughput, not latency
- No awareness of SLA constraint in batching config
- Assumption: "larger batch = better always" is wrong

**Detection:**
```
Alert: if (P99_latency > SLA_threshold) → WARN
Monitor: histogram of queue_wait_time, see tail hitting 200ms
```

**Recovery:**
1. Reduce batch size: 256 → 64 (reduces max_wait 200ms → 60ms)
2. Reduce max_wait timeout: 200ms → 50ms
3. Accept partial batches: if timeout triggers, infer smaller batch

**Prevention:**
- Set max_wait based on SLA: `max_wait = SLA * 0.3` (leave 70% for actual inference)
- Monitor P99 latency distribution, alert if trending up
- Dynamic batch sizing: reduce batch_size if P99 > SLA

---

**Scenario 2: Queue Overflow (Memory Explosion)**

**What breaks:** Peak traffic spike (100 QPS → 500 QPS). Batching system can process 100 QPS but requests arrive at 500 QPS. Queue accumulates: 100 requests queued after 1 sec, 400 after 2 sec, 800 after 3 sec. At 5 seconds: 2000 requests queued = 100MB RAM. At 10 seconds: 4000 requests = OOM crash (out of memory).

**Why it happens:**
- No backpressure mechanism (queue unbounded)
- Spike in traffic (viral feature, DDoS, etc)
- Assumption: "queue handles all requests" but memory is finite

**Detection:**
```
Metric: queue_depth (count of pending requests)
Alert: if (queue_depth > threshold * 0.8) → WARN
Alert: if (queue_depth > threshold) → CRITICAL (OOM imminent)

Check memory: if (memory_used > 80% of allocated) → CRITICAL
```

**Recovery:**
1. Immediate (30 sec): Start rejecting new requests (circuit breaker)
   - Return 503 Service Unavailable to clients
   - Clients see overload, implement backoff
2. Short-term (2-5 min): Investigate spike cause
   - DDoS? Feature viral? Legitimate traffic?
3. Scale up: Deploy additional GPU instances if legitimate traffic

**Prevention:**
- Implement queue size limit: max 1000 pending requests
- Circuit breaker: if queue > 1000, reject with 503
- Auto-scaling trigger: if queue > 500 consistently, add GPU
- Monitor queue depth, alert if > 800

---

**Scenario 3: Variable Sequence Lengths (Padding Waste)**

**What breaks:** NLP model batching sequences with variable lengths (5, 500, 20, 480 tokens). To batch, pad all to max length (500 tokens). Only 20% of compute is actual work, 80% wasted on padding. Effective throughput is 20% of batch size.

**Why it happens:**
- NLP tasks have inherent variable-length inputs
- Naive batching pads all to max
- Padding reduces actual throughput despite high "batch size"

**Detection:**
```
Metric: padding_efficiency = avg_length / max_length in batch
Alert: if (padding_efficiency < 0.5) → WARN (50% wasted compute)
Monitor: why is efficiency low? (too many short + few long sequences?)
```

**Recovery:**
1. Bucket by length: Sort sequences by length before batching
   - Batch 1: lengths 1-50 (max_pad=50)
   - Batch 2: lengths 51-200 (max_pad=200)
   - Batch 3: lengths 200+ (max_pad=500)
   - Result: 90%+ efficiency, 2-3x actual throughput improvement

2. Alternative: Dynamic padding (for specific models)
   - Only pad within batch (don't pad to global max)
   - Extra complexity but recovers lost efficiency

**Prevention:**
- Profile padding efficiency in production
- If <70%, implement length bucketing
- Monitor per-batch efficiency, alert on degradation

---

## Implementation Guidance

**Wrong:** No timeout, requests queue forever.
```python
batch.add(request)  # Waits indefinitely
```

**Right:** Timeout after max_wait_ms.
```python
batch.add(request, timeout=50ms)  # Times out if batch not full in 50ms
if batch.full() or batch.timed_out():
    infer(batch)
```

---

## Sophisticated Interview Q&A

**Q1: Batch 32 adds 30ms latency. SLA is 50ms total. Accept the trade-off?**

A: Yes, with caveats. 30ms of 50ms SLA = 60% consumed by batching, 40% for actual inference. This is workable IF:
- Actual inference takes <20ms (you have headroom)
- P50 latency is lower (batching only affects tail)
- Throughput gain (30x) justifies slight latency hit

If P99 latency breaches SLA:
- Reduce batch size: 32 → 16 (sacrifices 15% throughput for 15ms latency)
- Reduce max_wait: 30ms → 20ms (requests don't queue as long)
- Accept partial batches: if timeout hits before batch fills

**Follow-up:** How would you measure if 32 is actually acceptable?
- Answer: A/B test: 10% traffic on batch=32, 90% on batch=16. Compare P99 latency and user-perceived latency.

---

**Q2: Queue grows to 5000 requests during a traffic spike. System about to OOM. What's your immediate action?**

A: **Immediate (next 30 seconds):**
1. Enable circuit breaker: reject new requests with 503
   - Prevent queue from growing further
   - Clients see overload, implement backoff
2. Check: Is this legitimate spike or DDoS?
   - Look at request source distribution
   - If DDoS: block malicious IP ranges

**Short-term (next 5 minutes):**
1. Investigate root cause: Why did traffic spike?
   - Feature viral? Bug causing infinite retries? Cache cleared?
2. Add GPU resources if legitimate spike
3. Monitor for recovery once upstream is fixed

**Prevention (next sprint):**
- Implement queue size limit (max 1000)
- Auto-scaling: add GPU when queue > 500
- Client backoff: exponential backoff on 503 errors

---

**Q3: Batch size tuning: Start with 32. How do you determine if increase to 64 or decrease to 16?**

A: Measure these metrics:
```
P99_latency = is it > SLA?
Throughput = is it < target?
GPU_util = what % of GPU is utilized?
```

**Decision logic:**
- If P99_latency > SLA: decrease batch size (trade throughput for latency)
- If throughput < target AND P99_latency OK AND GPU_util < 90%: increase batch size
- If GPU_util = 99% AND throughput still < target: add more GPUs (not batching problem)

**Example trace:**
```
Batch=32: P99=95ms (OK, <100ms SLA), Throughput=600/sec (OK, >500 target), GPU=95%
→ Sweet spot, keep it

Batch=64: P99=140ms (BAD, >100ms SLA), Throughput=1000/sec (OK), GPU=98%
→ Latency violated, revert to 32
```

---

**Q4: Variable sequence lengths (NLP): batch=32 but padding efficiency drops to 40%. What's happening?**

A: Example: batch contains sequences [5, 500, 10, 490, 20, 480, ...]. All padded to 500.
- 5 tokens → 495 padding waste
- 500 tokens → 0 padding
- Average: 250 tokens actual, 250 padding
- Efficiency: 250/500 = 50%

**Root cause:** Natural language has variable length. Some short, some long in same batch.

**Fix: Length bucketing**
```
Bucket 1: lengths 1-100 (max_pad=100, efficiency=80-100%)
Bucket 2: lengths 100-300 (max_pad=300, efficiency=60-100%)
Bucket 3: lengths 300+ (max_pad=500, efficiency=60-100%)
```

Result: average efficiency 80%, actual throughput 2x improvement over naive batching.

---

**Q5: You're choosing between batch=32 (100ms P99) and batch=1 (50ms P99). Latency SLA is 100ms. Which?**

A: Counterintuitively, choose **batch=32** unless throughput is extremely low.

Why? Two reasons:
1. **SLA budget has other components:** 100ms SLA includes network, preprocessing, inference, postprocessing
   - If just inference is 100ms SLA, then batch=32 breaches it
   - But if total system is 100ms SLA and inference is only 30% of it, batch=32 still fits
   
2. **Cost efficiency:** batch=1 requires 30x more GPUs
   - Cost: $1,800/month vs $360/month
   - Unless extreme latency requirement, cost savings worth it

**Decision:** Quantify other latencies. If they sum to >40ms, use batch=32. Otherwise batch=1 or batch=8.

---

**Q6: Batch size 256, but QPS fluctuates 10-500 throughout day. Static batch size workable?**

A: Static batch=256 is problematic:
- **Off-peak (10 QPS):** batch takes 26 seconds to fill (256/10), but max_wait timeout fires at 100ms. So serving partial batches ~10-30 size.
- **Peak (500 QPS):** batch fills in <1 second, good throughput but high latency variance

**Better approach: Dynamic batching**
```python
if QPS > 300:
    batch_size = 256  # high throughput
elif QPS > 100:
    batch_size = 64
else:
    batch_size = 16   # off-peak, keep latency low
```

Or: use max_wait to auto-adapt. Let batch_size be static 64, adjust only max_wait:
```python
max_wait = max(50ms, 3000/QPS)  # wait longer if QPS is low
```

---

**Q7: How handle model inference failures within a batch (1 of 32 requests has bad input)?**

A: **Option 1: Partial failure (recommended)**
```python
try:
    results = model.predict(batch)  # inference fails on 1 sample
except:
    # Retry individually
    for i, sample in enumerate(batch):
        try:
            results[i] = model.predict([sample])
        except:
            results[i] = ERROR  # mark as failed
    return results  # 31 OK, 1 failed
```

**Option 2: Strict (all-or-nothing)**
- If any sample fails, fail entire batch
- Simpler but loses 31 good predictions

**Option 3: Validate before batch**
- Pre-check inputs before batching
- Reject bad inputs before they reach model
- Cleanest but requires robust validation

**Recommendation:** Option 1 (partial failure). Maximum salvage of good predictions.

---

**Q8: How measure if batching is actually helping in production?**

A: Compare two scenarios:
```
Batch=1 baseline:
  - GPU util: 5%
  - Throughput: 20 req/sec
  - Hardware needed: 5 GPUs to handle 100 QPS peak

Batch=32 with batching:
  - GPU util: 95%
  - Throughput: 600 req/sec
  - Hardware needed: 1 GPU to handle 100 QPS peak
  - Latency increase: 50ms → 100ms (acceptable?)

ROI: Reduce 5 GPUs to 1 GPU = save $2,400/month
Cost of 50ms latency increase: affects CTR? Revenue?
```

If 50ms latency has <0.1% CTR impact and saves $2.4K/month, batching is hugely beneficial.

---

## Cost & Resource Analysis

Batching reduces infrastructure 30-50% (fewer GPUs needed for same throughput).

---

## Monitoring & Observability

Metrics: batch_size_actual, queue_length, latency_added_by_batching. Alert: queue>1000, latency>SLA.

## Best Practices
- **Start small:** batch=8, measure throughput/latency
- **Adapt batch size:** increase if throughput <target, decrease if latency >SLA
- **Padding:** pad shorter sequences to match batch, or sort by length
- **Backpressure:** if queue>1000, apply client backoff (circuit breaker)
- **Monitor:** track batching efficiency = (actual_batch_size / target_batch_size)

## Code Example
```python
import asyncio, queue
import numpy as np

class BatchProcessor:
    def __init__(self, model, batch_size=32, max_wait_ms=50):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        self.queue = asyncio.Queue()
    
    async def forward(self, features):
        future = asyncio.Future()
        await self.queue.put((features, future))
        return await future
    
    async def batch_worker(self):
        while True:
            batch = []
            deadline = asyncio.get_event_loop().time() + self.max_wait_ms
            
            while len(batch) < self.batch_size:
                try:
                    wait_time = deadline - asyncio.get_event_loop().time()
                    if wait_time <= 0:
                        break
                    item = await asyncio.wait_for(self.queue.get(), timeout=wait_time)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                features = [item[0] for item in batch]
                predictions = self.model.predict(np.array(features))
                for (_, future), pred in zip(batch, predictions):
                    future.set_result(pred)
```

## Interview Q&A

Q: Single request latency 50ms. Batch 32 requests?
A: A: Latency per request: 50ms (fill batch) + 10ms (inference) = 60ms. Single request: 50ms. Trade: 10ms slower, but 32x more throughput. Worth it for most use cases.

Q: Batch size too small (1-2)?
A: A: GPU idle, low throughput. Increase to 32, GPU utilization jumps from 10% to 95%.

Q: Max wait time too long (>200ms)?
A: A: Queued requests wait too long. SLA violation. Lower max_wait to 50ms (batch fills faster with traffic, or timeout and send smaller batch).

Q: No backpressure: queue grows unbounded?
A: A: OOM. Add backpressure: if queue >1000, reject new requests (HTTP 503). Signals to client: server busy.

Q: Batching different models together?
A: A: Don't. Incompatible batch dimensions. Separate queue per model. Batch independently.

Q: Dynamic batch size?
A: A: Start small (8), increase if queue_depth >50. Adapt to traffic. High traffic → larger batches. Low traffic → smaller batches.

Q: Fallback if wait exceeds SLA?
A: A: Bypass queue, process single request (slow but meets SLA). Or return cached prediction if available.

Q: Variable-length inputs (text sequences)?
A: A: Pad short sequences to max_length (wastes compute but simple). Or sort by length, batch similar lengths (more complex, efficient).
## Interview Quick-Reference
| Config | Throughput | Latency | When |
|--------|-----------|---------|------|
| 1 | 20/s | 50ms | Single-threaded, low load |
| 32 | 600/s | 100ms | Standard, balanced |
| 256 | 2000/s | 200ms | High throughput, latency-tolerant |

## Related Topics
- [Model Serving](05-model-serving.md)
- [Load Balancing](10-load-balancing.md)

## Resources
- [Clipper: Prediction Serving System](https://arxiv.org/abs/1612.03079)

