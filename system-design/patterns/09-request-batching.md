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

| Batch Size | Throughput | Latency | GPU Util | Queue Wait |
|------------|-----------|---------|----------|-----------|
| 1 | 20/sec | 50ms | 5% | 0ms |
| 8 | 120/sec | 75ms | 40% | 10ms |
| 32 | 600/sec | 100ms | 95% | 30ms |
| 256 | 2000/sec | 200ms | 99% | 100ms |

**Decision:** Start batch=32. If latency >SLA, reduce. If throughput <target, increase.

---

## Production Failure Scenarios

**Scenario 1: Batch timeout too long**
- Request waits 500ms for batch. SLA 100ms. Timeout.
- Fix: Set max_wait=50ms. Accept partial batches.

**Scenario 2: Queue overflow**
- No backpressure. Queue grows to 10K. OOM crash.
- Fix: Reject requests if queue>1000. Circuit breaker.

**Scenario 3: Variable sequence lengths**
- Batch 32 sequences. Lengths: 10, 500, 20, 480... Can't batch different lengths.
- Fix: Pad to max length or sort by length before batching.

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

**Q1: Batch 32 adds 30ms latency. SLA 50ms. Accept?**
A: Yes. Throughput gain (30x) outweighs 30ms latency for most systems. If SLA violated, reduce batch to 8-16.

**Q2: Queue growing, requests piling up. Fix?**
A: Reject if queue>threshold. Circuit breaker. Force clients to backoff. Prevents OOM.

**Q3: Batch size tuning strategy?**
A: Start 32. Measure throughput/latency. If latency >SLA, reduce. If throughput <target, increase. Find sweet spot.

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

