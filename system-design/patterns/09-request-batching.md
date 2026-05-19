# Request Batching

## TL;DR
Accumulate per-request inference queries, batch them together, run single inference pass. 10-100x throughput gain (GPU fully utilized vs idle). Trade-off: add 10-50ms latency vs single-request.

## Core Intuition
Single GPU requests = GPU idle between requests. Batching 32 requests = GPU 32x more efficient. Process once instead of 32 times.

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
**Q: Batch size 32 works great, but latency suddenly spikes. Cause?**
A: Queue backed up. If incoming rate > outgoing rate, queue grows. Investigation: (1) Check GPU util—if 100%, increase batch_size or add replicas. (2) Check model latency—may have degraded. (3) Feature extraction time—blocking on DB queries. Fix: identify bottleneck, unblock.

**Q: Variable-length sequences in a batch?**
A: Option A: Pad to max_length. Simple, wastes compute. Option B: Sort by length, create variable-length batches. More complex, efficient.

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
