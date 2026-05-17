# Request Batching

## TL;DR
Group requests, process together, return results. Throughput: 10-100x better. Latency: higher per-request. Trade-off: delay tolerance vs throughput.

## Core Intuition
Process one request slowly. Process 100 requests together only slightly slower → batch wins on throughput.

## How It Works
```
Request arrives → queue
When queue size ≥ batch_size OR timeout → process batch
Return results to all requests in batch
```

**Example:**
- Individual: 10ms per prediction
- Batch of 64: 15ms total (60x faster per pred)

## Trade-offs
- Throughput: 10-100x
- Latency: higher (wait for batch)
- Cost: much cheaper per request

## Common Mistakes / Gotchas
- **Timeout too long:** requests wait forever
- **Batch too large:** exceed memory
- **Imbalanced batches:** last request waits alone

## Interview Quick-Reference
**Request batching?** Group requests, process together. 10-100x throughput, higher per-request latency.

## Related Topics
- [Model Serving](model-serving.md)
- [Inference Optimization](../llm/concepts/inference-optimization.md)

## Resources
- [Clipper: Batching Inference](https://arxiv.org/abs/1611.08613)
