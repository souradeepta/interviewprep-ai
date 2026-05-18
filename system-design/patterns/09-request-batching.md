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

## Interview Q&A

**Q: What is the difference between static batching and dynamic batching for ML inference?**
A: Static batching: fixed batch size, pad shorter sequences, wait for the batch to fill before processing. Simple to implement but inefficient when traffic is bursty or sequence lengths vary greatly. Dynamic batching: batch requests as they arrive, process when either max batch size or max latency timeout is reached. More complex but significantly better GPU utilization. Dynamic batching with a 10ms max-wait typically achieves 5-10x higher throughput than static batching at similar latency, because the GPU processes full batches more often.

**Q: How does request batching interact with SLA requirements and queue management?**
A: Larger batches improve throughput but increase latency—a request that waits for a batch to fill waits longer than a request in a singleton batch. Design: set max-batch-wait-time based on your TTFT SLA (e.g., if SLA is 500ms and model inference is 200ms, max wait time is 200ms). Implement priority queues: premium-tier requests have shorter max-wait times. Monitor queue depth: if queue consistently exceeds 10 requests, add capacity. Queue depth is a leading indicator of latency degradation before the SLA is actually violated.

**Q: What padding and masking considerations affect batching efficiency?**
A: For variable-length sequences (text, time series), shorter sequences must be padded to match the longest sequence in the batch. If one sequence is 512 tokens and all others are 50 tokens, 90% of compute is wasted on padding. Mitigate: sort sequences by length before batching (bucket by length range), use dynamic padding (pad only to the max length in each batch), implement sequence packing (concatenate multiple short sequences into one long sequence). These techniques can improve throughput 2-4x for highly variable length inputs.

**Q: How do you implement adaptive batching that responds to traffic changes?**
A: Adaptive batching adjusts batch size based on current queue depth and latency observations. Algorithm: start with max-wait-time=T. If recent P95 latency is above SLA: decrease max-wait-time (process smaller batches faster). If queue depth is growing: increase batch size to process more per cycle. If GPU utilization is low: increase batch size. Implement a PID controller or simple threshold rules. Test the adaptive controller under different traffic patterns: sudden spike, sustained high load, gradual increase.

**Q: What failure modes does batching introduce that singleton request serving doesn't have?**
A: Batch failure modes: one bad request in a batch fails the entire batch (implement per-request error isolation), head-of-line blocking (large requests delay small ones in the same batch), batch timeout cascade (when batches time out, the queue grows and subsequent batches also time out), and memory allocation failure for oversized batches. Mitigate: validate requests individually before batching, implement request size limits, use separate queues for large and small requests, and implement circuit breakers that degrade to smaller batches under memory pressure.

## Interview Quick-Reference
**Request batching?** Group requests, process together. 10-100x throughput, higher per-request latency.

## Related Topics
- [Model Serving](05-model-serving.md)
- [Inference Optimization](../llm/concepts/inference-optimization.md)

## Resources
- [Clipper: Batching Inference](https://arxiv.org/abs/1611.08613)
