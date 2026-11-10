# KV Cache Optimization

**TL;DR:** A KV cache avoids recomputing prior attention keys and values, trading repeated computation for memory whose size grows with tokens, layers, heads and precision.

## Prerequisites

[Transformer attention](11-flash-attention.md) and [long context management](27-long-context-management.md).

## Learning Outcomes

- Calculate KV-cache bytes from sequence and model dimensions.
- Explain paged block allocation and identify fragmentation trade-offs.
- Describe safe prefix sharing and why cache identity must include semantic inputs.
- Identify eviction and quantization effects on correctness and latency.

## Core Intuition

During generation, earlier tokens do not change. Keeping their keys and values is like keeping prepared ingredients instead of repeating all earlier work for every new token. The saved compute consumes memory. At high concurrency, the system must decide which sequences reserve blocks, which common prefixes can be shared, and what to do when capacity is exhausted.

## How It Works

For batch sequences with token counts T_b, L layers, H_kv key/value heads, head dimension D and s bytes per scalar, approximate storage is \(2sLH_{kv}D\sum_bT_b\). The factor 2 accounts for keys and values. This excludes allocator metadata, fragmentation, temporary buffers, replication and quantization scales. Grouped-query attention lowers H_kv relative to query heads. A longer prompt and longer generated output both increase cache demand.

Paged allocators divide sequence state into fixed-size blocks and map logical positions to physical blocks. Prefix sharing can let requests reuse immutable blocks for a common system prompt. Writes to a shared prefix require copy-on-write or immutable-prefix discipline. Eviction policies must respect active references: freeing a block still referenced by another sequence corrupts its cache. Prefix entries need identity that includes model, adapter, positions, masks and relevant prompt content.

The formula is an estimate of payload, not the full allocator reservation. For example, the same number of cached tokens can have different memory use under multi-head attention and grouped-query attention because the latter stores fewer key/value heads. Weight precision does not necessarily equal KV precision: a model can have low-bit weights and a higher-precision cache. Include both in capacity planning. Also account for layer replication across tensor-parallel workers, temporary buffers during prefill, and the fact that allocator blocks round each sequence upward to a block boundary.

Paged allocation is useful because requests have different prompt and output lengths and complete at different times. A contiguous maximum-size reservation wastes space when actual lengths are short. With blocks, a sequence grows by allocating more blocks as tokens arrive. Block size sets a trade-off: smaller blocks reduce internal fragmentation but increase page-table entries and allocation operations; larger blocks simplify metadata while reserving more unused tail capacity. Measure fragmentation as allocated cache capacity divided by live token payload, across a realistic length distribution.

Prefix caching is more than string matching. The key must capture all inputs that change hidden states: exact tokenized prefix, model weights and revision, adapter, positional settings, attention mask, multimodal embeddings and any prefix-specific conditioning. Security boundaries matter too. Sharing a prefix cache across tenants must not expose data through timing or content. A cache hit should be verified as equivalent to recomputing the prefix, and invalidation should follow model or prompt version changes.

Eviction should distinguish active request state from reusable cached prefix state. Active blocks are pinned until the request completes or is safely migrated. Prefix blocks may be evicted when their reuse value is low, but shared prefixes need reference counts or another ownership mechanism. LRU is simple but can evict a large, expensive-to-recompute prefix immediately before a burst of reuse; frequency, size, tenant fairness and recomputation cost may improve the choice. Record hits, bytes reused, eviction cause and recomputation time.

Quantizing the cache can lower memory bandwidth and allow more concurrent requests, but attention repeatedly consumes these values, so approximation error can accumulate over long generations. Evaluate logit divergence and task-level output quality over context length, not just reconstruction error on isolated tensors. Dynamic growth should have an explicit policy when the pool is full: queue, preempt, spill, or reject. Silent truncation changes semantics and makes capacity pressure look like model failure.

When comparing policies, replay the same arrival and completion trace so that block reuse and fragmentation are directly comparable. Measure peak live bytes, allocated-but-unused bytes, prefix hit ratio and cache-miss recomputation. Include a burst after warm-up: a cache that looks efficient under steady traffic may evict a popular prefix just before a synchronized workload needs it. If requests belong to different tenants, report fairness and verify that eviction cannot expose another tenant's cached state.

## Trade-offs

| Technique | Benefit | Cost / boundary |
|---|---|---|
| Fixed contiguous allocation | Simple addressing | Fragmentation and difficult variable-length batching |
| Paged blocks | Flexible allocation | Mapping and allocator overhead |
| Prefix sharing | Saves duplicate prompt storage/work | Correct cache keys and reference counting required |
| KV quantization | Lower bytes per value | Approximation error and dequantization work |

## Implementation with Python

Run `python3 modern-ai/implementations/29-kv-cache-optimization.py`. The formula estimates bytes for specified model dimensions; the small block-pool sketch demonstrates reuse and eviction concepts. Its allocator is instructional and does not model concurrent locking or production reference counts.

```python
import runpy
m = runpy.run_path("modern-ai/implementations/29-kv-cache-optimization.py")
cache_bytes = m.get("cache_bytes")
bytes_required = cache_bytes(tokens=1024, layers=24, kv_heads=8, head_dim=128)
assert bytes_required == 2 * 2 * 1024 * 24 * 8 * 128
```

## Failure Modes

| Failure | Diagnostic | Mitigation |
|---|---|---|
| OOM despite average fit | Peak live-token and fragmentation telemetry | Reserve against burst/concurrency and test long-tail lengths |
| Invalid prefix reuse | Cache-key collision or output mismatch | Include all semantic inputs in key; parity-test cached versus uncached output |
| Premature eviction | Missing or corrupted active sequence blocks | Pin live blocks and use explicit references before reclaiming |

- Capacity estimate is low: peak active token count exceeds the planning percentile; profile concurrency and leave room for temporary buffers and block rounding.
- Prefix reuse is incorrect: cached and uncached outputs differ; expand the cache key and parity-test across adapters, positions and masks.
- Eviction corrupts a live request: block references outlast their owner; pin active state and reclaim only after reference release.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| How does KV memory scale? | Linear in cached tokens and layers, proportional to KV heads, head size and bytes; ask for a concrete estimate. |
| What does paging improve? | Variable-length allocation and fragmentation behavior; follow up on mapping overhead. |
| When is prefix reuse unsafe? | When any model, adapter, prompt, position or mask input differs; ask how to design a cache key. |

## Related Concepts

See [long context](27-long-context-management.md), [speculative decoding](28-speculative-decoding.md), [notebook 29](../notebooks/29-kv-cache-optimization.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#kv-cache-optimization).

## References

[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) and [GQA](https://arxiv.org/abs/2305.13245).
