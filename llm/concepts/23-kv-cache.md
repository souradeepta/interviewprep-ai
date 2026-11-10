# KV Cache

TL;DR: A key-value cache stores the keys and values already computed for a causal transformer so decoding each new token does not repeat projection work for the entire prefix.

## Prerequisites

This lesson builds on [semantic caching](22-semantic-caching.md) only in the broad sense of reusing prior work; a KV cache is an exact, model-state cache and is not a cache of final answers. It assumes familiarity with causal self-attention, autoregressive decoding, tensor dimensions, and the distinction between prefill (processing a prompt) and decode (generating tokens).

## Learning Outcomes

- Derive the approximate KV-cache memory for a decoder from its layer, head, token, and element-size dimensions.
- Explain which computation a cache avoids during autoregressive decoding and which computation remains.
- Diagnose cache growth or incorrect reuse using token counts, memory estimates, and sequence identity.
- Select a cache layout or precision trade-off for a stated latency and memory constraint.

## Core Intuition

Imagine reading a growing conversation while keeping a notebook of the facts you have already extracted. When the next word arrives, you do not reread and re-encode every earlier word; you append the new state and attend to the stored history. In a transformer, the reusable state is each layer's key and value projection for each earlier token. The query for the new token is still computed, and it still compares against prior keys and combines prior values. The cache saves repeat work; it does not make attention to a long history constant-time.

This cache is fundamentally different from semantic caching. Semantic caching may return a previous response to a similar request. A KV cache holds intermediate activations for the exact prefix, model weights, positional convention, and attention state that produced them. Reusing it for a different conversation can leak content or produce incorrect logits. A cache hit is safe only when the serving engine can prove that the prefix and computation configuration are compatible.

## How It Works

For a decoder layer, let the current hidden states be projected into queries, keys, and values. In ordinary causal attention, token t can use keys and values from positions 0 through t. During prompt prefill, the model processes many prompt tokens in parallel and produces K and V tensors for every layer. During one-token decoding, only the newly generated token needs new projections. Its query attends to the concatenation of cached keys and the new key, and the resulting attention weights combine cached values and the new value.

For batch size B, L layers, H_kv key/value heads, head dimension d, sequence length T, and b bytes per element, the uncompressed cache size is approximately:

`M_KV = 2 * B * L * T * H_kv * d * b` bytes.

The factor two accounts for keys and values. This estimate excludes allocator padding, block tables, scales for quantized caches, temporary activations, and framework metadata. With conventional multi-head attention, H_kv is often the number of query heads. Multi-query or grouped-query attention uses fewer KV heads, reducing cache memory while sharing K/V across query heads. It changes model architecture and quality/training behavior; it is not simply a cache compression switch for an arbitrary checkpoint.

Without a cache, a naïve implementation recomputes the prefix projections at each decode step. With a cache, each generated token adds roughly one token's K/V state per layer. The attention operation for the new token still reads a prefix of length T, so its attention arithmetic is proportional to T times the attention width. Across N generated tokens after a prompt of P tokens, attention work still grows with the attended context; caching mainly avoids repeatedly projecting all P prior tokens and avoids recomputing prior-token states. Exact total complexity depends on what the baseline recomputes and on the kernel, so “O(1) per token” is incorrect.

Caching usually improves decode latency, but consumes memory in proportion to active tokens and concurrent sequences. At a fixed device-memory budget, KV capacity competes with model weights, temporary buffers, and fragmentation. The practical limit may therefore be the number of live tokens rather than the model's nominal context window. Cache allocation can be contiguous, paged into blocks, or dynamically grown; each strategy trades allocation simplicity against fragmentation, copying, and bookkeeping.

## Trade-offs

| Choice | Benefit | Cost or assumption | Useful diagnostic |
|---|---|---|---|
| Full-precision KV cache | Preserves checkpoint's activation precision closely | Highest bytes per active token | Bytes per token and peak allocated memory |
| Lower-precision KV cache | More active tokens may fit in memory | Quantization error can affect attention, especially at long contexts | Quality by context length and cache dtype |
| Multi-query/grouped-query architecture | Fewer stored KV heads | Requires a model trained or adapted for that architecture | KV-head count and task-quality comparison |
| Paged/block cache | Reduces external fragmentation and supports flexible scheduling | Block tables and allocator logic add complexity | Free blocks, fragmentation, allocation failures |
| Recompute or cache eviction | Reduces resident memory | Adds latency or loses exact full-history access | Recomputed tokens, evictions, and latency tails |

For example, with B=1, L=32, T=4096, H_kv=8, d=128, and two-byte elements, the estimate is 536,870,912 bytes, or 512 MiB in binary units. If H_kv rises to 32, the estimate becomes 2 GiB. The real allocation can differ, but the ratio is informative when other dimensions and dtype remain fixed. Doubling batch size or sequence length doubles this estimate; doubling both multiplies it by four.

## Implementation

The following standard-library-only example computes the idealized cache size and checks scaling relationships. It does not simulate GPU allocation, kernel latency, or cache correctness for a particular model; those require the target runtime and workload.

```python
def kv_cache_bytes(batch, layers, tokens, kv_heads, head_dim, bytes_per_value):
    dimensions = (batch, layers, tokens, kv_heads, head_dim, bytes_per_value)
    if any(not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("all cache dimensions must be positive integers")
    return 2 * batch * layers * tokens * kv_heads * head_dim * bytes_per_value


base = kv_cache_bytes(1, 32, 4096, 8, 128, 2)
assert base == 536_870_912  # 512 MiB, keys and values included
assert kv_cache_bytes(2, 32, 4096, 8, 128, 2) == 2 * base
assert kv_cache_bytes(1, 32, 8192, 8, 128, 2) == 2 * base
assert kv_cache_bytes(1, 32, 4096, 32, 128, 2) == 4 * base

try:
    kv_cache_bytes(1, 32, 0, 8, 128, 2)
except ValueError:
    pass
else:
    raise AssertionError("zero-length dimensions should be rejected")
```

In an actual serving system, inspect active cached tokens per request, cache bytes by dtype, allocation failures, evictions, and p50/p95/p99 time per output token. Compare cached and uncached generation only with identical prompts, model revision, sampling settings, and hardware. Warm up kernels, synchronize the device when timing asynchronous work, and report multiple runs; a one-off wall-clock ratio is not a reliable speedup estimate.

## Failure Modes

- **Cross-request state reuse:** A cache from another user or a changed prefix can corrupt output or expose context. Bind cache entries to a verified token prefix, model revision, adapter, positional settings, and tenant boundary; test that changing any identity component invalidates reuse.
- **Memory exhaustion at concurrency peaks:** A cache sized for one request may fail when many long prompts remain active. Estimate bytes from live tokens, reserve room for weights and workspaces, and alert on free blocks, allocation retries, and OOMs.
- **Misreading cache as constant-time attention:** Decode still reads the attended history. Plot per-token latency against context length; rising latency can be expected even when cache reuse is correct.
- **Incorrect position or mask handling:** Reusing K/V with an inconsistent position offset or causal mask changes attention semantics. Compare cached token-by-token logits with a full-prefix reference on a short deterministic input.
- **Unbounded conversation growth:** Retaining every turn eventually exhausts the token budget. Track tokens per sequence and enforce a policy such as truncation, summarization, or a supported sliding window; each policy changes available context.
- **Assuming framework cache formats are portable:** Tensor layout and metadata differ across runtimes and versions. Use documented conversion paths or rebuild the cache, and include runtime/model versions in reproducibility records.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| What does a KV cache avoid? | It reuses prior layers' K/V projections during autoregressive decoding; new queries and attention over the prefix remain. | Debug why per-token latency still rises with context length. |
| How do you estimate memory? | `2 * B * L * T * H_kv * d * bytes_per_value`, plus runtime overhead. | Design a capacity estimate for peak live tokens. |
| Why does GQA help? | It reduces the number of distinct KV heads stored, given an architecture that shares KV heads. | Compare quality and memory for two checkpoints. |
| How would you find stale cache reuse? | Compare cached and full-prefix logits and verify prefix, model, adapter, and position identity. | Design a safe cache key across tenants. |

## Related Concepts

- [Attention Optimization](24-attention-optimization.md) explains kernels and architectural changes that complement caching.
- [Continuous Batching](26-continuous-batching.md) schedules requests whose KV allocations have different lifetimes.
- [Inference Optimization](28-inference-optimization.md) places caching among broader serving techniques.
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) describes recurrence across segments; it is related context reuse, not the same serving cache mechanism.

## Practice

- [KV cache interview practice](../interview-prep/curriculum-practice.md#kv-cache) includes a memory estimate, cache reuse diagnosis, and validation plan.

## References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) presents block-based KV memory management and serving evaluation.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) studies KV-head sharing and its memory/quality trade-off.
