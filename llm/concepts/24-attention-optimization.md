# Attention Optimization

TL;DR: Attention optimization changes how attention is computed or represented to reduce memory traffic, arithmetic, or KV storage while preserving the model's intended attention semantics as closely as the method allows.

## Prerequisites

This lesson assumes the causal attention operation and the [KV cache](23-kv-cache.md). It focuses on exact attention kernels and architectural or algorithmic variants; it does not treat every technique as interchangeable or claim that one kernel accelerates every workload.

## Learning Outcomes

- State the standard scaled dot-product attention equations and their dense complexity in sequence length.
- Explain how tiled exact attention reduces intermediate memory traffic without materializing the full score matrix.
- Distinguish kernel optimizations from sparse patterns and grouped-query architecture changes.
- Diagnose whether a workload is limited by arithmetic, memory bandwidth, launch overhead, or cache capacity.

## Core Intuition

Consider calculating every pairwise relationship in a long sequence while writing the entire score matrix to a large scratchpad. Much of the cost comes from moving that matrix between high-bandwidth memory and the compute units, not only from the multiply operations. FlashAttention changes the calculation order: it loads a tile of queries, keys, and values, computes local scores, and updates a compact running softmax summary. It avoids storing the complete sequence-by-sequence score matrix in high-bandwidth memory while still computing dense attention.

This is unlike sparse attention, which removes selected query-key interactions and therefore changes the attention pattern unless that pattern is already part of the model design. It is also unlike grouped-query attention (GQA), which changes how query heads share key/value heads and primarily reduces KV-cache size. These methods can be combined, but each addresses a different resource and has different correctness assumptions.

## How It Works

For queries Q, keys K, values V, and head dimension d, scaled dot-product attention is `softmax(Q K^T / sqrt(d) + M) V`, where M encodes causal or other masking. For sequence length T, dense attention's score computation and value aggregation require O(T²d) arithmetic per head and a naïve implementation may materialize O(T²) scores. The softmax must be numerically stable and respect masking, including rows where some keys are disallowed.

FlashAttention tiles Q, K, and V so intermediate score tiles can remain in fast on-chip memory. For each query row, it maintains a running maximum m, normalized exponential sum l, and weighted-value accumulator. When a new key/value tile arrives, the old accumulator and new tile are rescaled to a common maximum before combination. This online softmax recurrence yields the same dense mathematical result, up to floating-point roundoff, without retaining the full score matrix. The implementation remains quadratic in sequence length in arithmetic; the gain is primarily I/O efficiency and reduced intermediate-memory use. Kernel versions also differ in work partitioning and support for head sizes, masks, dropout, and hardware.

Sparse attention restricts each query to a subset of keys. If a pattern attends to w positions per token, idealized work can approach O(Twd), but the actual speedup depends on regularity, index overhead, and kernel support. A random sparse pattern can be slower than dense fused attention at moderate lengths. Local windows can also remove long-range information, so evaluate quality on tasks that require distant dependencies.

GQA uses fewer KV heads than query heads. If there are H_q query heads and H_kv KV heads, with H_kv < H_q, keys and values are shared among groups of query heads. This reduces KV-cache storage approximately in proportion to H_kv/H_q relative to standard multi-head attention. It does not eliminate the dense query-key work per query head, and converting a trained model to GQA is an architectural/model-quality question rather than a transparent kernel toggle.

Performance therefore depends on phase and shape. Prefill processes many prompt tokens and can be dominated by attention compute or memory traffic. Single-token decode reads the KV history and may be bandwidth-bound. Short sequences may be dominated by launch overhead, while long contexts stress memory and quadratic arithmetic. Benchmark the actual batch, sequence lengths, dtype, mask, and hardware.

## Trade-offs

| Technique | What it reduces | Main cost or assumption | Diagnostic for selection |
|---|---|---|---|
| Fused/tiled exact attention | Intermediate score materialization and memory traffic | Shape, dtype, mask, and device support vary | Peak memory and prefill latency against a correct dense baseline |
| Sparse/local attention | Number of attended pairs | Pattern may omit useful distant context; irregular kernels add overhead | Quality by distance and measured latency at target lengths |
| GQA/MQA | KV-cache bytes per token | Requires compatible model architecture; query-head arithmetic remains | Cache bytes, decode throughput, and task quality |
| Lower precision | Bytes moved and sometimes arithmetic cost | Rounding or underflow can alter output | Error and quality by dtype, head size, and context |
| Longer context without sparsity | Context coverage | Dense compute grows quadratically with T | Prefill time and memory across context lengths |

For exact attention, compare output tensors with a trusted reference under the same mask and tolerance. Floating-point reduction order means bitwise equality is usually not expected. For approximate or sparse methods, a numerical match may be inappropriate; compare task quality, long-distance retrieval, and latency against the original architecture.

## Implementation

This dependency-free example computes dense attention and verifies both causal masking and row normalization. It is a transparent correctness reference, not an optimized GPU kernel and not a benchmark. Python's double-precision arithmetic makes it suitable for illustrating equations on a tiny input only.

```python
import math


def causal_attention(q, k, v):
    if not q or len(q) != len(k) or len(k) != len(v):
        raise ValueError("q, k, and v must have the same nonzero sequence length")
    width = len(q[0])
    if width == 0 or any(len(row) != width for rows in (q, k, v) for row in rows):
        raise ValueError("all vectors must have one consistent, nonzero width")
    result, probabilities = [], []
    for i, query in enumerate(q):
        scores = [sum(a * b for a, b in zip(query, key)) / math.sqrt(width)
                  for key in k[:i + 1]]
        peak = max(scores)
        weights = [math.exp(score - peak) for score in scores]
        total = sum(weights)
        weights = [weight / total for weight in weights]
        output = [sum(weights[j] * v[j][column] for j in range(i + 1))
                  for column in range(width)]
        result.append(output)
        probabilities.append(weights)
    return result, probabilities


q = [[1.0, 0.0], [0.0, 1.0]]
k = [[1.0, 0.0], [0.0, 1.0]]
v = [[10.0, 0.0], [0.0, 10.0]]
output, weights = causal_attention(q, k, v)
assert len(weights[0]) == 1  # token zero cannot see token one
assert abs(sum(weights[1]) - 1.0) < 1e-12
assert output[0] == [10.0, 0.0]
assert all(math.isfinite(x) for row in output for x in row)
```

When evaluating a production kernel, measure prefill and decode separately, sweep representative sequence lengths and batch sizes, and synchronize accelerator work around timing. Record peak allocated memory as well as latency. Check masks, padding, cache position offsets, and numerical tolerance before comparing speed. A faster result that changes causal visibility or silently ignores a mask is a correctness bug.

## Failure Modes

- **Calling a fused kernel asymptotically linear:** FlashAttention still performs dense pairwise attention arithmetic. Inspect FLOP scaling and benchmark increasing T; reduced memory use is not proof of linear compute.
- **Comparing unlike kernels or workloads:** A benchmark with different batch size, dtype, causal mask, or warm-up can misattribute speed. Hold these conditions constant and profile prefill and decode separately.
- **Wrong causal or padding mask:** Fast kernels may use different mask conventions or reject unsupported layouts. Test a tiny input where future tokens have distinctive values and verify they cannot affect earlier outputs.
- **Numerical instability in a custom online softmax:** Exponentiating raw scores can overflow. Subtract the running maximum, use stable accumulation, and compare against a high-precision reference on large-magnitude scores.
- **Sparse pattern harms long-range tasks:** Local attention can miss distant evidence even when average benchmark scores look stable. Add distance-controlled retrieval tests and inspect errors by dependency distance.
- **Assuming GQA is a drop-in runtime option:** Head sharing is encoded in model weights and configuration. Verify model architecture and checkpoint support; compare quality after any conversion or fine-tuning.
- **Unsupported shape silently falls back:** Runtime dispatch may choose a slower implementation for a mask, head dimension, or dtype. Capture profiler/kernel selection and report fallback frequency.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| How does FlashAttention help? | It tiles the exact softmax-attention calculation to reduce high-bandwidth-memory traffic and avoid materializing all scores. | Explain why arithmetic remains quadratic in sequence length. |
| How is sparse attention different? | Sparse attention omits some query-key pairs and can change model behavior; a fused exact kernel changes computation order, not the intended dense result. | Design tests for lost long-range evidence. |
| What does GQA optimize? | It reduces the number of stored KV heads, primarily lowering KV-cache memory. | Debug a decode memory limit after applying GQA. |
| How would you benchmark an attention change? | Match shapes, masks, dtype, hardware, and warm-up; separate prefill/decode latency, memory, and quality. | Identify a fallback from profiler evidence. |

## Related Concepts

- [KV Cache](23-kv-cache.md) covers reuse of K/V states across decode steps.
- [Continuous Batching](26-continuous-batching.md) covers request scheduling and KV allocation.
- [Context Window](25-context-window.md) discusses limits and behavior at long sequence lengths.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) introduces the I/O-aware exact algorithm.

## Practice

- [Attention optimization interview practice](../interview-prep/curriculum-practice.md#attention-optimization) applies kernel, sparsity, and GQA trade-offs to a serving decision.

## References

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) analyzes parallelization and work partitioning.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) evaluates grouped-query attention and KV-head sharing.
