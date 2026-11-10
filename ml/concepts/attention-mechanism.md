# Attention Mechanism

**TL;DR:** Attention builds a context-dependent weighted combination of value vectors, with weights determined by query-key compatibility.

## Prerequisites

Prerequisites: matrix multiplication, softmax, and [neural networks](neural-networks.md). You can derive scaled dot-product attention, distinguish causal and padding masks, explain quadratic sequence cost, and diagnose masking errors. This ML note treats attention as a trainable operation and scaling decision; [transformers](transformers.md) compose it into a full architecture, while the AI overview supplies domain context.

## Learning Outcomes

- Explain the central assumptions and mechanism of attention mechanism.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

When interpreting “bank” in “the boat passed the bank,” the surrounding words carry useful context. Attention assigns each query a distribution over keys and combines their values accordingly. Unlike a fixed-size recurrent state, each position can directly retrieve information from any permitted position. Query means what a position seeks, key means what it advertises, and value means what it contributes.

## How It Works

Given Q∈Rᵀˣᵈᵏ, K∈Rˢˣᵈᵏ, and V∈Rˢˣᵈᵛ, compute A=softmax(QKᵀ/√dₖ + M), then output AV. The √dₖ scale keeps dot-product magnitudes from growing with dimension and saturating softmax. A boolean mask M assigns disallowed scores a large negative value (conceptually −∞) before softmax. Self-attention derives all three projections from one sequence; cross-attention uses queries from one sequence and key/value vectors from another. Multi-head attention repeats this operation in learned subspaces, concatenates outputs, and projects them.

Dense attention costs O(TS·dₖ) compute and O(TS) score memory; self-attention with T=S is quadratic. Flash-style kernels reduce memory traffic and avoid materializing the whole score matrix but do not change dense attention's fundamental pairwise arithmetic. Position information must enter through embeddings, rotations, or another position-aware mechanism because the base operation alone is permutation equivariant.

The weight matrix is a normalized distribution over allowed keys for each query. It is often tempting to interpret a large attention weight as a causal explanation of a model's decision, but this is not generally justified: value vectors, output projections, residual pathways, and later layers also affect the result. Attention weights describe one intermediate computation under a particular parameterization. To explain a prediction, combine them with interventions, ablations, attribution methods, or carefully designed counterfactuals rather than treating the displayed matrix as a complete explanation.

Masking rules deserve careful shape and numerical reasoning. For batched inputs, scores typically have shape [batch, heads, query_length, key_length], while a causal mask may be [query_length, key_length] and broadcast across batch and heads. A padding mask may have a different broadcast shape. A mask that accidentally broadcasts over the query axis rather than key axis can produce valid tensor shapes but incorrect information flow. A large finite negative sentinel approximates negative infinity, but its safe magnitude depends on dtype; some low-precision formats overflow when converting an arbitrarily large negative value. Prefer library mask conventions and test exact edge positions.

Attention's pairwise computation can also be wasteful when the task structure is known. Local attention only relates nearby positions, block-sparse patterns connect selected regions, and cross-attention can restrict a decoder to a separate source. These patterns reduce work only when the implementation uses kernels that skip masked pairs; simply adding a dense mask may still calculate every score. Sequence length, head count, head dimension, and batch size all affect memory. Estimate these costs before allocating a large context and distinguish algorithmic sparsity from a dense implementation that merely zeros outputs.

In autoregressive inference, keys and values for past positions are reused rather than recomputed. This key/value cache avoids repeating projection work at each generated token but grows linearly with sequence length, number of layers, batch size, and the aggregate key/value width. Cache dtype, paging, prefix reuse, and eviction are serving decisions, not properties of the attention equation alone. When designing an attention module, ask whether the intended use is parallel training, causal decoding, or cross-attention retrieval because the optimal mask and memory strategy differ.

## Trade-offs and Selection Boundary

| Design | Strength | Boundary |
|---|---|---|
| Dense self-attention | Direct interactions between every pair | Quadratic cost limits long sequences |
| Causal attention | Supports next-token generation without future leakage | Requires correct triangular mask and KV caching for efficient decoding |
| Cross-attention | Lets decoder retrieve from a separate source | Adds source projections and memory |
| Local/sparse attention | Reduces interactions and memory | Can miss distant dependencies; pattern is a modeling assumption |

Use dense attention when sequence lengths are moderate and global interactions matter. For long sequences, select sparse, chunked, or recurrent alternatives based on task structure and validate quality as well as cost.

## Implementation

```python
import math

def attention(q, k, v, allowed=None):
    scores = [[sum(a*b for a,b in zip(qi,kj))/math.sqrt(len(qi)) for kj in k] for qi in q]
    outputs = []
    for i, row in enumerate(scores):
        valid = [j for j in range(len(row)) if allowed is None or allowed[i][j]]
        peak = max(row[j] for j in valid)
        weights = [math.exp(row[j]-peak) for j in valid]
        total = sum(weights)
        outputs.append([sum(weights[t]/total*v[j][d] for t,j in enumerate(valid)) for d in range(len(v[0]))])
    return outputs

q = [[1., 0.], [0., 1.]]; k = q; v = [[10., 0.], [0., 20.]]
causal = attention(q, k, v, [[True, False], [True, True]])
assert causal[0] == [10., 0.] and causal[1][1] > 0
```

## Failure Modes and Diagnostics

1. **Future-token leakage:** causal mask is reversed or omitted. Verify each output at position i is invariant to changes at positions greater than i.
2. **All keys masked:** softmax over only negative infinity yields NaNs. Ensure every query has a valid key, often a self or sentinel token.
3. **Missing scale:** attention becomes too peaky at larger dimensions. Inspect score and entropy distributions; include 1/√dₖ.
4. **Quadratic memory surprise:** a long prompt causes OOM despite modest parameter count. Estimate T² score storage and use memory-efficient kernels or a suitable sparse pattern.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why divide by √dₖ? | Dot-product variance grows with dimension; scaling helps avoid saturated softmax and weak gradients. |
| Self-attention versus cross-attention? | Self uses one sequence for Q/K/V; cross uses queries from one sequence and K/V from another. |
| Debug follow-up: generation depends on future tokens during training. | Inspect causal-mask orientation and assert prefix outputs are unchanged when suffix tokens change. |

## Related Concepts

- [Transformers](transformers.md), [RNNs/LSTMs](rnns-lstms.md), and [layer normalization](layer-normalization.md)
- [Matching notebook](../notebooks/02-attention-mechanism.ipynb)
- [Interview practice: scaled dot-product attention](../interview-prep/ml-coding-questions.md#q2-implement-scaled-dot-product-attention-from-scratch)

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135).
