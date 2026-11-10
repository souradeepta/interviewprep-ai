# Transformers

**TL;DR:** Transformers combine self-attention and position-wise feed-forward layers with residual pathways, enabling parallel sequence training and flexible context-dependent representations.

## Prerequisites

Prerequisites: [neural networks](neural-networks.md), [attention](attention-mechanism.md), and [layer normalization](layer-normalization.md). You can outline a transformer block, compare encoder/decoder/encoder-decoder masks, explain positional information and sequence cost, and select an architecture for a task. This ML lesson covers architecture mechanics; the LLM track studies language-model objectives and serving, while this page remains the general sequence-model foundation.

## Learning Outcomes

- Explain the central assumptions and mechanism of transformers.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

An RNN carries information from one token to the next in a chain. A transformer lets each position gather information directly from other permitted positions using attention. The input sequence can therefore be processed in parallel during training, and each output representation can use a content-dependent mixture of context. Position information must be added because attention without position cues does not know whether a token appeared first or last.

A transformer is not “attention only.” Feed-forward networks transform each position's representation, residual connections provide direct pathways, normalization stabilizes block inputs, and embeddings map discrete tokens or patches into vectors. Attention and feed-forward layers serve different roles: communication across positions and nonlinear processing within positions.

## How It Works

An embedding maps input token i to vector xᵢ∈Rᵈ. Position information is added or encoded through mechanisms such as learned embeddings, sinusoidal encodings, or rotary position transforms. A pre-normalization transformer block computes u=x+Attention(LN(x)), then y=u+FFN(LN(u)); dropout may be applied during training. The feed-forward sublayer commonly expands to a wider dimension, applies a nonlinearity or gated activation, then projects back. Residual addition requires compatible dimensions; a projection is needed when width changes.

Multi-head self-attention forms Q=XWQ, K=XWK, and V=XWV, splits channels across heads, computes softmax(QKᵀ/√dₖ)V, concatenates the head outputs, and applies an output projection. Encoder attention can be bidirectional; decoder self-attention uses a causal mask to prevent future-token access. Encoder-decoder architectures also use cross-attention from decoder queries to encoder states. Padding masks ensure nonexistent tokens do not influence valid outputs.

Architecture classes reflect mask and objective choices. Encoder-only models build bidirectional representations and are natural for classification or token understanding. Decoder-only models use causal attention and next-token prediction for generation. Encoder-decoder models encode an input and generate a target sequence, such as translation. These are common patterns, not rigid task boundaries; architecture choice also depends on pretraining and data availability.

Dense attention on sequence length T costs O(T²d) arithmetic and O(T²) score storage in a straightforward implementation; memory-efficient kernels reduce intermediate storage but not the number of dense token pairs. Feed-forward layers cost roughly O(Td·d_ff). Autoregressive inference generates tokens sequentially even though prompt prefill processes positions in parallel; caching keys and values avoids recomputing old projections but uses memory proportional to layers, sequence length, batch size, and head dimensions. These are systems consequences of the architecture.

Transformers are permutation-equivariant without position encodings. Positional mechanisms have different extrapolation and compatibility behavior, so changing them in a pretrained model is not a neutral modification. Pre-LN often improves optimization in deep stacks by keeping residual pathways accessible; normalization type, epsilon, and order must match checkpoint structure. Initialization and learning-rate warmup matter, especially for training large models from scratch.

Attention masks are best understood as part of the model's information-flow contract. A causal language model may attend to its own token and all earlier tokens, but never later ones. A padding mask removes positions that do not correspond to input. Cross-attention may use a source padding mask while the decoder also applies a causal mask to its own history. Implementations combine these masks differently, including boolean conventions where “true” can mean either allowed or blocked; inspect the API and test a tiny example rather than relying on the mask's name. A single sign error can create target leakage while leaving tensor shapes and training loss apparently valid.

The feed-forward sublayer often contributes a large share of parameters and computation, despite attention receiving more conceptual focus. It applies the same MLP independently at each position, so its operation count scales linearly with sequence length. Attention communicates across positions; the MLP transforms each position's features. This division explains why optimizing only the attention kernel may not yield the expected total speedup. Profile the whole block and account for projections, normalization, residual additions, and data movement.

Inference has a different execution pattern from training. During prompt prefill, the model computes representations for all prompt tokens; during autoregressive decode, it repeatedly computes the next token. Key/value caching avoids recomputing previous keys and values, but the cache grows with context length and batch size. Training's dense attention memory and inference's cache memory are distinct bottlenecks. Serving decisions such as batch size, maximum context, and cache precision therefore affect the feasible architecture and must be included in design evaluation.

## Trade-offs and Selection Boundary

| Architecture | Useful for | Boundary |
|---|---|---|
| Encoder-only | Bidirectional context for classification/representation | Not naturally autoregressive without architectural changes |
| Decoder-only | Causal next-token generation and prompt continuation | Context attention and KV-cache costs grow with sequence length |
| Encoder-decoder | Conditional sequence-to-sequence tasks | More components and cross-attention state |
| Sparse/local attention | Longer contexts with structured interactions | Pattern may omit important distant dependencies |
| RNN/temporal CNN | Streaming or local sequence structure | Different parallelism and dependency trade-offs |

Use dense transformers when global interactions and data scale justify their compute. For long sequences, compare sparse attention, hierarchical processing, recurrence, and retrieval against task-specific quality and memory constraints. Choose encoder/decoder form from whether the task needs bidirectional input representation, causal generation, or conditional generation.

## Implementation

```python
import math

def self_attention(x, causal=False):
    n = len(x); d = len(x[0]); out=[]
    for i, qi in enumerate(x):
        scores = [sum(a*b for a,b in zip(qi,k))/math.sqrt(d) for k in x]
        valid = range(i+1) if causal else range(n)
        peak = max(scores[j] for j in valid)
        exp = [math.exp(scores[j]-peak) for j in valid]
        total = sum(exp)
        out.append([sum(exp[t]/total*x[j][col] for t,j in enumerate(valid)) for col in range(d)])
    return out

x = [[1.,0.], [0.,1.], [1.,1.]]
causal = self_attention(x, causal=True)
assert causal[0] == x[0]
changed_suffix = self_attention([x[0], x[1], [9.,9.]], causal=True)
assert causal[0] == changed_suffix[0]  # future position cannot affect earlier output
```

## Failure Modes and Diagnostics

1. **Causal mask missing or reversed:** training leaks future tokens. Change suffix tokens and assert earlier causal outputs remain unchanged.
2. **Padding included in attention or loss:** outputs depend on batch padding length. Test identical sequence with different padding and audit masks.
3. **Context memory unexpectedly high:** score or KV-cache growth exceeds estimates. Calculate per-layer cache bytes and use realistic batch/context lengths in load tests.
4. **Position encoding mismatch after fine-tuning:** model quality degrades at longer or all lengths. Preserve checkpoint configuration and evaluate by relative position/context length.
5. **Pre-LN/Post-LN or normalization mismatch:** loaded weights produce unstable activations. Verify model config and tensor shapes against the exact checkpoint architecture.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What are the main parts of a transformer block? | Normalization, attention, residual addition, feed-forward network, and another residual path. |
| Encoder versus decoder attention? | Encoder attention can see both directions; decoder self-attention masks future positions for causal generation. |
| Design follow-up: long-context serving runs out of memory. | Estimate KV-cache and attention costs, profile prefill/decode separately, then evaluate cache sharing, batching, or sparse/retrieval alternatives. |

## Related Concepts

- [Attention mechanism](attention-mechanism.md), [layer normalization](layer-normalization.md), and [RNNs/LSTMs](rnns-lstms.md)
- [Matching notebook](../notebooks/38-transformers.ipynb)
- [Interview practice: transformer attention](../interview-prep/ml-coding-questions.md#q2-implement-scaled-dot-product-attention-from-scratch)

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Stanford CS25, [Transformers United](https://web.stanford.edu/class/cs25/).
