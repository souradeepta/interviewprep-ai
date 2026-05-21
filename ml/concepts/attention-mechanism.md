# Attention Mechanism

## TL;DR
Attention lets a model focus on the most relevant parts of an input when producing each output.
Scaled dot-product attention is the core operation in all transformers. Understanding attention
is mandatory for any LLM or agentic AI engineering role.

## Core Intuition
Translating "The bank by the river" — the word "bank" is ambiguous. Attention lets the model
look back at "river" to resolve it. Each position directly attends to all other positions,
weighted by relevance.

## How It Works

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Q (queries):** what this position is looking for
- **K (keys):** what each position offers
- **V (values):** what each position contributes if attended to
- **$\sqrt{d_k}$ scaling:** prevents dot products from growing large (softmax saturation)

**Self-attention:** Q, K, V all from the same sequence.
**Cross-attention:** Q from decoder, K/V from encoder.
**Multi-head:** h heads in parallel, each with different projections. Each head captures different relationship types.
**Causal masking:** mask future positions → each token attends only to past positions. Required for autoregressive generation.

## Key Properties / Trade-offs
- O(T²) memory and compute — quadratic scaling limits context length
- Flash Attention: O(T) memory via tiling (no change to output, just compute order)
- Permutation-invariant — positional encodings are needed to inject order

## Common Mistakes / Gotchas
- Forgetting $\sqrt{d_k}$ scaling → softmax saturation → gradients vanish
- Confusing self-attention (one sequence) and cross-attention (two sequences)
- Missing causal mask in language modeling → model peeks at future tokens

## Code Example
```python
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return w @ V

T, d = 5, 8
out = attention(np.random.randn(T, d), np.random.randn(T, d), np.random.randn(T, d))
print(out.shape)  # (5, 8)
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain scaled dot-product attention." | Compute Q·Kᵀ/√d_k, softmax → attention weights, weighted sum of V. Q=what I want, K=what I offer, V=what I contribute. |
| "Why scale by √d_k?" | Without scaling, large dot products → near one-hot softmax → vanishing gradients. |
| "What is multi-head attention?" | Run h heads in parallel with different learned Q/K/V projections. Concatenate and project. Each head captures different relationship types. |

## Related Topics
- [Transformers](transformers.md) — [Coding: Implement Attention](../../../coding/ml-coding/implement-attention.md)
- [LLM: Context Window](../../../llm/concepts/context-window.md)

## Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
