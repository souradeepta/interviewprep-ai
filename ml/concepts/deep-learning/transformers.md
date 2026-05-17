# Transformers

## TL;DR
The transformer architecture (Vaswani et al., 2017) replaces recurrence with self-attention.
Foundation of all modern LLMs: GPT, BERT, T5, Llama, Claude. Understanding transformers
mechanistically is required for any LLM engineering role.

## Core Intuition
RNNs process tokens sequentially — slow and forgetful over long sequences. Transformers process
all tokens at once via self-attention, letting every token directly attend to every other.
This parallelism enables training on massive datasets and predictable scaling.

## How It Works

**Transformer block:**
```
x → LayerNorm → MultiHeadSelfAttention → Residual add
  → LayerNorm → FFN(GELU) → Residual add → x_out
```
FFN: two linear layers with nonlinearity, width = 4× model dimension.

**Architectures:**
- **Encoder-only (BERT):** bidirectional attention, masked language modeling. Used for classification, embeddings.
- **Decoder-only (GPT/Llama/Claude):** causal attention, next-token prediction. Used for generation.
- **Encoder-Decoder (T5/BART):** for seq2seq tasks (translation, summarization).

**Positional Encoding:**
- Sinusoidal (original paper): $PE(pos, 2i) = \sin(pos/10000^{2i/d})$
- Learned embeddings (GPT-2)
- RoPE (Llama): relative position baked into Q/K rotation — extends to longer contexts

**Scaling laws:** performance scales predictably with params, data, compute.
Chinchilla: ~20 tokens per parameter for compute-optimal training.

## Key Properties / Trade-offs
- O(T²) attention — context length limited by memory
- Pre-LN (LayerNorm before attention) more stable than Post-LN
- Residual connections are critical — not optional

## Common Mistakes / Gotchas
- BERT ≠ GPT: BERT is encoder (bidirectional), GPT is decoder (causal)
- FFN accounts for ~⅔ of parameters, not attention — often forgotten
- Confusing d_model and d_k = d_model/h_heads

## Code Example
```python
import torch, torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d=512, h=8, ff=2048, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0])
        x = x + self.drop(self.ff(self.ln2(x)))
        return x
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain the transformer." | N blocks of: Pre-LN → multi-head self-attention → residual → Pre-LN → FFN → residual. Input needs positional encoding since self-attention is permutation-invariant. |
| "BERT vs GPT?" | BERT: encoder-only, bidirectional, MLM pretraining. For classification/embeddings. GPT: decoder-only, causal, next-token prediction. For generation. |
| "Why do transformers scale better than RNNs?" | Full parallelism → efficient GPU use. Short gradient paths (no sequential bottleneck). Predictable scaling with data + compute. |

## Related Topics
- [Attention Mechanism](attention-mechanism.md) — [LLM: Pretraining](../../../llm/concepts/pretraining.md)
- [Coding: Implement Transformer](../../../coding/ml-coding/implement-transformer.md)

## Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [GPT-3 paper](https://arxiv.org/abs/2005.14165)
