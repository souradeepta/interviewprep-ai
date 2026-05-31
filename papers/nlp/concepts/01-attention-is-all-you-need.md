---
title: "Attention Is All You Need"
authors: "Vaswani, Shazeer, Parmar, et al."
year: 2017
venue: "NeurIPS"
doi: "https://doi.org/10.5555/3294996"
arxiv: "https://arxiv.org/abs/1706.03762"
domain: "nlp"
difficulty: "intermediate"
interview_frequency: "very_high"
related_concepts:
  - llm/concepts/01-transformers
  - llm/concepts/02-self-attention
---

# Attention Is All You Need

## Paper Overview

**Title:** Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Parmar Aravind, et al. (Google Brain & University of Toronto)

**Published:** NeurIPS 2017 | [arXiv](https://arxiv.org/abs/1706.03762)

**Citation:** 80,000+ (one of the most influential papers in AI)

Before 2017, sequence-to-sequence models relied on RNNs (Recurrent Neural Networks) and LSTMs for translation, summarization, and other NLP tasks. RNNs process sequences sequentially—position 1, then 2, then 3—which makes them slow to train and difficult to parallelize. They also suffer from vanishing gradients over long sequences. This paper introduced the **Transformer architecture**, which replaced recurrence entirely with **self-attention mechanisms** that can process entire sequences in parallel. The paper's famous title captures the key insight: you don't need recurrence; attention mechanisms alone are sufficient. This single paper fundamentally changed how all modern NLP systems work: BERT, GPT, T5, and every large language model today are based on Transformer architecture.

**Why this matters for interviews:** The Transformer is the foundation of modern NLP and AI. Every interviewer assumes you understand self-attention, multi-head attention, and positional encoding. You need to explain not just what these components are, but *why* they work and how to implement them. This paper is essential for any NLP, LLM, or AI systems role.

---

## Core Contribution

### The Problem: RNN Bottlenecks

RNNs process sequences sequentially:
- Position 1 processes input[0] → hidden[0]
- Position 2 processes input[1] + hidden[0] → hidden[1]
- Position N processes input[N-1] + hidden[N-1] → hidden[N]

**Consequences:**
- Cannot parallelize (must wait for position i-1 to compute position i)
- Slow training on long sequences (100+ tokens)
- Vanishing gradients over 50+ steps
- Cannot attend efficiently to distant positions

### The Solution: Self-Attention Mechanism

Instead of recurrence, each position attends to *all* other positions in one operation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Why this works:**
- Parallelizable: all positions computed simultaneously
- Efficient attention to distant positions: query-key similarity captures relevance
- Strong gradient flow: attention weights are differentiable end-to-end
- Interpretable: attention weights show which positions matter

### Key Innovation: Multi-Head Attention

Run multiple attention mechanisms in parallel, then combine:

$$\text{MultiHead}(Q, K, V) = \text{concat}(h_1, h_2, \ldots, h_h)W^O$$

where each head uses different learned projections of Q, K, V.

**Why multiple heads:**
- Different heads learn different relationships (syntax, semantics, long-range dependencies)
- Ensemble effect: more robust than single attention
- Parallel computation: h heads run in parallel

---

## Key Ideas & Algorithm

### How Transformer Self-Attention Works

**Step 1: Linear Projections**
- Input sequence X ∈ ℝ^(N×d_model)
- Project to Query, Key, Value:
  - Q = XW^Q ∈ ℝ^(N×d_k)
  - K = XW^K ∈ ℝ^(N×d_k)
  - V = XW^V ∈ ℝ^(N×d_v)

**Step 2: Compute Attention Weights**
- Similarity: QK^T ∈ ℝ^(N×N)
- Scale by √d_k: (QK^T)/√d_k (prevents saturation)
- Softmax: attention weights ∈ ℝ^(N×N)

**Step 3: Apply to Values**
- Output = Attention × V ∈ ℝ^(N×d_v)

**Step 4: Multi-Head Combination**
- Run h heads in parallel
- Concatenate outputs: ℝ^(N×hd_v)
- Final projection: W^O → ℝ^(N×d_model)

### Positional Encoding

Transformers have no recurrence, so they need to know position. Use sinusoidal positional encodings:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**Why sinusoidal?**
- Constant offset applies to all positions (model can learn relative positions)
- Periodic: captures both short-range and long-range position relationships
- Generalizes to longer sequences than training data

### Transformer Block Architecture

```
Input X
  ↓
MultiHeadAttention(X, X, X)  ← Self-attention
  ↓
Add & Norm (residual connection)
  ↓
FeedForward (2 linear layers with ReLU)
  ↓
Add & Norm (residual connection)
  ↓
Output
```

Stack 6 of these blocks in encoder, 6 in decoder.

### Encoder-Decoder Structure

**Encoder:** Self-attend over input sequence
```
Input → MultiHead(Self) → FFN → ... (6 blocks) → Context
```

**Decoder:** Attend to input AND generate output
```
Output → MultiHead(Self) → MultiHead(Encoder) → FFN → ... (6 blocks) → Logits
```

**Masking in decoder:** Prevent attending to future positions (causal attention)

---

## Architecture & Trade-offs

### Attention Mechanism Variants

| Variant | Query Attends To | Use Case |
|---------|------------------|----------|
| Self-Attention | All positions in same sequence | BERT, GPT (encoder/decoder) |
| Cross-Attention | All positions in *different* sequence | Seq2Seq decoder attends to encoder |
| Causal Attention | All positions ≤ current | Language modeling (autoregressive) |
| Sparse Attention | Subset of positions | Long sequences (efficient) |

### Multi-Head vs Single-Head

| Aspect | Single-Head | Multi-Head (h=8) |
|--------|-------------|-----------------|
| Capacity | d_k = d_model | d_k = d_model / h |
| Interpretability | One attention pattern | h different patterns |
| Computation | 1 × (N² operations) | h × (N² / h²) = same total |
| Robustness | Sensitive to initialization | Ensemble effect |

### Design Choices

**1. Why √d_k scaling?**
- Without scaling: softmax becomes sharp (gradient → 0)
- With scaling: softmax stays smooth → strong gradients
- Factor: √d_k is optimal (empirically; comes from variance analysis)

**2. Why 6 encoder/decoder blocks?**
- More blocks = deeper models = more capacity
- Trade-off: more blocks = slower training
- 6-12 blocks standard; modern models use 24-96 blocks

**3. Why position-wise FFN?**
- Adds nonlinearity (ReLU)
- Independent at each position (parallelizable)
- Hidden size 4× d_model: 512 → 2048 → 512
- Alternative names: feedforward network, MLP, dense layers

**4. How many attention heads?**
- h=8: d_k = 512/8 = 64 (balanced)
- h=1: single head, less expressive
- h=16: finer attention, more parameters
- Empirically h=8-12 works well

---

## Interview Q&A

**Q: Explain self-attention. Why is it better than RNNs?**

A: Self-attention computes a weighted combination of all positions in the sequence, where weights are learned from query-key similarity. Each position attends to all others based on relevance. This is better than RNNs because: (1) parallelizable—all positions computed simultaneously vs. RNN's sequential processing, (2) efficient long-range dependencies—attention score depends on query-key similarity regardless of distance, (3) stronger gradients—each position has direct paths to all others for backprop. RNNs bottleneck at sequential processing and suffer vanishing gradients over 50+ steps. Answer with: "Self-attention is O(N²) but parallelizable; RNN is O(N) but sequential—Transformer wins on modern hardware."

**Q: Why scale attention scores by √d_k?**

A: Without scaling, softmax on large d_k produces very small gradients (nearly 0). With scaling by √d_k, the dot product QK^T has variance ≈1, so softmax stays in a reasonable range (gradient ~0.1-0.3). This is essential for training stability. Mathematically: if Q, K ~ N(0, 1) and dimension d_k, then QK^T ~ N(0, d_k), so variance is d_k. Scaling by √d_k normalizes this to variance ≈1. Without this, you get "attention collapse" where softmax is nearly uniform or one-hot, causing optimization issues.

**Q: What's the purpose of multiple attention heads?**

A: Different heads learn different relationships: one head might attend to nearby words (syntax), another to distant semantics, another to entity references. This is an ensemble effect—multiple diverse representations are more robust than a single attention pattern. Mathematically, you're decomposing the d_model dimensions into h independent subspaces, each with their own query-key-value projections. This increases model capacity without increasing computation (h heads at d_k = d_model/h is same FLOPs as 1 head at d_model). In practice, heads learn complementary patterns; you can verify this by inspecting attention weight matrices in visualizations.

**Q: Why do you need positional encoding? Can't the model learn position from attention?**

A: Without positional encoding, the model has no information about token order. Attention is permutation-invariant: if you shuffle input tokens, attention weights change but structure remains similar. With sinusoidal encodings, each position has a unique encoding, and the model can learn to extract relative/absolute position from this signal. Sinusoidal is chosen because: (1) generalizes to longer sequences than training data (Fourier basis), (2) relative position differences are constant offsets (easier to learn), (3) works without pre-training (unlike learned embeddings). Modern variants (RoPE, ALiBi) use different positional encodings, but sinusoidal is the original and still used in many models.

**Q: What's the computational complexity of self-attention?**

A: O(N² d_k) where N is sequence length, d_k is key dimension. For each of N queries, you compute dot product with N keys (O(N d_k)), then multiply by N values (O(N d_k)). Total: O(N² d_k). This is quadratic in sequence length—problematic for long documents (8K+ tokens). Sparse attention variants (local, strided, reformer) reduce this to O(N log N) or O(N). In practice, for 512-token sequences (GPT-2), O(N²) is acceptable. For 4K+ tokens (GPT-3, BERT), efficiency matters. Answer: "Self-attention is O(N²), which is why context windows are typically 2-32K, not unlimited."

**Q: How does causal masking work in language modeling?**

A: In autoregressive models (GPT), you generate one token at a time, so you can't attend to future tokens (they don't exist yet). Causal masking sets attention weights to 0 (via -∞ in softmax) for positions j > i when computing position i's attention. This forces the model to predict token i based only on tokens before it (i-1, i-2, ..., 0). Mathematically: Attention(Q, K, V)_ij = softmax(QK^T / √d_k)_ij × V, where softmax sets j > i to 0. This prevents information leakage and ensures the model can be used autoregressively at test time. BERT doesn't use causal masking because it's bidirectional (no autoregressive generation).

**Q: When would you NOT use Transformers?**

A: Transformers aren't optimal for: (1) very short sequences (1-10 tokens)—simpler models suffice, (2) extremely long sequences (100K+ tokens without sparse attention)—O(N²) becomes prohibitive, (3) tasks where strict sequential processing is essential (though rare), (4) very low-latency inference—Transformer requires attending to all input tokens; RNNs can output incrementally. Mostly, Transformers have won across the board. For long sequences, use: sparse attention (Longformer, BigBird), hierarchical (Transformer-XL), or specialized architectures (Mamba, state-space models). But for standard NLP (translation, QA, summarization, language modeling), Transformers are default.

---

## Best Practices

- **Layer normalization:** Apply LayerNorm before (pre-norm) or after (post-norm) attention/FFN. Pre-norm is more stable for deep networks; use it for models >12 layers.

- **Dropout:** Add dropout (0.1-0.3) on attention weights and FFN outputs to regularize. Also apply DropOut to positional encodings and token embeddings.

- **Initialization:** Initialize weights carefully. Query/Key/Value projections: Xavier uniform. Output projection of MultiHeadAttention: small scale (scale by 1/√(d_model)). Positional encodings: no learning needed; just compute.

- **Learning rate:** Use lower LR for Transformers than CNNs (0.001-0.0001 typical, vs. 0.1 for CNNs). Use learning rate warmup (linear increase over 10K steps) for stability.

- **Gradient clipping:** Clip gradients to max norm (1.0-5.0) to prevent explosion in early training.

- **Batch size & accumulation:** Use large batch sizes (256-512) for stable training. If GPU memory is limited, use gradient accumulation to simulate larger batches.

- **Attention head count:** Use h=8-12 for most tasks. Fewer heads (h=1-4) for small models (<100M parameters). More heads (h=16+) for large models only if you have the compute.

---

## Common Pitfalls

- **Mistake: Computing self-attention without scaling by √d_k.** Attention becomes too sharp (softmax ≈ one-hot), gradients vanish, training fails. Always include the 1/√d_k factor. Impact: loss plateaus, model stops learning.
  → Fix: Add scaling factor: attention = softmax((QK^T) / sqrt(d_k)) @ V

- **Mistake: Forgetting positional encodings.** Model ignores word order; "I love dogs" and "Dogs love I" look identical. Performance on any sequence task drops dramatically. Impact: accuracy drops 10-30%.
  → Fix: Add positional encodings to input embeddings: token_embed + position_embed

- **Mistake: Causal masking in bidirectional models (BERT, encoders).** If you mask future positions in BERT, it can't attend to both sides—defeats the purpose of bidirectional context. Only use causal masking in autoregressive decoders (GPT). Impact: breaks model's ability to use bidirectional context.
  → Fix: Don't mask in encoder. Only mask in decoder (causal attention during decoding).

- **Mistake: Very large learning rates with Transformers.** Transformers are sensitive to LR; too-high LR causes exploding gradients even after clipping, or divergence in first epoch. Impact: training diverges or loss oscillates wildly.
  → Fix: Use 10-100× smaller LR than you'd use for CNNs. Use learning rate warmup.

- **Mistake: Using too many attention heads for small models.** If d_model = 256 and h = 16, each head gets d_k = 16 dimensions—too small, loses expressiveness. Better to use fewer heads (h=4) or larger d_model. Impact: worse performance with more parameters.
  → Fix: Ensure d_k ≥ 32-64 per head. If model is small, use fewer heads.

---

## Code Examples

### Example 1: Scaled Dot-Product Attention from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V"""
    
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len, d_k)
            key:   (batch, seq_len, d_k)
            value: (batch, seq_len, d_v)
            mask:  Optional mask for causal attention
        
        Returns:
            output: (batch, seq_len, d_v)
            attention_weights: (batch, seq_len, seq_len) - for visualization
        """
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask (e.g., causal for autoregressive models)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention to values
        output = torch.matmul(weights, value)
        
        return output, weights


# Test on synthetic data
batch_size = 2
seq_len = 4
d_k = 64
d_v = 64

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

attention = ScaledDotProductAttention(d_k)
output, weights = attention(Q, K, V)

print(f"Output shape: {output.shape}")  # (2, 4, 64)
print(f"Attention weights shape: {weights.shape}")  # (2, 4, 4)
print(f"Attention weights sum to 1 per query: {weights.sum(dim=-1)}")  # ~1.0
```

### Example 2: Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention: run h attention heads in parallel"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V (shared across all heads)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection after concatenating heads
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len, d_model)
            key:   (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Project and reshape for multi-head: (batch, seq_len, num_heads, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Now: (batch, num_heads, seq_len, d_k)
        
        # Apply attention to each head
        attn_output, _ = self.attention(Q, K, V, mask)
        
        # Concatenate heads: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.W_o(attn_output)
        
        return output


# Test
d_model = 512
num_heads = 8
seq_len = 10

mha = MultiHeadAttention(d_model, num_heads)
X = torch.randn(batch_size, seq_len, d_model)
output = mha(X, X, X)
print(f"MultiHeadAttention output shape: {output.shape}")  # (batch, seq_len, d_model)
```

### Example 3: Transformer Block with Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1)].unsqueeze(0)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block: MultiHeadAttention + FFN with residual connections"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.mha(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """Full Transformer encoder: embeddings → positional encoding → stack of blocks"""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # x: (batch, seq_len) - token indices
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x  # (batch, seq_len, d_model)


# Example: sequence classification
vocab_size = 1000
d_model = 256
num_heads = 4
d_ff = 1024
num_layers = 2

encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)

# Dummy input: batch_size=2, seq_len=10
input_ids = torch.randint(0, vocab_size, (batch_size, 10))
encoder_output = encoder(input_ids)
print(f"Encoder output: {encoder_output.shape}")  # (2, 10, 256)

# Classification head
classifier = nn.Sequential(
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(d_model, 2)  # 2 classes
)
logits = classifier(encoder_output.transpose(1, 2))
print(f"Classification logits: {logits.shape}")  # (2, 2)
```

---

## Related Concepts

- [BERT: Pre-training of Deep Bidirectional Transformers](./02-bert.md) — Bidirectional transformer pre-training
- [Vision Transformer](../vision/concepts/02-vision-transformer.md) — Applying Transformers to images
- [Scaling Laws for Neural Language Models](./04-scaling-laws.md) — How Transformer scale affects performance
- [LoRA: Low-Rank Adaptation](./05-lora.md) — Efficient fine-tuning of Transformers

