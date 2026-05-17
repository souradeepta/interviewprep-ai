# Attention Optimization

## TL;DR
Optimize attention (O(T²) bottleneck): Flash Attention (I/O aware, 2-4x faster, same quality), sparse attention (attend only to relevant tokens, 2-4x faster, <1% quality loss), grouped-query attention (share KV across Q heads, 50% KV cache reduction). Choose based on hardware and latency requirements.

## Core Intuition
Standard attention computes QK^T (a T×T matrix), which requires O(T²) memory and transfers tons of data between slow HBM (GPU RAM) and fast SRAM. This is the LLM inference bottleneck. Flash Attention reorganizes computation to keep data on-chip. Sparse attention realizes you don't need T² comparisons—local + strided patterns suffice. GQA reuses KV across multiple Q heads to reduce cache size.

## How It Works

**Standard (Naive) Attention:**
```
Input: Q (T×D), K (T×D), V (T×D)
Memory: HBM (slow, high bandwidth but high latency)

Algorithm:
1. Load Q, K from HBM → compute QK^T (T×T) → write to HBM  [memory I/O]
2. Load QK^T → softmax → write to HBM                       [memory I/O]
3. Load softmax(QK^T), V → multiply → write output to HBM   [memory I/O]

Memory transactions: O(T²) with large matrices
Compute: O(T² × D)
Bottleneck: memory I/O dominates compute time
```

**Flash Attention (I/O-Aware):**
```
Key insight: reorganize to maximize on-chip computation

Algorithm (tiling):
1. Divide Q, K, V into blocks (e.g., 64×64)
2. For each block of Q:
   a. Load small block Q_i into SRAM (fast)
   b. Iterate over blocks of K, V:
      - Load K_j, V_j into SRAM
      - Compute attention(Q_i, K_j, V_j) on-chip
      - Accumulate results
      - No writing intermediate QK^T to HBM
3. Write final output once

Result:
- Reduced HBM I/O: from O(T² × D) to O(T × D)
- Speedup: 2-4x (less memory transfer)
- Quality: identical (same computation, just reorganized)

Hardware requirement: GPU with large SRAM (A100, H100, L40)
Compatibility: not portable to all GPUs
```

**Sparse Attention Patterns:**

Standard: O(T²) full attention
```
Token attends to: all previous tokens
Example (T=10):
  t0: [x]
  t1: [x x]
  t2: [x x x]
  t3: [x x x x]
  ...
  t10: [x x x x ... x] (10 comparisons)
```

Local attention: O(T × w) where w = window
```
Token attends to: last w tokens only
Example (w=4):
  t10: [x x x x]  (4 comparisons instead of 10)
```

Strided: O(T × √T) per layer
```
Token attends to: local window + periodic stride
Example (w=4, stride=8):
  t10: [x x x x] + [x . . . . . . x]  (8 comparisons)
```

Local + strided: O(T × √T) hierarchical
```
Each layer: some layers use local (cheap), some use strided (context)
Trade: <1% quality loss, 2-4x speedup
```

**Grouped-Query Attention (GQA):**

Standard Multi-Head Attention:
```
Q: num_heads=32, head_dim=64
K: num_heads=32, head_dim=64
V: num_heads=32, head_dim=64

Each head has independent K, V.
KV cache size per sequence: seq_len × 32 × 64 × 2 (K and V)
```

GQA (Group size = 8):
```
Q: num_heads=32, head_dim=64
K: num_heads=4, head_dim=64    (32 / 8 = 4)
V: num_heads=4, head_dim=64

Multiple Q heads share one K, V head.
head_0-7 → KV_head_0
head_8-15 → KV_head_1
head_16-23 → KV_head_2
head_24-31 → KV_head_3

KV cache size: seq_len × 4 × 64 × 2 = 8x smaller!
```

## Key Properties / Trade-offs

| Method | Speedup | Memory | Quality | Latency | Deploy |
|--------|---------|--------|---------|---------|--------|
| Standard | 1x | 1x | 100% | Baseline | Any GPU |
| Flash Attn | 2-4x | Same | 100% | -50-75% | A100/H100+ |
| Sparse (local) | 2-4x | 50% | 99% | -50-75% | Any GPU |
| Sparse (strided) | 3-8x | 25% | 98% | -70-85% | Any GPU |
| GQA (no sparse) | 1.2-1.5x | 50% (KV only) | 99% | -10% | Any GPU |
| Flash + Sparse | 4-8x | 50% | 98% | -75-90% | A100/H100+ |

**Practical Combinations:**

- **High latency sensitivity (chat):** Flash Attention + GQA
- **Long context (RAG docs):** Sparse (local+strided) + GQA
- **Cost-sensitive:** Sparse local attention only
- **Research/offline:** Standard (not optimized, accurate baseline)

## Common Mistakes / Gotchas

- **Hardware assumptions:** Flash Attention requires A100/H100. Deploy on A10/L4 → not available (falls back to standard). Always check hardware support.

- **Sparse pattern selection:** Strided attention with stride=1 effectively full attention (defeats purpose). Choose stride carefully. Local-only for short contexts, hybrid for long.

- **Combining incompatible methods:** Some sparse patterns don't work with certain KV cache implementations. Check library docs.

- **Quality degradation at scale:** 1% loss per layer × 30 layers = ~30% total loss. Unlikely but monitor on large models.

- **Not using tied implementations:** custom sparse attention → slower than optimized library. Use Flash Attention from FlashInfer, vLLM, or transformers.

- **Forgetting to update positions:** if using positional encodings with sparse attention, ensure positions still reflect actual token positions (not local positions).

- **Inference-training mismatch:** train with Flash Attention, deploy with standard → slight numerical differences. Use same method for both.

## Code Example

```python
from transformers import AutoModel
from flash_attn import flash_attn_2_cuda

# Flash Attention in transformers
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b",
    attn_implementation="flash_attention_2"  # Automatic Flash Attention
)

# GQA (Llama-2 has GQA built-in)
# Check if model uses GQA:
if hasattr(model.config, 'num_key_value_heads'):
    gqa_ratio = model.config.num_attention_heads / model.config.num_key_value_heads
    print(f"GQA ratio: {gqa_ratio}")  # Example: 32 / 4 = 8

# Sparse attention (with deepspeed or custom)
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b")
config.attention_config = {
    "attention_type": "sparse",
    "sparse_block_size": 64,
    "sparse_stride": 8,  # stride attention
}
model = AutoModelForCausalLM.from_config(config)

# Manual Flash Attention usage
import torch
from flash_attn import flash_attn_func

batch_size, seq_len, dim = 2, 1024, 768
q = torch.randn(batch_size, seq_len, dim)
k = torch.randn(batch_size, seq_len, dim)
v = torch.randn(batch_size, seq_len, dim)

# Standard attention (slow)
scores = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)  # T×T matrix
attn_weights = torch.softmax(scores, dim=-1)
output_standard = torch.matmul(attn_weights, v)

# Flash Attention (fast)
q_grouped = q.view(batch_size, -1, seq_len // 64, 64)  # reshape for tiling
k_grouped = k.view(batch_size, -1, seq_len // 64, 64)
v_grouped = v.view(batch_size, -1, seq_len // 64, 64)

output_flash = flash_attn_func(q, k, v)  # Equivalent, faster

# Verify same result
print(torch.allclose(output_standard, output_flash, atol=1e-4))  # True
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Optimize attention?" | Flash Attention (2-4x), sparse (2-8x, <1% loss), GQA (KV cache 50%). Choose by hardware/latency. |
| "Flash Attention?" | Reorganize computation to keep Q,K,V blocks on-chip SRAM. Reduces HBM I/O from O(T²) to O(T). 2-4x speedup, same quality. |
| "Which sparse?" | Local (window) good for short context, cheap. Strided for long context. Combine both for best trade-off. |
| "GQA?" | Multiple Q heads share K,V. Reduces KV cache 4-8x. Minimal quality loss. Works with all methods. |
| "Hardware dependency?" | Flash Attention needs A100/H100. Sparse works anywhere. Check deployment GPU. |
| "Quality impact?" | Flash: 0%. Sparse local: <1%. Sparse strided: <2%. GQA: <1%. Combine carefully. |

## Related Topics
- [[transformers]] — core architecture with attention
- [[kv-cache]] — KV cache reduction with GQA
- [[inference-optimization]] — attention optimization as part of broader optimization
- [[quantization]] — orthogonal compression technique

## Resources
- [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [GQA: Training Generalized Multi-Query Transformers](https://arxiv.org/abs/2305.13245)
- [Efficient Attention: You Only Need to Care About Output Tokens](https://arxiv.org/abs/2302.10379)
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://arxiv.org/abs/2309.06180)
