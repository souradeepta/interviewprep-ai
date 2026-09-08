---
title: "Vision Transformer: An Image is Worth 16x16 Words"
authors: "Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, et al."
year: 2020
venue: "ICLR"
doi: "https://doi.org/10.1145/3394486"
arxiv: "https://arxiv.org/abs/2010.11929"
domain: "vision"
difficulty: "intermediate"
interview_frequency: "medium"
related_concepts:
  - papers/nlp/concepts/01-attention-is-all-you-need
---

# Vision Transformer: An Image is Worth 16x16 Words

## Paper Overview

**Title:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Authors:** Alexei Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dmitry Weissenborn, Xiaoying Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Grangier, Johann Poley, Yonatan Belinchuk (Google Research)

**Published:** ICLR 2021 | [arXiv](https://arxiv.org/abs/2010.11929)

**Citation:** 30,000+ (influential modern vision architecture)

For decades, convolutional neural networks (CNNs) dominated computer vision. ResNets, EfficientNets, and other CNN variants set the accuracy standard. Then Vision Transformer (ViT) came along and proved something unexpected: you don't need convolutions at all. By treating image patches as "words" and applying the transformer architecture (the same thing that powers language models), ViT achieved competitive or better accuracy than CNNs while being more sample-efficient at scale. The key insight: **transformers can learn visual representations by simply attending to different image patches.**

**Why this matters for interviews:** Vision Transformers represent a paradigm shift: the same architecture (transformers) works for text, images, and multimodal tasks. This is important because it shows architectural universality and raises fundamental questions about what makes models work. Interviewers ask about ViT frequently because it challenges the CNN assumption and demonstrates scalability principles.

---

## Core Contribution

### The Problem: Are Convolutions Necessary for Vision?

For ~10 years, the answer seemed obvious: yes. CNNs exploit spatial locality and translation equivariance—properties that seem fundamental to vision. But what if you applied transformers directly to images? The challenge: images are high-dimensional (e.g., 224×224×3 = 150,528 pixels). Applying self-attention to all pixels would be prohibitively expensive.

### The Solution: Linear Patch Embeddings

Instead of processing pixels directly, ViT splits images into **non-overlapping patches** (e.g., 16×16 patches from 224×224 images):
- 224×224 image → 14×14 = 196 patches of 16×16
- Each patch: 16×16×3 = 768 values
- Linear projection to embedding dimension: 768 → 768 (or smaller)

Now you have 196 tokens (like words in NLP), and you can apply the standard transformer architecture:

```
Image (224×224×3)
  ↓
Patch Embedding (196 × 768)
  ↓
Add Position Embeddings
  ↓
Transformer Blocks (Multi-head Self-attention + MLP)
  ↓
Classification Token (CLS token)
  ↓
Linear Classifier
```

### Key Innovation: Patch-Based Representation

Instead of learning hierarchical features (like CNNs do), ViT operates on flat sequences of patches:
- No spatial hierarchy (patches don't maintain 2D structure)
- Global receptive field from the first layer (self-attention attends to all patches)
- Parameter-sharing through transformer weights (all patches processed by same layers)

This design is radically different from CNNs but remarkably effective.

---

## Key Ideas & Algorithm

### How Vision Transformer Works

**Step 1: Patch Embedding**
1. Split image into non-overlapping patches (e.g., 14×14 patches for 224×224 image)
2. Flatten each patch: 16×16×3 = 768 values
3. Linear projection: 768 → d (embedding dimension, typically 768)
4. Add special CLS token (learnable, like [CLS] in BERT)
5. Add position embeddings (learnable, 1D positional encoding)

**Step 2: Transformer Encoder**
1. Stack L transformer blocks (e.g., 12 blocks)
2. Each block:
   - Multi-head self-attention: patches attend to all patches
   - MLP feed-forward network
   - Layer normalization and residual connections
3. Output: contextualized embeddings for each patch + CLS token

**Step 3: Classification**
1. Take CLS token embedding (output of last transformer block)
2. Linear classifier: d → num_classes
3. Softmax for probability distribution

### Comparison to CNN

| Aspect | CNN (ResNet-50) | Vision Transformer |
|--------|-----------------|-------------------|
| **Receptive Field** | Grows with depth (local → global) | Global from layer 1 |
| **Patches/Regions** | Overlapping, hierarchical | Non-overlapping, flat |
| **Parameter Sharing** | Spatial (convolution kernels) | Full (transformer weights) |
| **Inductive Bias** | Locality, translation equivariance | Permutation invariant (before pos encoding) |
| **Pre-training Data** | Works well with ImageNet (1M images) | Better with JFT-300M or similar |

### Scaling Properties

Vision Transformers have different scaling characteristics than CNNs:

```
ImageNet Accuracy vs. Model Size

ResNet:        Scaling plateaus around 88% accuracy
Vision Transformer: Continues improving with more data/compute
                    (with JFT-300M: 88-90% accuracy)
```

Key finding: **ViT needs more data than CNN to reach the same accuracy**, but with sufficient pre-training data, ViT wins on accuracy and transfer learning.

---

## Architecture & Trade-offs

### Patch Size Trade-offs

| Patch Size | Tokens | Computation | Info Density | Use Case |
|------------|--------|-------------|--------------|----------|
| 4×4 | 3136 (224×224) | Very High | Highest detail | Small images, fine details |
| 8×8 | 784 (224×224) | High | High | Balanced |
| 16×16 | 196 (224×224) | Low | Medium | Standard, efficient |
| 32×32 | 49 (224×224) | Very Low | Lowest | Very large images |

**Trade-off Analysis:**
- Smaller patches → more tokens → more computation, more detail
- Larger patches → fewer tokens → faster, but lose fine-grained info
- Standard: 16×16 balances detail and efficiency

### Model Variants

| Model | Patch | Layers | Hidden | Heads | Parameters |
|-------|-------|--------|--------|-------|-----------|
| ViT-Tiny | 16 | 12 | 192 | 3 | 5M |
| ViT-Small | 16 | 12 | 384 | 6 | 22M |
| ViT-Base | 16 | 12 | 768 | 12 | 86M |
| ViT-Large | 16 | 24 | 1024 | 16 | 304M |
| ViT-Huge | 14 | 32 | 1280 | 16 | 632M |

### Position Embeddings

**Options:**

1. **Learnable 1D Position Embeddings** (ViT standard)
   - One embedding per patch (0, 1, ..., N)
   - Learned during pre-training
   - Pro: Simple, works well in practice
   - Con: Not explicitly position-aware

2. **2D Position Embeddings**
   - Encode row, column position
   - Pro: More explicit spatial structure
   - Con: More complex, not always better

3. **Relative Position Bias**
   - Attention based on relative distance between patches
   - Pro: Spatial structure, generalizes to different resolutions
   - Con: Added complexity

**Empirical finding:** Simple learnable 1D embeddings work surprisingly well.

---

## Interview Q&A

**Q: Why does Vision Transformer work without convolutions? What does it gain?**

A: Convolutions enforce locality (each pixel sees small neighborhoods). ViT uses self-attention to let patches attend to *all* other patches from layer 1, giving global context immediately. This is more flexible: ViT can learn which patches to attend to, rather than using hand-crafted locality. The trade-off: ViT needs more pre-training data to learn those attention patterns. With enough data (JFT-300M), it outperforms CNNs.

**Q: How do you choose patch size? What's the trade-off?**

A: Smaller patches (4×4, 8×8) preserve fine detail but require more computation (more tokens). Larger patches (32×32) are efficient but lose detail. Standard is 16×16 for 224×224 images. For high-resolution images or detail-critical tasks (medical imaging), use smaller patches. For speed, use larger patches. The right choice depends on your image resolution, compute budget, and task requirements.

**Q: How does ViT compare to ResNet for transfer learning?**

A: ResNet works well with ImageNet pre-training (~1M images). ViT actually *needs* more pre-training data to reach its potential—it underperforms ResNet on ImageNet-only pre-training. But with larger datasets (JFT-300M), ViT transfers better to downstream tasks. For practical scenarios: if you have access to large pre-trained ViT (e.g., from OpenAI CLIP, Google, Meta), use it. Otherwise, ResNet is safer on ImageNet.

**Q: Why add a CLS token? Why not use global average pooling like CNNs?**

A: The CLS token (borrowed from BERT) is a learned aggregation mechanism. It attends to all patches and learns what information is important for the task. Global average pooling is simpler but treats all patches equally. In practice, CLS is more flexible and often performs slightly better. Some variants use global average pooling over all patch embeddings—both work, but CLS is the standard.

**Q: How do position embeddings work in ViT? Why not 2D?**

A: ViT uses simple learnable 1D position embeddings. You might expect 2D embeddings (row, column) to be better, but experiments show 1D works just as well. Why? Self-attention is powerful enough to learn spatial structure implicitly. The main surprise: ViT learned meaningful spatial structure without explicit 2D bias, suggesting transformers are more flexible than assumed.

**Q: How does ViT scale compared to CNNs?**

A: ViT scales differently: it benefits more from large datasets and compute. With fixed compute, CNN might be faster. But at scale (massive datasets + compute), ViT's parameterefficiency and global attention often win. This is why large labs (Google, OpenAI, Meta) pre-train huge ViTs. For inference on consumer hardware, CNNs may still be preferable.

---

## Best Practices

- **Pre-training data matters:** ViT significantly underperforms CNNs on ImageNet-only. Use ImageNet + additional data (e.g., unlabeled web data) or switch to pre-trained models (OpenAI, Google, Hugging Face).

- **Patch size selection:** 16×16 is standard for 224×224 images. For medical/satellite (high-res), try 8×8. For speed, try 32×32. Benchmark on your task.

- **Layer normalization:** ViT uses LayerNorm instead of BatchNorm (because it doesn't depend on batch statistics). Essential for training stability.

- **Learning rate warmup:** ViT benefits from learning rate warmup (e.g., 40 epochs linear warmup for 300 total). This improves training stability and final accuracy.

- **Regularization:** DropOut, Stochastic Depth, Label Smoothing help. ViT tends to overfit more than CNNs on small datasets, so regularization is important.

- **Resolution adaptation:** ViT can handle different input resolutions by interpolating position embeddings. Use linear interpolation and fine-tune for best results.

- **Efficiency:** Use ViT distillation (KD from large ViT to smaller ViT) or pruning for deployment. For real-time applications, consider CNN alternatives.

---

## Common Pitfalls

- **Mistake: Using ViT without pre-training on large data.** ViT from random initialization on ImageNet fails. Always use pre-trained weights (e.g., from Hugging Face, OpenAI, Google). Training from scratch requires millions of images.

- **Mistake: Using standard BatchNorm in ViT.** ViT needs LayerNorm. BatchNorm (which depends on batch statistics) can be unstable with transformers, especially with small batch sizes.

- **Mistake: Ignoring position embeddings.** Some variants experimented with *no* position embeddings or absolute pixel positions. Without position info, accuracy drops significantly. Always include position embeddings.

- **Mistake: Assuming ViT is slower than CNN.** ViT inference is actually comparable to ResNet-50 when using similar compute. But ViT benefits more from larger batches and GPUs—on CPU or small batches, CNN may be faster.

- **Mistake: Neglecting the CLS token.** Some practitioners try global average pooling instead of CLS. While it can work, CLS is the standard and typically better. It's explicitly learned to aggregate task-relevant information.

- **Mistake: Fine-tuning with high learning rates.** When fine-tuning pre-trained ViT, use lower LR (0.0001-0.001) than you'd use for CNN. ViT weights are well-initialized and need gentle adjustment.

---

## Code Examples

### Example 1: Patch Embedding from Scratch

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection: flatten patches to embed_dim
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        # proj: Conv2d with stride=patch_size → (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)  # (B, embed_dim, num_patches^0.5, num_patches^0.5)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

# Test
patch_embed = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
patches = patch_embed(x)
print(f"Input: {x.shape}, Patches: {patches.shape}")
# Output: Input: torch.Size([2, 3, 224, 224]), Patches: torch.Size([2, 196, 768])
```

### Example 2: Vision Transformer Block

```python
class TransformerBlock(nn.Module):
    """Vision Transformer block: Multi-head Attention + MLP"""
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x

# Test
block = TransformerBlock(embed_dim=768, num_heads=12, mlp_dim=3072)
x = torch.randn(2, 197, 768)  # (batch, num_patches + cls_token, embed_dim)
output = block(x)
print(f"Input: {x.shape}, Output: {output.shape}")
# Output: Input: torch.Size([2, 197, 768]), Output: torch.Size([2, 197, 768])
```

### Example 3: Complete Vision Transformer

```python
class VisionTransformer(nn.Module):
    """Complete Vision Transformer for image classification"""
    
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=768,
                 num_heads=12, num_layers=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x[:, 0])  # Take CLS token
        x = self.fc(x)
        
        return x

# Test
model = VisionTransformer(num_classes=10)
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"Input: {x.shape}, Output: {output.shape}")
# Output: Input: torch.Size([2, 3, 224, 224]), Output: torch.Size([2, 10])
```

---

## Related Concepts

- [ResNet](./01-resnet.md) — CNN alternative that ViT compares against
- [Attention Is All You Need](../../nlp/concepts/01-attention-is-all-you-need.md) — Original transformer architecture
- [Scaling Laws for Neural Language Models](../../nlp/concepts/04-scaling-laws.md) — Scaling principles apply to ViT too
- [CLIP](../../retrieval/concepts/02-clip.md) — Vision Transformer for multimodal tasks
