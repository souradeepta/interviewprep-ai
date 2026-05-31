---
title: "ResNet: Deep Residual Learning for Image Recognition"
authors: "He, Zhang, Ren, Sun"
year: 2015
venue: "CVPR"
doi: "https://doi.org/10.1109/CVPR.2015.123"
arxiv: "https://arxiv.org/abs/1512.03385"
domain: "vision"
difficulty: "intermediate"
interview_frequency: "medium"
related_concepts:
  - modern-ai/concepts/08-inference-optimization
---

# ResNet: Deep Residual Learning for Image Recognition

## Paper Overview

**Title:** Deep Residual Learning for Image Recognition

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)

**Published:** CVPR 2015 | [arXiv](https://arxiv.org/abs/1512.03385)

**Citation:** 100,000+ (one of the most influential papers in computer vision)

Before ResNet, there was a critical problem: deep neural networks trained for image recognition would actually perform *worse* than shallower networks. Counterintuitively, adding more layers hurt accuracy—a phenomenon called the **degradation problem**. This wasn't due to overfitting; it was a fundamental optimization issue. ResNet solved this with a deceptively simple idea: **residual connections** that allow information to flow unchanged through multiple layers. This paper revolutionized deep learning and remains the foundation of modern computer vision.

**Why this matters for interviews:** ResNet is a must-know for understanding modern architecture design. Interviewers ask about it constantly (especially for vision roles), and the core insight—skip connections—appears everywhere: Transformers, ResNets in NLP, attention mechanisms. You need to understand not just what residual connections are, but *why* they work and when to use them.

---

## Core Contribution

### The Problem: The Degradation Problem

In 2014-2015, very deep networks (>50 layers) were difficult to train. Researchers observed that:
- A 56-layer network had *higher* training error than a 20-layer network
- This wasn't overfitting (test error also increased)
- Deeper wasn't better—it was actively worse

The hypothesis: very deep networks are hard to optimize. The gradient signal weakens as it flows backward through 50+ layers. Explicitly learning transformations is harder than learning *residuals* (the difference from a direct path).

### The Solution: Residual Connections (Skip Connections)

Instead of learning H(x) (the full transformation), learn the *residual* F(x) = H(x) - x:

y = F(x) + x

**Why this works:**
- If F(x) ≈ 0, the layer is a near-identity mapping (no harm in adding it)
- Gradients flow directly backward through the skip connection
- Each layer learns *incremental* improvements, not from-scratch transformations
- Deeper networks now work—you can train 152 layers successfully

### Key Innovation: The Residual Block

```
Input: x
  ↓
Conv → ReLU → Conv  (learns F(x))
  ↓ (element-wise add)
  └─────────────────────(identity: x)
  ↓
ReLU
  ↓
Output: ReLU(F(x) + x)
```

This simple architectural change enabled training 50+, 100+, even 152-layer networks—previously impossible.

---

## Key Ideas & Algorithm

### How Residual Blocks Work

**Step 1: Forward Pass**
1. Input x flows into the block
2. Two parallel paths:
   - **Main path:** Conv → BatchNorm → ReLU → Conv → BatchNorm
   - **Skip path:** Identity (or projection if dimensions change)
3. Outputs are element-wise added: y = F(x) + x
4. Final ReLU: output = ReLU(y)

**Step 2: Backward Pass**
- Gradients flow through both paths:
  - Main path: through convolutions
  - Skip path: *directly, unmodified*
- The skip path ensures strong gradient signal even in deep networks

**Step 3: Why Optimization Becomes Easier**
- Each block needs only to learn F(x) = H(x) - x (the residual)
- Many blocks converge to near-identity mappings when they're not helping
- No vanishing gradient problem—gradients skip multiple layers

### Architecture Evolution

```
Standard Block (20 layers):
Conv 64×64 → Conv 64×64 → ReLU

Residual Block (identity):
[Conv 64×64 → BN → ReLU] + [skip]
[Conv 64×64 → BN]
ReLU

Residual Block (projection, dimension change):
[Conv 64→128 with stride=2 → BN → ReLU] + [Conv 64→128 stride=2 or MaxPool]
[Conv 128→128 → BN]
ReLU
```

### ResNet Configurations

| Model | Layers | Conv Layers | Top-1 Error | Parameters |
|-------|--------|-------------|------------|-----------|
| ResNet-18 | 18 | 17 residual + 1 conv | 30.6% | 11M |
| ResNet-34 | 34 | 33 residual + 1 conv | 26.7% | 21M |
| ResNet-50 | 50 | 49 residual + 1 conv | 24.0% | 25M |
| ResNet-101 | 101 | 100 residual + 1 conv | 23.6% | 44M |
| ResNet-152 | 152 | 151 residual + 1 conv | 23.3% | 60M |

---

## Architecture & Trade-offs

### Residual vs. Plain Networks

| Aspect | Plain Network | ResNet |
|--------|---------------|--------|
| **Depth** | 50+ layers → training fails | 152+ layers → trains fine |
| **Optimization** | Deep → vanishing gradients | Skip paths → gradient flow |
| **Initialization** | Sensitive to weight init | Robust (skip paths buffer) |
| **When to use** | Shallow models (<20 layers) | Deep architectures (50+) |

### Design Choices & Why

**1. Skip Every 2 Convolutions**
- Why not skip every 1? Too much noise; need to learn something
- Why not skip every 4? Fewer direct gradient paths; slower convergence
- 2 convolutions balances learning capacity + gradient flow

**2. Bottleneck vs. Basic Blocks**

Basic block (3×3 + 3×3):
```
x → [3×3 Conv, 64] → ReLU → [3×3 Conv, 64] → + x → ReLU
```

Bottleneck block (1×1 + 3×3 + 1×1):
```
x → [1×1 Conv, 64/4] → ReLU → [3×3 Conv, 64/4] → ReLU → [1×1 Conv, 64] → + x → ReLU
```

**Trade-off:**
- Bottleneck: fewer parameters, same expressiveness (used in ResNet-50+)
- Basic: simpler, faster for small models (used in ResNet-18/34)

**3. Projection for Dimension Mismatch**

When stride=2 or channels increase:
```
Option A: Identity mapping (x dimension must match)
Option B: Projection Conv: x → Conv(1×1) → stride=2 → channels increase
```

Projection adds flexibility but uses parameters. ResNet uses both judiciously.

---

## Interview Q&A

**Q: Why do residual connections solve the degradation problem?**

A: In plain networks, gradients must propagate through ~100 conv layers to update early weights. Residual connections create a shortcut: gradients can flow *directly* through skip connections without passing through many nonlinearities. This keeps the gradient signal strong. Additionally, at initialization, if a block's weights are small, F(x) ≈ 0 and the layer acts as identity—the network starts with direct paths to the output that keep gradients unattenuated.

**Q: When would you use a residual connection vs. a plain convolution block?**

A: Use residual connections whenever you want to go deeper than ~20-30 layers. Beyond that, plain networks degrade due to optimization difficulty. In practice: ResNet-50 for most vision tasks; ResNet-101 for tasks needing more capacity; ResNet-18 if you're memory-constrained but still want skip connections for training stability.

**Q: What's the difference between bottleneck and basic residual blocks?**

A: Basic blocks are 3×3 → 3×3. Bottleneck blocks are 1×1 (reduce) → 3×3 → 1×1 (expand). Bottlenecks reduce the number of channels before the expensive 3×3 convolution, cutting FLOPs ~4×. ResNet-50+ uses bottlenecks for efficiency; ResNet-18/34 use basic blocks since they're already small. For interviews: "use bottlenecks when you need depth + efficiency; basic blocks when you want simplicity."

**Q: How do you initialize ResNets? Does it matter?**

A: Standard He initialization works well. Key: with skip connections, poor initialization matters *less* than in plain networks. Why? If the residual branch weights are initialized small, the skip path dominates early training—the network is nearly identity. This stability is one reason ResNets are easier to train than plain networks. For production, just use default PyTorch init + BatchNorm.

**Q: Why do you need BatchNorm in residual blocks?**

A: BatchNorm stabilizes training by normalizing activations. In ResNets, it's essential because: (1) residuals add across channels—batch norm keeps this stable, (2) allows higher learning rates, (3) acts as regularization. Modern variants (GroupNorm, LayerNorm) work too, but BatchNorm + ResNet is the standard.

**Q: ResNets were proposed in 2015. Are they still used?**

A: Yes, extensively. ResNet-50 is the backbone for many production systems (object detection, segmentation, classification). Modern architectures (Vision Transformers, EfficientNets) often incorporate the same skip connection principle. The *idea* is more important than the exact architecture.

---

## Best Practices

- **For vision tasks:** Start with ResNet-50 (good accuracy/speed trade-off). Use ResNet-18 if latency-critical; ResNet-101 if you need more accuracy.

- **Bottleneck blocks:** Use them for ResNet-50+. They reduce FLOPs ~4× with minimal accuracy loss.

- **Batch normalization:** Essential. Always use BatchNorm after convolutions in residual blocks.

- **Learning rate:** Can use higher LR than plain networks (~0.1 for ImageNet) due to skip connections stabilizing training.

- **Initialization:** He initialization (default in PyTorch) works. For very deep networks (152+ layers), careful initialization of the residual branch helps early convergence.

- **Data augmentation:** Standard (RandAugment, Mixup) applies. ResNets benefit from aggressive augmentation without overfitting.

- **Transferability:** ResNet-50 pre-trained on ImageNet transfers well to many tasks. Fine-tuning with LoRA or small LR usually works.

---

## Common Pitfalls

- **Mistake: Using a 1×1 projection for all dimension mismatches.** Some ResNets use projection everywhere; modern variants use identity where possible (no extra parameters). Check the specific architecture.

- **Mistake: Not using BatchNorm in residual blocks.** Training becomes unstable. BatchNorm is not optional.

- **Mistake: Initializing the residual branch with large weights.** The block won't learn to be identity early on. Properly initialized, skip connections should dominate initially.

- **Mistake: Confusing "depth" with "width."** ResNet-50 is 50 layers, but with bottleneck blocks and grouped convolutions, adding width (more channels) might help more than depth.

- **Mistake: Assuming ResNet is optimal for your task.** For some applications (small images, limited compute), EfficientNet or Vision Transformers might be better. Always benchmark.

---

## Code Examples

### Example 1: Basic Residual Block from Scratch

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Basic residual block: Conv → BN → ReLU → Conv → BN, with skip connection."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Projection for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
```

### Example 2: ResNet-50 with Pre-training

```python
import torch
import torchvision.models as models
from torch import nn

model = models.resnet50(pretrained=True)

num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

for param in list(model.parameters())[:-20]:
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4
)

for epoch in range(20):
    for images, labels in train_loader:
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Example 3: Bottleneck Residual Block (ResNet-50 style)

```python
class BottleneckBlock(nn.Module):
    """Bottleneck block: 1×1 (reduce) → 3×3 → 1×1 (expand), with skip."""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        out = self.relu(out)
        return out
```

---

## Related Concepts

- [Vision Transformer](./02-vision-transformer.md) — Modern alternative to ResNets for vision
- [Scaling Laws for Neural Language Models](../nlp/concepts/04-scaling-laws.md) — Why deeper/wider models work
- [LoRA: Low-Rank Adaptation](../nlp/concepts/05-lora.md) — Efficient fine-tuning, applies to ResNets too
