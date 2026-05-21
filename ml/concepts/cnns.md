# Convolutional Neural Networks (CNNs)

## TL;DR
Specialized neural networks for grid-structured data (images, audio spectrograms). Use convolutional
filters with weight sharing to detect the same pattern anywhere in the input. Foundation for
computer vision — ResNet, EfficientNet, and Vision Transformers all build on this.

## Core Intuition
A human recognizes a cat regardless of where in an image it appears. CNNs achieve this via weight
sharing: the same filter applied at every spatial location. Early filters detect edges; later
filters detect textures, parts, and objects.

## How It Works

**Convolution:** filter $W \in \mathbb{R}^{k \times k}$ slides across input with stride s.
Output size: $\lfloor(H - k)/s + 1\rfloor$.

**Padding:** "same" padding preserves spatial dimensions; "valid" reduces them.

**Pooling:** max pooling takes the max in each window — reduces spatial dims, adds translation invariance.

**Standard CNN block:** Conv → BatchNorm → ReLU → MaxPool

**ResNet skip connection:** $F(x) + x$ — learn the residual. Allows 100+ layer networks.
Gradient highway: $F(x) = 0$ is the easy identity path.

## Key Properties / Trade-offs
- Parameter sharing: a 3×3 conv has only 9k params regardless of input spatial size
- Receptive field grows with depth — deeper networks capture larger context
- Depthwise separable convolutions (MobileNet): 8–9× fewer ops than standard conv

## Common Mistakes / Gotchas
- Confusing equivariance (conv) and invariance (pooling)
- Not using batch norm — training very deep CNNs without it is extremely difficult
- Forgetting to normalize inputs to [0,1] or [-1,1]

## Code Example
```python
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Why are CNNs good for images?" | Weight sharing + local connectivity = efficient, translation-equivariant feature detection. Far fewer params than fully connected layers. |
| "What is a skip connection?" | $F(x) + x$ adds input directly to output. Creates gradient highways for backprop; learning identity is trivial, enabling very deep networks. |

## Related Topics
- [Neural Networks](../neural-networks.md) — [Attention Mechanism](attention-mechanism.md)
- [Implementations: CNN Image Classifier](../../implementations/cnn-image-classifier.ipynb)

## Resources
- [CS231n Conv Networks](https://cs231n.github.io/convolutional-networks/)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) — He et al.
