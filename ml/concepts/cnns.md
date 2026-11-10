# Convolutional Neural Networks

**TL;DR:** CNNs exploit local connectivity and shared filters to process grid-structured inputs efficiently while preserving spatial organization.

## Prerequisites

Prerequisites: tensors, [neural networks](neural-networks.md), and gradient descent. You can compute convolution output shape and parameter count, distinguish translation equivariance from invariance, explain receptive fields and residual blocks, and identify when a convolutional inductive bias fits a task. The ML lesson emphasizes architecture trade-offs; the CV track develops image-specific pipelines and tasks.

## Learning Outcomes

- Explain the central assumptions and mechanism of cnns.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A small stencil that detects an edge can be slid across an image. The same learned filter recognizes that edge wherever it appears, instead of learning a separate detector for every pixel. Early layers detect local patterns; successive layers combine them into parts and larger structures. Weight sharing makes this substantially more parameter-efficient than connecting every pixel to every hidden unit.

Convolution is translation equivariant: shifting the input shifts the feature map (away from boundaries). It is not automatically translation invariant. Pooling, striding, global pooling, and data augmentation can add tolerance, but may also discard exact location needed for segmentation or keypoint prediction.

## How It Works

For input H×W with kernel k, padding p and stride s, output height is floor((H+2p−k)/s)+1 (similarly for width). A convolution with Cᵢ input and Cₒ output channels has k²CᵢCₒ weights plus biases, independent of image dimensions. Its approximate multiply-add count is HₒWₒk²CᵢCₒ. Padding “same” preserves dimensions when stride is one; stride reduces resolution. Dilated convolution increases receptive field without proportionally increasing parameters.

Typical blocks combine convolution, normalization, and nonlinearity. Residual connections add an identity path, making it easier to optimize deep stacks. Depthwise-separable convolution factors spatial filtering per channel from channel mixing, reducing arithmetic for mobile models. Input channel order, scaling, resize/crop policy, and normalization are part of the model contract.

Receptive field describes how much of the original input can influence one feature. A stack of 3×3 convolutions grows the receptive field gradually; stride and dilation change that growth. The theoretical receptive field may be larger than the effective area that materially influences a prediction. For classification, global average pooling can aggregate spatial evidence; for segmentation, dense prediction requires retaining or recovering spatial alignment through upsampling or skip connections. The task's output geometry should guide downsampling rather than applying pooling mechanically.

Convolution boundary behavior is part of the learned function. Zero padding injects artificial dark pixels near edges; reflect or replicate padding provides different context. “Same” padding definitions can differ for even kernels or strides, and resize/crop pipelines may shift objects relative to labels. Check output sizes analytically, then visualize overlays between predicted maps and targets. A one-pixel alignment error can be severe for segmentation and nearly invisible in an aggregate classification score.

Data augmentation often improves CNN generalization by exposing label-preserving variations such as crops, flips, or color shifts. But transformations encode assumptions: horizontal flips are invalid for text, laterality, or directional symbols; aggressive crops can remove the object; color jitter can erase diagnostic cues. Training augmentation belongs only on training examples, with validation representing the intended inference preprocessing. Augmentation strength should be tuned against both overall quality and important small-object or subgroup slices.

Pretrained CNNs can transfer useful visual filters to small datasets. Match input scale, color order, normalization, and architecture to the checkpoint, then compare a frozen feature extractor with partial and full fine-tuning. If target imagery differs substantially in camera or lighting, frozen low-level filters may not transfer. Per-domain error slices can reveal this issue sooner than an overall score. For high-resolution data, tiling or patch inference changes the context seen by the model and may create boundary artifacts, so evaluate the deployed tiling strategy.

## Trade-offs and Selection Boundary

| Architecture choice | Benefit | Cost or boundary |
|---|---|---|
| Standard convolution | Strong local spatial prior and mature hardware support | Compute grows with channels and spatial resolution |
| Stride/pooling | Reduces later-layer compute | Loses fine localization and can alias patterns |
| Residual blocks | Supports deeper networks and gradient flow | More memory/compute; shape changes need projection shortcuts |
| Depthwise-separable conv | Lower parameter and multiply-add count | Hardware kernels may not deliver proportional latency gain |
| CNN vs ViT | CNN is data-efficient for local spatial structure | Global interactions are indirect; ViTs may scale better with data |

Prefer CNNs for images with strong local structure, limited data, or efficient edge deployment. A tree model may be a poor choice for raw high-resolution pixels, while CNNs may be a poor fit when global relations dominate and training data supports attention-based models.

## Implementation

```python
def conv2d_single(image, kernel, stride=1, padding=0):
    h, w = len(image), len(image[0]); k = len(kernel)
    padded = [[0.0]*(w+2*padding) for _ in range(h+2*padding)]
    for i in range(h):
        for j in range(w): padded[i+padding][j+padding] = image[i][j]
    oh = (h+2*padding-k)//stride+1; ow = (w+2*padding-k)//stride+1
    out = [[sum(padded[i*stride+a][j*stride+b]*kernel[a][b]
                for a in range(k) for b in range(k)) for j in range(ow)] for i in range(oh)]
    return out

edge = [[1,0,-1],[1,0,-1],[1,0,-1]]
out = conv2d_single([[1,1,0],[1,1,0],[1,1,0]], edge)
assert len(out) == len(out[0]) == 1 and out[0][0] != 0
assert conv2d_single([[1]*5 for _ in range(5)], [[1]*3 for _ in range(3)], stride=2)[0][0] == 9
```

## Failure Modes and Diagnostics

1. **Wrong channel or input normalization:** images are interpreted incorrectly and pretrained performance collapses. Compare one preprocessed image and tensor range with training pipeline expectations.
2. **Excessive stride early:** small objects disappear. Inspect feature-map resolution and per-size validation metrics; preserve higher resolution for detection/segmentation.
3. **Boundary artifacts from padding:** predictions near image edges differ. Evaluate edge slices and consider padding policy or training augmentation.
4. **Assuming fewer FLOPs guarantees faster inference:** depthwise ops may be poorly utilized. Benchmark actual target hardware and batch sizes.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why are CNNs parameter-efficient on images? | Local connectivity and shared filters reuse the same detector across spatial positions. |
| Equivariance versus invariance? | Convolution preserves a shifted response location (equivariance); pooling or aggregation may reduce sensitivity to that shift (invariance). |
| Design follow-up: tiny objects are missed. | Check resize/stride and feature-map resolution, then measure recall by object size before increasing model capacity. |

## Related Concepts

- [Neural networks](neural-networks.md), [batch normalization](batch-normalization.md), [transfer learning](transfer-learning.md)
- [Matching notebook](../notebooks/05-cnns.ipynb)
- [Interview practice: image classifier implementation](../interview-prep/ml-coding-questions.md)

## References

- LeCun et al., [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).
- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
