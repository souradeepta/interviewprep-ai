# Weight Initialization

**TL;DR:** Weight initialization sets starting parameter scales so signals and gradients can flow through a network before learning has established useful weights.

## Prerequisites

Prerequisites: [neural networks](neural-networks.md), variance, and [activation functions](activation-functions.md). You can explain why zero initialization breaks hidden-unit symmetry, derive fan-in/fan-out scaling intuition, distinguish Xavier from He initialization, and diagnose activation/gradient scale problems. This ML lesson focuses on optimization stability; the AI roadmap treats architecture more broadly.

## Learning Outcomes

- Explain the central assumptions and mechanism of weight initialization.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Before training, hidden neurons have no reason to specialize. If every weight begins identically, identical neurons receive identical updates and remain clones, wasting model capacity. Random initialization breaks that symmetry. But randomness alone is not enough: values that are too large can make activations and gradients explode, while values that are too small can make signals vanish through many layers. Good initialization aims to preserve useful signal scale as it propagates forward and backward.

The ideal scale depends on architecture, activation, normalization, and residual structure. Initializers are principled starting points rather than guarantees. Pretrained weights should normally be loaded as trained; applying a fresh initializer to them destroys learned information.

## How It Works

Consider a linear layer zⱼ=Σᵢwᵢⱼxᵢ. If inputs are independent with zero mean and variance Var(x), then Var(zⱼ)≈fan_in·Var(w)·Var(x). To preserve forward variance, choose Var(w)≈1/fan_in. Backpropagation instead depends on fan_out. Xavier/Glorot initialization balances both directions, commonly with variance 2/(fan_in+fan_out) or a uniform distribution bounded by a corresponding square-root scale. It is often used with tanh or sigmoid-like activations, though saturation can still occur.

ReLU sets roughly half of centered pre-activations to zero under simplifying assumptions. He/Kaiming initialization accounts for that reduction by using variance about 2/fan_in for forward preservation. Normal and uniform distributions are both used. Fan-in and fan-out conventions depend on tensor layout, such as convolutional kernels, so framework helpers avoid many manual errors. Biases are commonly initialized to zero, though architecture-specific exceptions exist.

Symmetry breaking concerns hidden units with shared downstream roles: initializing all weights identically means their gradients match. Initializing every weight to zero is therefore problematic for multilayer networks, though output biases or some convex linear models can be initialized safely at zero. Orthogonal matrices can help preserve norms in recurrent or very deep networks, but do not replace activation-aware scaling. Residual networks and normalization alter signal propagation, which is why modern architectures use initialization schemes matched to their block structure.

Diagnose initialization by inspecting layerwise activation means/variances and gradient norms at the start of training, before extensive updates. Saturated sigmoid/tanh units indicate scale problems. Dead ReLUs can arise from bad initial scale or excessive learning rates. If batch/layer normalization is present, initialization still matters for residual branches and optimization, though normalization may reduce sensitivity in some layers.

Initialization conventions depend on how a framework stores a matrix. A dense layer might store weights as [out_features, in_features] and multiply by the transpose, while a derivation writes W as [in_features, out_features]. A fan calculation applied to the wrong axis changes the intended variance. Convolutional kernels add spatial dimensions to fan-in and fan-out; grouped or depthwise convolution changes those counts again. Use the library helper for standard layers, and test custom tensor layouts with a small controlled sample before relying on formulas copied from another framework.

Residual architectures introduce another scale consideration. Each block adds a transformed branch to an identity path. If every branch begins with large variance, repeated additions can make activation scale grow with depth; some designs initialize the final normalization scale or residual branch near zero so the block begins close to identity. Recurrent networks also reuse a matrix repeatedly over timesteps, so orthogonal initialization may help control norm changes, but gate biases and nonlinearities still matter. Follow the initialization choices for the exact architecture rather than assuming a single fan-based rule covers all layers.

Initialization should be evaluated before a long run. Run a forward pass on a representative batch and collect per-layer activation mean, standard deviation, fraction of zeros for rectifiers, and non-finite counts. Run one backward pass and record gradient norms by layer. These checks do not prove that training will succeed, but they rapidly locate scale collapse, saturation, and shape mistakes. After the first optimizer step, verify parameter deltas are finite and of reasonable size relative to the initial weights.

## Trade-offs and Selection Boundary

| Scheme | Typical use | Boundary |
|---|---|---|
| Xavier/Glorot | Tanh/sigmoid or roughly linear activations | Can be suboptimal for ReLU's zeroed activations |
| He/Kaiming | ReLU and related rectifiers | Nonlinearity and fan mode must match forward/backward intent |
| Orthogonal | Recurrent matrices or norm-preserving transformations | Does not address all nonlinear or residual scale issues |
| Zero biases | Common default with random weights | Some architectures benefit from a task-specific bias initialization |
| Pretrained checkpoint | Transfer learning and fine-tuning | Preserve source weights and configuration; do not reinitialize accidentally |

For a new dense network, start with the framework's He initializer for ReLU or Xavier for tanh-like layers. Confirm the framework interprets fan dimensions as expected, especially for transposed or convolutional weights. If using custom gates, residual scaling, or normalization, follow the architecture's published initialization recipe and verify the initial signal statistics empirically.

## Implementation

```python
import random, math

def he_normal(fan_in, count, seed=5):
    rng = random.Random(seed)
    std = math.sqrt(2.0/fan_in)
    return [rng.gauss(0.0,std) for _ in range(count)]

w = he_normal(fan_in=100, count=10000)
mean = sum(w)/len(w)
variance = sum((x-mean)**2 for x in w)/len(w)
assert abs(mean) < .01
assert abs(variance-.02) < .002
assert w == he_normal(100,10000,seed=5)  # fixed seed makes the demo repeatable
```

## Failure Modes and Diagnostics

1. **All hidden weights initialized to zero:** units stay symmetric and learn the same feature. Use a suitable random initializer and verify layer weight diversity.
2. **Wrong fan calculation:** variance grows or shrinks with width. Check tensor layout and measure pre-activation variance across layers.
3. **Initializer mismatched to activation:** ReLU layers receive Xavier scales and early activations shrink. Use activation-aware initialization and inspect gradient/activation histograms.
4. **Pretrained weights overwritten:** fine-tuning behaves like random initialization. Confirm state-dict loading and compare parameter checksums before and after model construction.
5. **Initialization blamed for learning-rate failure:** updates are unstable despite reasonable initial activations. Track update-to-weight norm and run a learning-rate test before changing the initializer.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why not initialize every weight to zero? | Hidden units become symmetric and receive identical gradients, so they fail to specialize. |
| Xavier versus He? | Xavier balances fan-in/fan-out variance for near-linear/tanh activations; He compensates for rectifier activation sparsity with larger variance. |
| Debug follow-up: gradients vanish in deeper layers. | Inspect activation and gradient scales by layer, verify activation-matched fan scaling, and check saturation/normalization. |

## Related Concepts

- [Activation functions](activation-functions.md), [batch normalization](batch-normalization.md), and [optimization](optimization.md)
- [Matching notebook](../notebooks/40-weight-initialization.ipynb)
- [Interview practice: backpropagation](../interview-prep/ml-coding-questions.md)

## References

- Glorot and Bengio, [Understanding the Difficulty of Training Deep Feedforward Neural Networks](http://proceedings.mlr.press/v9/glorot10a.html).
- He et al., [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852).
