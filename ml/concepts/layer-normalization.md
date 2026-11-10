# Layer Normalization

**TL;DR:** Layer normalization normalizes each example across its feature dimension, stabilizing hidden-state scale without depending on other examples in the batch.

## Prerequisites

Prerequisites: vectors, mean and variance, and neural-network layers. By the end, you can compute layer normalization, distinguish it from [batch normalization](batch-normalization.md), explain its role in transformer blocks, and diagnose instability caused by incorrect axes or placement. The ML lesson focuses on training behavior and deployment boundaries; the broader AI material uses normalization as one part of architecture design.

## Learning Outcomes

- Explain the central assumptions and mechanism of layer normalization.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Imagine every example arriving with a different volume knob setting. Normalization adjusts the scale within each example so downstream layers see a consistent range. Layer norm uses the features belonging to that one example, so its output does not change because unrelated examples were placed beside it in a minibatch. This makes it suitable for variable-length sequences and tiny or changing batch sizes.

It does not erase useful information permanently: learned scale γ and offset β restore a task-specific location and scale after standardization. It also does not make a network immune to poor initialization, bad learning rates, or distribution shift.

## How It Works

For a feature vector x∈Rᵈ, calculate μ=(1/d)Σⱼxⱼ and σ²=(1/d)Σⱼ(xⱼ−μ)². Then yⱼ=γⱼ(xⱼ−μ)/√(σ²+ε)+βⱼ. ε avoids division by zero; γ and β are learned per normalized feature. The operation costs O(d) per example and has no running averages. For sequences shaped [batch, tokens, hidden], transformer layer norm commonly normalizes the final hidden dimension independently at every token.

Placement matters. Pre-LN blocks normalize before attention/MLP sublayers and generally provide a direct residual path that helps optimization at depth. Post-LN normalizes after residual addition and can require careful warmup or initialization. Architectures vary; this is a design choice, not a universal theorem.

The normalized dimensions are part of the layer's semantics. For a tensor shaped [batch, tokens, hidden], normalizing the final hidden dimension gives each token its own mean and variance. Accidentally normalizing over tokens makes one token's representation depend on neighboring tokens and sequence length. In a causal language model, this may introduce unwanted dependence on future positions even when attention itself is causally masked. Check the exact axes and add a test that changes another batch element or token while holding the current vector fixed.

The learnable scale and shift allow the transformation to represent an identity-like mapping when γ≈√(variance+ε) and β≈mean, though the exact statistics depend on each input vector. Layer norm therefore does not simply constrain every feature permanently to zero mean and unit variance after training. It stabilizes the input to a sublayer while preserving learned flexibility. RMSNorm omits mean subtraction and uses root-mean-square scaling, reducing some computation and changing the inductive bias. Replacing layer norm with RMSNorm can require retraining or careful conversion, not just a function-name swap.

Normalization affects how residual branches and initialization work together. In a pre-normalized transformer, the residual stream can pass through many blocks without each branch forcing a fresh normalization after addition. This contributes to stable gradient paths, but residual activations can still grow and optimization can still diverge. Monitor hidden-state norms across depth and sequence length. If only late blocks have abnormal scale, investigate residual branch scaling, attention outputs, and initialization rather than changing epsilon indiscriminately.

Layer norm computes statistics independently for each example/token, so predictions are stable when requests are batched differently. This is useful for online serving where batch composition changes continuously. However, it does not normalize feature groups independently unless the architecture explicitly does so, and learned affine parameters may encode training-specific scale. Under distribution shift, layer norm can hide some absolute scale changes while leaving semantic changes untouched. It is not a general-purpose domain adaptation method.

## Trade-offs and Selection Boundary

| Choice | Advantage | Cost or boundary |
|---|---|---|
| Layer norm | Independent of batch composition; works for sequence models and small batches | Normalizes across features, which may be undesirable when feature groups have distinct meaning |
| Batch norm | Often effective for convolutional models with stable, large batches | Training uses batch statistics and inference uses running estimates |
| RMSNorm | Removes mean subtraction and normalizes root mean square | Different inductive bias; verify compatibility with pretrained architecture |
| No normalization | Simpler and may suit shallow models | Deep optimization may become more sensitive to scale |

Prefer layer norm for transformer hidden states, recurrent sequences, and irregular batch sizes. Do not swap normalization in a pretrained model casually: the learned parameters and statistics are architecture-specific.

## Implementation

```python
import math

def layer_norm(row, gamma, beta, eps=1e-5):
    mean = sum(row) / len(row)
    var = sum((x - mean) ** 2 for x in row) / len(row)
    return [g * (x - mean) / math.sqrt(var + eps) + b
            for x, g, b in zip(row, gamma, beta)]

x = [1.0, 2.0, 5.0]
y = layer_norm(x, [1.0] * 3, [0.0] * 3)
assert abs(sum(y)) < 1e-9
assert round(sum(v*v for v in y) / len(y), 5) == 1.0
# A common offset and scale do not alter normalized values (when eps is negligible).
z = layer_norm([11.0, 13.0, 19.0], [1.0]*3, [0.0]*3)
assert max(abs(a-b) for a, b in zip(y, z)) < 1e-4
```

## Failure Modes and Diagnostics

1. **Wrong normalized axis:** normalizing across batch mixes examples and makes predictions depend on their batch companions. Test a single example alone and with unrelated examples; outputs should match.
2. **Epsilon too small or omitted:** near-constant vectors can produce unstable values. Include ε and monitor activation/gradient finiteness.
3. **Replacing pretrained normalization:** weights learned for one normalization rule may no longer behave correctly. Compare layer outputs and validation metrics after any architectural change.
4. **Assuming normalization solves exploding gradients:** inspect residual-stream and gradient norms; adjust initialization, clipping, or learning rate when needed.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What statistics does layer norm use? | Mean and variance over features of one example (often per token), then learned scale and shift. |
| Layer norm versus batch norm? | Layer norm is independent of other examples and has no running statistics; batch norm uses batch statistics during training. |
| Debug follow-up: predictions change with batch composition. What do you inspect? | Normalization axes and train/eval mode; true layer norm should not depend on neighboring examples. |

## Related Concepts

- [Batch normalization](batch-normalization.md), [transformers](transformers.md), and [neural networks](neural-networks.md)
- [Matching notebook](../notebooks/18-layer-normalization.ipynb)
- [Interview practice: gradient descent](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Ba, Kiros, and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450).
- Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).
