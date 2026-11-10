# Activation Functions

**TL;DR:** Activation functions add nonlinear transformations between learned affine maps, letting a neural network represent functions that a single linear layer cannot.

## Prerequisites

Prerequisites: [neural networks](neural-networks.md), affine transformations, and derivatives. After this lesson, you should be able to explain why stacking linear layers alone stays linear, compare ReLU, sigmoid, tanh, and GELU, identify saturation and dead-unit behavior, and choose a hidden-layer or output activation from the task's output constraints. This ML lesson treats activations as modeling and optimization decisions; the broader [AI roadmap](../../ai/README.md) introduces neural-network vocabulary.

## Learning Outcomes

- Explain the central assumptions and mechanism of activation functions.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

An affine layer rotates, scales, and shifts a space. Composing two such layers still produces one affine map: W₂(W₁x+b₁)+b₂. A nonlinear activation bends that map. Think of a hinge in a flexible ruler: each layer can build a new piecewise region, and more layers combine these regions into useful decision boundaries. Without the hinge, adding layers only adds redundant matrix multiplication.

ReLU, max(0,z), keeps positive evidence and clips negative evidence. A stack of ReLUs creates piecewise-linear regions. Sigmoid squashes a score to (0,1), useful for a binary probability output, but its flat tails provide tiny gradients. Tanh is zero-centered but also saturates. GELU smoothly gates values according to their magnitude and is common in transformer feed-forward blocks.

## How It Works

For layer l, compute z⁽ˡ⁾ = W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾ and h⁽ˡ⁾ = φ(z⁽ˡ⁾). In backpropagation, the local derivative φ'(z) multiplies the upstream gradient. If many derivatives are near zero, gradients vanish; if the derivative is zero for a unit and all its inputs, that ReLU unit may stop learning. ReLU's derivative is 1 above zero and 0 below; the value at zero is conventionally chosen as zero. Sigmoid's derivative is σ(z)(1−σ(z)), at most 1/4. GELU is approximately zΦ(z), where Φ is the standard normal CDF.

The activation usually costs O(Bd) for batch B and width d, small relative to dense projection O(Bd²). Output layers encode task semantics: identity for unconstrained regression, sigmoid for binary probability, softmax for mutually exclusive classes, and no normalization when training logits with a numerically stable cross-entropy implementation. Independent multi-label classes generally need separate sigmoids rather than a softmax.

Activation choice also changes the distribution and scale of signals passed to later layers. ReLU outputs are nonnegative and place exact zeros in the representation, which can make computations sparse but shifts the mean upward. Tanh is centered around zero, a property that can help optimization, but large positive or negative inputs sit on a flat region. GELU and SiLU retain a smooth negative tail and are differentiable around zero; their extra computation is often small relative to large matrix multiplications, but accelerator kernels and exact approximations vary. These effects interact with normalization and initialization, so changing the activation in a pretrained network changes more than a local formula.

For a hidden layer, the useful question is not simply which activation is newest. Consider depth, expected input scale, gradient flow, model family, and hardware support. A ReLU network with He initialization is a conventional robust baseline. A transformer checkpoint trained with GELU or a gated variant should generally retain its specified activation during fine-tuning. If training from scratch, compare candidate activations under identical data, initialization policy, optimizer, and update budget. Monitor activation histograms and gradient norms; the activation distribution can explain a failure that the final metric alone cannot.

The output layer has a separate semantic contract. For regression constrained to nonnegative values, an identity output can produce invalid negatives; a softplus or a transformed target may be appropriate. For a bounded probability, sigmoid or softmax produces values in a range, but the loss API may expect logits and perform the normalization internally. Applying softmax twice changes gradients and probabilities. For ordinal categories, a standard softmax ignores order unless the output/loss explicitly represents it. Treat output activation, target encoding, and loss as one matched design decision.

## Trade-offs and Selection Boundary

| Choice | Useful when | Cost or boundary |
|---|---|---|
| ReLU | Fast hidden layers and sparse activations | Negative-side units have zero gradient; monitor dead units |
| Leaky ReLU | ReLU units die under negative inputs | Extra slope choice; does not guarantee better generalization |
| GELU | Smooth gating in modern language and vision models | More compute than ReLU; gains depend on architecture and training |
| Sigmoid / tanh | Bounded outputs or gates | Saturate; avoid as default deep hidden activations |
| Softmax | Exactly one class among K outcomes | Inappropriate for independent labels; probabilities compete |

Use a hidden activation after most affine layers, but generally not after the final regression layer. For class probabilities, match the activation and loss API: many libraries expect raw logits and combine softmax/sigmoid with cross-entropy for stability.

## Implementation

This tiny example evaluates four activations without external packages and shows sigmoid saturation in its derivative.

```python
import math

def relu(x): return max(0.0, x)
def leaky_relu(x, alpha=0.1): return x if x >= 0 else alpha * x
def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
def gelu_approx(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x**3)))

for z in (-4.0, 0.0, 2.0):
    s = sigmoid(z)
    print(z, relu(z), leaky_relu(z), round(s, 4), round(gelu_approx(z), 4), round(s*(1-s), 4))
assert relu(-4) == 0 and leaky_relu(-4) < 0
assert sigmoid(0) == 0.5 and sigmoid(8) * (1 - sigmoid(8)) < 0.001
```

## Failure Modes and Diagnostics

1. **Dead ReLU population:** many units output zero for every validation example. Inspect activation histograms; lower an excessive learning rate, revise initialization, or use a small negative slope.
2. **Saturated sigmoid hidden stack:** activations cluster near zero and one and gradients shrink. Check gradient norms by layer; use ReLU/GELU and suitable initialization in hidden layers.
3. **Wrong output semantics:** softmax on a multi-label task forces labels to compete. Confirm label ontology and use independent sigmoid logits with binary cross-entropy when labels can co-occur.
4. **Numerical overflow in hand-written sigmoid:** exp(-z) can overflow for large negative z. Use a stable library implementation or branch on the sign of z.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why are activations necessary? | A composition of affine layers is still affine; nonlinearities make richer functions possible. |
| Why is ReLU often preferred to sigmoid in hidden layers? | It is cheap and has a non-saturating positive-side gradient, though dead units remain possible. |
| Design follow-up: a label may have several classes at once; what output do you use? | Independent sigmoid logits and binary cross-entropy, not mutually exclusive softmax. |

## Related Concepts

- [Neural networks](neural-networks.md) and [weight initialization](weight-initialization.md)
- [Matching notebook: activation functions](../notebooks/01-activation-functions.ipynb)
- [Interview practice: gradient descent variants](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Goodfellow, Bengio, and Courville, [Deep Learning, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html).
- Hendrycks and Gimpel, [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
