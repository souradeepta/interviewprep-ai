# Activation Functions

## Detailed Explanation

Introduces nonlinearity enabling deep learning...

## Core Intuition

A key technique in machine learning.

## How It Works

1. Receive pre-activation value z = w·x + b (weighted sum of inputs)
2. Apply the non-linear activation function: a = σ(z)
3. For ReLU: a = max(0, z) — zero for negative inputs, identity for positive
4. For sigmoid: a = 1/(1+e⁻ᶻ) — squashes output to (0,1), used for binary output
5. For softmax (output layer): aₖ = e^(zₖ)/Σⱼ e^(zⱼ) — produces probability distribution over K classes
6. The activation's derivative σ'(z) is used in backpropagation: δ = δ_next · σ'(z)
7. Choice of activation affects gradient flow — ReLU avoids vanishing gradients; sigmoid causes them in deep networks

```mermaid
graph TD
    A[Input] --> B[Process]
    B --> C[Output]
```

## Architecture / Trade-offs

Trade-off 1 vs trade-off 2

## Interview Q&A

**Q: What is the dying ReLU problem and how do you fix it?**
A: ReLU outputs zero for any negative input, and its gradient is also zero there. If a neuron's weights are updated such that its pre-activation is always negative for all training examples, that neuron will never activate and its weights will never be updated — it's "dead". Causes: too large learning rate, bad initialization. Fixes: LeakyReLU (small negative slope), ELU (smooth negative region), or careful LR and initialization tuning.

**Q: Why is ReLU preferred over sigmoid for hidden layers?**
A: Sigmoid saturates at both extremes (output near 0 or 1), where the derivative approaches 0 — this causes vanishing gradients in deep networks. ReLU has constant gradient of 1 for positive inputs, so gradients flow easily through deep networks. ReLU also has sparser activations (many zeros), which can be computationally efficient. Sigmoid is still used for binary output layers where a probability interpretation is needed.

**Q: What is GELU and why is it used in transformers?**
A: GELU (Gaussian Error Linear Unit) computes x·Φ(x) where Φ is the standard Gaussian CDF. Unlike ReLU which has a sharp cutoff at 0, GELU smoothly gates inputs based on their value relative to the distribution. Empirically, GELU provides small but consistent accuracy gains in transformers (BERT, GPT use it). The smooth gradient may help optimization in attention-heavy architectures.

**Q: When would you use softmax vs sigmoid for the output layer?**
A: Sigmoid for binary classification (single output, probability of positive class) or multi-label classification where classes are independent (each label can be independently 0 or 1). Softmax for mutually exclusive multiclass classification where probabilities must sum to 1. Never use softmax in hidden layers — it kills gradient flow by squashing all activations to compete against each other.

**Q: How does activation function choice affect weight initialization?**
A: He initialization (variance = 2/n_in) is designed for ReLU, which zeroes half its inputs — the factor of 2 compensates for this. Xavier initialization (variance = 2/(n_in + n_out)) is for symmetric activations like tanh and sigmoid. Using the wrong pairing (e.g., Xavier with ReLU) causes the variance to shrink with depth, slowing convergence. Always match initialization to activation.

**Q: What is the exploding gradient problem and how does it differ from vanishing gradients?**
A: Exploding gradients occur when gradient magnitudes grow exponentially with depth, causing parameter updates that are too large and destabilize training (loss diverges to NaN). Most common in RNNs with long sequences. Fix: gradient clipping (clip_grad_norm_(1.0)). Vanishing gradients cause parameters in early layers to barely update, preventing deep networks from learning hierarchical features. Both arise from multiplying many values during backprop.
## Best Practices

- Use ReLU as default for hidden layers in feedforward networks
- Use GELU for transformers and attention-based models (standard in BERT/GPT)
- Use sigmoid only for binary output layer probabilities
- Use softmax only for multiclass output layer
- Use tanh for RNNs where negative outputs matter
- Monitor dead neuron rate (fraction with zero gradient) when using ReLU
- Try Swish/Mish if ReLU is causing issues — often small accuracy gains

## Common Pitfalls

- Using sigmoid in hidden layers of deep networks — vanishing gradient kills learning
- Dead ReLU neurons (always outputting 0) caused by high learning rates or bad initialization
- Applying softmax in hidden layers instead of output — it squashes gradients
- Forgetting that activation choice affects initialization — must pair ReLU with He init


## Code Examples

### Example 1: Activation Functions Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 100)

relu = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
elu = np.where(z > 0, z, 0.1 * (np.exp(z) - 1))

plt.figure(figsize=(12, 4))
plt.plot(z, relu, label='ReLU')
plt.plot(z, sigmoid, label='Sigmoid')
plt.plot(z, tanh, label='Tanh')
plt.plot(z, elu, label='ELU')
plt.xlabel('z'), plt.ylabel('f(z)')
plt.legend(), plt.title('Activation Functions')
plt.grid(), plt.show()
```

### Example 2: Dying ReLU Problem

```python
# Demonstrate dying ReLU
X_biased = X - 10  # Shift to negative region

relu_layer = nn.ReLU()
sigmoid_layer = nn.Sigmoid()

with torch.no_grad():
    X_torch = torch.FloatTensor(X_biased)
    relu_out = relu_layer(X_torch)
    sigmoid_out = sigmoid_layer(X_torch)

relu_dead = (relu_out == 0).sum() / relu_out.numel()
print(f"Dead ReLU percentage: {relu_dead:.1%}")
print(f"Sigmoid output min: {sigmoid_out.min():.4f}, max: {sigmoid_out.max():.4f}")
```

### Example 3: LeakyReLU vs ReLU

```python
from torch.nn import LeakyReLU

leaky_relu = LeakyReLU(negative_slope=0.1)
relu = nn.ReLU()

X_test_negative = torch.FloatTensor(X_biased)
relu_out = relu(X_test_negative)
leaky_out = leaky_relu(X_test_negative)

print(f"ReLU dead neurons: {(relu_out == 0).sum()}")
print(f"LeakyReLU dead neurons: {(leaky_out == 0).sum()}")
print(f"LeakyReLU allows gradients for negative inputs!")
```

## Related Concepts

- [Gradient Descent](./01-gradient-descent.md)
- [Cross-Validation](./22-cross-validation.md)
- [Hyperparameter Tuning](./26-hyperparameter-tuning.md)
