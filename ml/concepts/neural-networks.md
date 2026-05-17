# Neural Networks

## TL;DR
A neural network approximates any function via layers of linear transformations + nonlinear
activations. Trained by backpropagation and gradient descent. Architecture and training
details matter enormously in practice.

## Core Intuition
A pipeline of feature detectors. Early layers detect simple patterns (edges in images, character
n-grams in text). Later layers combine them into complex abstractions (faces, intent). Each
layer transforms its input to make the next layer's job easier.

## How It Works

**Layer:** $h = \sigma(Wx + b)$ where σ is an activation function.

**Activations:**
- ReLU: $\max(0, x)$ — most common. No vanishing gradient for positive inputs.
- Sigmoid: $(1+e^{-x})^{-1}$ — squashes to (0,1). Output layer for binary classification.
- Softmax: multinomial output probabilities. Output layer for multi-class.
- GELU: smooth ReLU variant used in transformers.

**Backprop:** chain rule from output to input. Gradient of loss w.r.t. each weight.

**Initialization:** He init for ReLU ($\sqrt{2/n_{in}}$); Xavier for tanh/sigmoid. Never zeros.

**Batch Normalization:** normalize activations within mini-batch, then scale/shift with learned γ, β.
Stabilizes training, allows higher learning rates.

## Key Properties / Trade-offs
- Residual connections (skip connections): $F(x) + x$ — solve vanishing gradient in deep nets
- Width vs depth: deeper = more abstract representations; wider = more patterns per layer
- Batch norm behavior differs train vs eval — always call `model.eval()` at inference

## Common Mistakes / Gotchas
- Sigmoid/tanh in hidden layers → vanishing gradient. Use ReLU.
- Zero initialization → symmetric weights, all neurons learn identically
- Forgetting `model.eval()` at inference — dropout and batchnorm behave differently
- Exploding gradients → loss NaN → use gradient clipping

## Code Example
```python
import numpy as np

def relu(x): return np.maximum(0, x)
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()

class TwoLayerNet:
    def __init__(self, d_in, d_h, d_out):
        self.W1 = np.random.randn(d_in, d_h) * np.sqrt(2/d_in)  # He init
        self.b1 = np.zeros(d_h)
        self.W2 = np.random.randn(d_h, d_out) * np.sqrt(2/d_h)
        self.b2 = np.zeros(d_out)

    def forward(self, x):
        self.h = relu(x @ self.W1 + self.b1)
        return softmax(self.h @ self.W2 + self.b2)
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Why ReLU over sigmoid in hidden layers?" | ReLU doesn't saturate for positive inputs — gradient always 1, no vanishing. Sigmoid saturates at both ends, killing gradients. |
| "What is batch normalization?" | Normalizes activations within mini-batch to zero mean/unit variance, then scales/shifts with learned params. Stabilizes training. |
| "What is the vanishing gradient problem?" | In deep nets with sigmoid/tanh, gradients shrink exponentially through layers. Fixed by: ReLU, skip connections, batch norm. |

## Related Topics
- [Optimization](optimization.md) — [Regularization](regularization.md) — [CNNs](deep-learning/cnns.md)
- [Implementations: Neural Net From Scratch](../implementations/neural-net-from-scratch.ipynb)

## Resources
- [CS231n Neural Networks](https://cs231n.github.io/neural-networks-1/)
- [Deep Learning book Ch. 6–8](https://www.deeplearningbook.org/)
