# Neural Networks

**TL;DR:** A neural network composes learned affine transformations and nonlinear activations, fitting flexible representations through gradient-based optimization.

## Prerequisites

Prerequisites: vectors and matrices, derivatives, and basic probability. You can describe a feed-forward network as a computation graph, explain forward and backward passes, connect loss/activation choices to tasks, recognize capacity and generalization risks, and choose a simple baseline before adding depth. This ML lesson establishes the model mechanics; the AI track discusses neural applications and architectures without replacing this foundation.

## Learning Outcomes

- Explain the central assumptions and mechanism of neural networks.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A neural network is a layered collection of simple operations whose composition can express complex functions. A layer computes weighted sums, shifts them by biases, and applies a nonlinear activation. Early layers can form reusable intermediate features; later layers combine them into a prediction. Think of a workshop where each station transforms material according to adjustable settings. Training adjusts those settings so the final product better matches examples.

The analogy should not suggest that hidden neurons necessarily correspond to human-readable concepts. A network can represent useful distributed features with no single unit having a clean semantic meaning. It can also fit spurious patterns if capacity, data, and regularization are poorly matched. More layers do not automatically produce better representations.

## How It Works

For input x, layer l computes zˡ=Wˡhˡ⁻¹+bˡ and hˡ=φ(zˡ), with h⁰=x. The final layer produces logits or a numeric estimate. A loss L(y,ŷ) measures disagreement with target y. Backpropagation applies the chain rule through the computation graph to compute ∂L/∂W and ∂L/∂b. An optimizer then updates parameters; backpropagation computes gradients but does not itself choose an update rule. These are distinct steps in training.

For a fully connected layer from width m to width n, parameters are mn+n, and a batch of size B costs about O(Bmn) multiply-adds. Convolutional layers exploit local connectivity and weight sharing; recurrent layers reuse state through sequence positions; attention mixes information through content-based weights. Residual paths and normalization can improve optimization in deep networks. [Activations](activation-functions.md) provide nonlinearity; [weight initialization](weight-initialization.md) sets initial scale; [loss functions](loss-functions.md) specify what gradients optimize.

Training usually separates data into training, validation, and final test sets. The training set fits parameters; validation selects architectures, hyperparameters, and stopping point; the test set estimates the chosen procedure's performance once. If observations share users, time, or sources, random splitting may leak information and overstate generalization. Data normalization and learned preprocessing should be fit inside training folds. The data contract is part of the model, not a peripheral step.

Neural-network optimization is nonconvex: different initializations or minibatch orders can lead to different solutions. Stochastic gradient descent and variants use minibatches to estimate gradients. Learning-rate schedules, batch size, initialization, normalization, and optimizer state affect the trajectory. Monitor both training and validation curves, but diagnose their split validity and label quality before adjusting model depth. A small model that overfits a tiny batch is a useful sanity check for a training pipeline.

## Trade-offs and Selection Boundary

| Choice | Benefit | Cost or boundary |
|---|---|---|
| Shallow/wide network | Can model nonlinear interactions with modest depth | May require many units and does not exploit task structure |
| Deep network | Composes features hierarchically | Optimization and data requirements grow; residual paths may help |
| Fully connected layers | Flexible for fixed-size vectors | Parameter-heavy and ignores spatial/sequence structure |
| CNN | Efficient local spatial processing | Global interactions require depth or additional mechanisms |
| Transformer | Content-based long-range interaction and parallel training | Attention has sequence-length costs and needs substantial data/compute |

Use neural networks when data volume, feature structure, and task complexity support their capacity, or when pretrained representations are available. For many tabular datasets, boosted trees are a strong baseline. Use linear or probabilistic models when interpretability, calibration, data efficiency, or easy operational validation matters more than representational flexibility. Compare systems at an equal data and compute budget.

## Implementation

This tiny one-hidden-layer regression example performs a forward pass, calculates mean squared error, and checks that a parameter can be updated in the direction of lower loss. It uses finite differences only to illustrate the gradient path.

```python
import math

def forward(x, w1, b1, w2, b2):
    hidden = max(0.0, w1*x+b1)
    return w2*hidden+b2

def loss(w1, b1, w2, b2):
    pairs = [(0.,1.), (1.,3.)]
    return sum((forward(x,w1,b1,w2,b2)-y)**2 for x,y in pairs)/len(pairs)

w1,b1,w2,b2 = 1., 0., 1., 0.
eps = 1e-5
grad_w2 = (loss(w1,b1,w2+eps,b2)-loss(w1,b1,w2-eps,b2))/(2*eps)
updated_w2 = w2 - .05*grad_w2
assert math.isfinite(loss(w1,b1,updated_w2,b2))
assert loss(w1,b1,updated_w2,b2) < loss(w1,b1,w2,b2)
```

## Failure Modes and Diagnostics

1. **Network cannot fit even a tiny batch:** suspect a broken gradient, label alignment, output/loss mismatch, or excessive regularization. Disable augmentation and regularization and verify a tiny subset can be memorized.
2. **Training improves while validation degrades:** capacity may fit noise, or validation distribution differs. Inspect learning curves, group/time split, and error slices before choosing regularization.
3. **Vanishing or exploding gradients:** gradient norms collapse or grow across layers. Check initialization, activation saturation, normalization, residual paths, learning rate, and clipping where appropriate.
4. **Training/inference preprocessing mismatch:** accuracy falls after deployment. Store transforms and schema with the model; compare tensors and predictions for the same examples through both pipelines.
5. **Metric improves only through leakage:** random split includes future or correlated entities. Re-evaluate with a split that reflects new users, sources, or time periods.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What does backpropagation do? | It applies the chain rule through a computation graph to compute parameter gradients; the optimizer uses them to update weights. |
| Why does a network need nonlinear activations? | Composing affine maps alone remains affine and cannot represent general nonlinear relationships. |
| Debug follow-up: network fails to learn a simple dataset. | Verify labels, output/loss semantics, gradient flow, and tiny-batch overfit before changing architecture. |

## Related Concepts

- [Activation functions](activation-functions.md), [optimization](optimization.md), [regularization](regularization.md)
- [Matching notebook](../notebooks/27-neural-networks.ipynb)
- [Interview practice: backpropagation implementation](../interview-prep/ml-coding-questions.md)

## References

- Goodfellow, Bengio, and Courville, [Deep Learning, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html).
- Rumelhart, Hinton, and Williams, [Learning Representations by Back-Propagating Errors](https://www.nature.com/articles/323533a0).
