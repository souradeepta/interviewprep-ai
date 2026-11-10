# Optimization

**TL;DR:** Optimization chooses parameter updates that reduce an objective, balancing gradient accuracy, update size, computational cost, and generalization behavior.

## Prerequisites

Prerequisites: derivatives, vectors, and [loss functions](loss-functions.md). You can derive gradient descent, compare batch/mini-batch and adaptive variants, diagnose learning-rate problems, and explain why minimizing training loss is not identical to maximizing deployment quality. This ML note focuses on optimizer mechanics and selection; the AI overview situates training within the broader model lifecycle.

## Learning Outcomes

- Explain the central assumptions and mechanism of optimization.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Imagine standing on a landscape whose height is the loss. The gradient points toward the steepest local increase, so a descent step moves in the opposite direction. Step size controls how far to move. Large steps make rapid progress in broad valleys but may jump across a narrow minimum; tiny steps are safe yet slow. In machine learning, the landscape is high-dimensional and stochastic, so the local slope is estimated from a minibatch rather than known exactly.

Optimization is not merely a search for the lowest possible training loss. The desired outcome is a model that generalizes under a realistic evaluation. A method that reaches low training loss quickly may need regularization, validation-based stopping, or a different schedule to achieve better test performance. Conversely, poor training loss is not evidence of generalization quality.

## How It Works

For parameters θ and objective J(θ), gradient descent uses θₜ₊₁=θₜ−η∇J(θₜ). Full-batch descent evaluates the gradient over all training examples per update; stochastic descent uses one; mini-batch descent averages B examples. Mini-batches trade gradient noise against hardware efficiency. If a per-example loss is averaged, changing B does not directly multiply the expected gradient, though it changes noise, update count per epoch, and useful learning-rate settings.

Momentum accumulates gradient history to damp oscillations. AdaGrad scales steps down for parameters with frequent large gradients; RMSProp uses a moving average of squared gradients; Adam combines first and second moments with bias correction. AdamW applies weight decay separately from its adaptive gradient update. These methods add state and may have distinct generalization behavior. Gradient clipping limits update-driving gradient magnitude when spikes are harmful; it does not repair a wrong loss or learning rate.

The learning rate is often the most consequential hyperparameter. Warmup can stabilize early updates, while cosine or step decay can refine later training. Schedulers must be indexed consistently by optimizer steps, especially with [gradient accumulation](gradient-accumulation.md) or distributed data parallelism. Track loss, gradient norm, update norm, and parameter norm; a falling loss alone may conceal unstable or excessively large updates.

For convex objectives, appropriate gradient descent conditions can guarantee convergence to a global optimum. Deep networks are nonconvex, so initialization, data order, architecture, and optimizer state influence the solution. Saddle points and flat regions complicate the landscape, but in practice training quality is diagnosed empirically with controlled experiments rather than a guarantee that one optimizer finds a global minimum.

An optimizer should be selected as part of an experimental protocol. Hold data order, augmentation, precision mode, stopping rule, and total compute reasonably constant when comparing two choices. Record the number of examples processed and wall-clock time: the same epoch count can represent different update counts when batch sizes differ. When one optimizer produces a better score, repeat the comparison over seeds if the gap is small, since initialization and minibatch order can create meaningful variability. The comparison should answer whether the optimizer improves final task quality, time to reach a target, or both; these are separate outcomes.

Optimization can also fail because the objective is poorly conditioned or contains invalid values, not because the optimizer family is wrong. Check labels and loss range, inspect gradients on a tiny batch, and verify that the model can reduce loss from a known initialization. If the training loop uses mixed precision or distributed workers, confirm loss scaling, gradient aggregation, and update frequency before tuning optimizer coefficients. A controlled numerical sanity check often saves more compute than launching a larger search.

## Trade-offs and Selection Boundary

| Optimizer | Useful property | Boundary |
|---|---|---|
| Full-batch GD | Exact gradient for the chosen dataset | Expensive updates for large datasets |
| Mini-batch SGD | Efficient stochastic updates; strong generalization baseline | Sensitive to scale, batch size, and schedule |
| Momentum SGD | Smooths noisy directions and accelerates consistent movement | Can overshoot; state convention matters |
| Adam/AdamW | Adaptive coordinate scaling and rapid early progress | More state; not universally best for final generalization |
| L-BFGS | Curvature-aware steps for smaller smooth problems | Memory and line-search cost make it awkward at large scale |

Start with the optimizer established for the model family: AdamW is common for transformer fine-tuning, momentum SGD is a robust image-model baseline, and quasi-Newton methods can suit smaller smooth objectives. Compare under equal compute and tune the learning-rate schedule before attributing gains to an optimizer. If the objective is nondifferentiable or discrete, consider surrogate optimization or a method designed for that setting.

## Implementation

```python
def sgd_step(theta, grad, lr):
    return [w-lr*g for w,g in zip(theta,grad)]

def momentum_step(theta, grad, velocity, lr=.1, beta=.9):
    velocity = [beta*v+g for v,g in zip(velocity,grad)]
    return sgd_step(theta, velocity, lr), velocity

theta = [3.0, -2.0]
grad = [6.0, -4.0]
plain = sgd_step(theta, grad, .1)
with_momentum, velocity = momentum_step(theta, grad, [0.,0.], .1, .9)
assert plain == [2.4, -1.6] and with_momentum == plain
assert all(abs(x) < abs(y) for x,y in zip(plain,theta))
```

## Failure Modes and Diagnostics

1. **Learning rate too high:** loss spikes, becomes non-finite, or oscillates. Reduce it and inspect update-to-parameter norm ratios; run a short range test.
2. **Learning rate too low:** loss decreases only imperceptibly. Plot loss against updates, confirm gradients are nonzero, then test a larger range.
3. **Wrong scheduler cadence:** accumulated training steps decay at microbatch frequency. Log optimizer updates explicitly and align total schedule steps.
4. **Exploding gradients:** update norms jump on a small number of batches. Identify offending inputs/losses, inspect normalization, and apply clipping only when appropriate.
5. **Confusing training objective with validation objective:** optimizer improves loss but model selection degrades. Monitor prespecified validation measures and retain the best checkpoint.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What is gradient descent? | Repeatedly move parameters opposite the objective gradient, scaled by a learning rate. |
| Why use mini-batches? | They provide efficient hardware utilization and lower-variance updates than single examples without full-dataset cost. |
| Debug follow-up: training is unstable after increasing batch size. | Revisit learning rate, warmup, schedule steps, gradient scaling, and update norms; batch changes alter optimization dynamics. |

## Related Concepts

- [Learning-rate schedules](learning-rate-schedules.md), [momentum](momentum-and-acceleration.md), and [regularization](regularization.md)
- [Matching notebook](../notebooks/28-optimization.ipynb)
- [Interview practice: gradient descent variants](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Bottou, Curtis, and Nocedal, [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173).
- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
