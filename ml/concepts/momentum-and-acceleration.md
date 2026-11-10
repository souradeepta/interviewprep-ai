# Momentum and Acceleration

**TL;DR:** Momentum accumulates a smoothed history of gradients, reducing oscillation across steep directions and accelerating movement along consistent directions.

## Prerequisites

Prerequisites: [optimization](optimization.md), gradients, and learning rates. You can write the momentum update, distinguish classical momentum from Nesterov and adaptive optimizers, explain its interaction with batch size and schedules, and diagnose overshooting. This ML note focuses on optimizer dynamics; the AI roadmap provides general model-training orientation.

## Learning Outcomes

- Explain the central assumptions and mechanism of momentum and acceleration.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A ball rolling down a valley does not choose a direction from only the slope beneath it at the current instant; it carries velocity from recent motion. Momentum gives gradient descent a similar memory. If gradients point consistently downhill, their accumulated direction grows and progress accelerates. If gradients alternate across a narrow ravine, averaging damps the side-to-side motion. The memory is helpful but can also carry a model past a minimum when the direction changes.

Momentum does not provide free acceleration independent of learning rate. The learning rate sets the scale of the gradient-driven update, while the momentum coefficient controls how much history persists. They must be tuned together, and optimizer comparisons need comparable training budgets.

## How It Works

One common convention defines velocity vₜ=βvₜ₋₁+gₜ and parameters θₜ₊₁=θₜ−ηvₜ, where gₜ=∇L(θₜ), η is learning rate, and β is in the range 0 ≤ β < 1 is momentum. Another convention multiplies the gradient contribution by 1−β; this changes the effective update scale and is why framework conventions should be checked. With a constant gradient g, velocity approaches g/(1−β) in the first convention. High β therefore amplifies sustained movement.

Nesterov momentum computes the gradient at a look-ahead position, approximately θₜ−ηβvₜ₋₁, then uses that gradient to correct direction. This can provide anticipatory correction in some settings. Adam maintains both first and second gradient moments and rescales coordinates using estimated variance; it is not merely momentum, even though its first moment is momentum-like. Momentum adds constant-size state per parameter, so memory overhead is O(number of parameters).

Momentum smooths stochastic minibatch gradients but does not remove their bias or fix mislabeled data. Large batch size reduces gradient noise and may change useful β or η values. Weight decay interaction also depends on optimizer implementation: decoupled weight decay applies a separate parameter shrinkage, while adding an L2 penalty feeds its gradient into momentum state.

The velocity is optimizer state and has a concrete lifecycle. It should persist across ordinary training batches because the purpose is to accumulate direction. It should be restored when resuming from a checkpoint, reset when training a genuinely new model, and treated deliberately when fine-tuning with changed data or learning rates. Keeping stale velocity after a major distribution or objective change can cause an abrupt update toward a direction no longer supported by current gradients. Resetting it every epoch, on the other hand, discards memory and changes the intended optimization process.

Momentum also interacts with gradient noise and batch size in ways that make rules of thumb conditional. With very noisy minibatches, smoothing can help; with a large batch and a carefully tuned rate, too much smoothing can delay response to changing curvature. If gradients alternate signs, velocity damps the oscillation but may lag when the optimum direction changes. Observe parameter-update norms, not just raw gradients: optimizer state can make a small current gradient yield a large step. A useful diagnostic is the ratio ||Δθ||/(||θ||+ε), plotted by layer and training step. Sudden jumps suggest an optimizer or schedule issue even if scalar loss has not yet diverged.

A useful way to understand the dynamics is to examine a quadratic bowl L(θ)=½θᵀHθ. Along an eigenvector of the Hessian with curvature λ, plain gradient descent multiplies the error by (1−ηλ) at each step. A single learning rate must work for both the steepest and flattest directions; a large condition number therefore causes zigzagging or slow progress. Momentum changes this into a second-order recurrence. It can damp oscillations in steep directions while accumulating progress in shallow directions, but its stability range still depends on η and the largest curvature. This explains why momentum helps ill-conditioned problems without making arbitrary learning rates safe.

In practice, treat the optimizer as a stateful algorithm. A training checkpoint must save parameters, momentum buffers, scheduler position, and often mixed-precision scaler state. Resuming with weights but no velocity is a different optimization trajectory. When switching from a pretrained checkpoint to a new task, preserving the weights is usually sensible while resetting optimizer state is often safer. If fine-tuning is a continuation of the same objective and data regime, retaining state may be appropriate; document the choice and compare validation behavior.

A fair optimizer comparison fixes the data order, batch size, number of examples seen, augmentation, and validation schedule. Reporting “epochs to convergence” can mislead when methods perform different numbers of updates or use different schedules. Record both wall-clock time and steps because an update can have different cost. Compare final quality and stability across several seeds where training noise matters. Monitor update-to-weight ratios by layer, gradient norms, and loss spikes; a stable scalar loss can hide large updates in a small subset of parameters.

Weight decay deserves special attention. Coupled L2 regularization adds λθ to the gradient before the momentum buffer, so decay participates in accumulated history. Decoupled weight decay applies shrinkage separately and is commonly paired with AdamW. These are not identical, especially with adaptive scaling or changing learning rates. Also distinguish momentum from an exponential moving average of model weights: optimizer momentum controls the update direction, while weight averaging smooths the evaluated parameters. Confusing the two can produce incorrect descriptions of EMA checkpoints.

Use classical momentum when dense gradients and a stable learning-rate schedule make its behavior predictable. If gradients are sparse or coordinates have widely different scales, compare an adaptive optimizer. If training overshoots after a schedule transition, inspect the effective step ηv and warmup/decay boundary before changing β. For a plateau, test whether the issue is a poor representation, noisy labels, or a schedule floor; increasing momentum cannot fix a misspecified objective. The selection boundary is empirical: use the simplest optimizer that meets quality and time targets on a controlled validation protocol.

## Trade-offs and Selection Boundary

| Method | Benefit | Boundary |
|---|---|---|
| SGD without momentum | Simple and easy to interpret | May zigzag or converge slowly in ill-conditioned valleys |
| Classical momentum | Smooths gradients and speeds consistent directions | Can overshoot after curvature or direction changes |
| Nesterov momentum | Corrects using a look-ahead gradient | Adds convention complexity; gain depends on problem |
| Adam / AdamW | Per-coordinate adaptive scaling; useful for sparse/noisy gradients | More state; may generalize differently and still needs tuning |

Use momentum SGD for well-understood dense optimization, especially vision workloads where it is a strong baseline. Use adaptive methods when gradients are sparse or scales differ substantially, then compare validation behavior and compute. If loss oscillates, inspect learning rate, momentum, batch noise, and curvature rather than reducing β reflexively.

## Implementation

```python
def momentum_step(theta, grad, velocity, lr=.1, beta=.9):
    velocity = [beta*v + g for v,g in zip(velocity, grad)]
    theta = [w-lr*v for w,v in zip(theta, velocity)]
    return theta, velocity

theta, v = [1.0], [0.0]
theta, v = momentum_step(theta, [-2.0], v, lr=.1, beta=.5)
assert theta == [1.2] and v == [-2.0]
theta, v = momentum_step(theta, [-2.0], v, lr=.1, beta=.5)
assert v == [-3.0] and theta == [1.5]  # sustained direction builds velocity
```

## Failure Modes and Diagnostics

1. **Oscillating or diverging loss:** learning rate and accumulated velocity are too large for local curvature. Plot update norms and reduce learning rate before changing several optimizer settings.
2. **Momentum state reset each batch:** intended smoothing never develops. Keep optimizer state across batches and recreate it only when starting a new run or deliberately resetting.
3. **Confusing weight decay with L2 under momentum:** regularization behaves differently across implementations. Record whether decay is decoupled and inspect the optimizer's parameter groups.
4. **Schedule changes with stale velocity:** a sudden learning-rate drop or restart can interact with accumulated state. Log both learning rate and update norm; consider a deliberate warm restart/reset only when justified.
5. **Comparing optimizers at unequal budgets:** faster early convergence is mistaken for better final quality. Compare equal examples, updates, or compute and report both convergence speed and final metric.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why use momentum? | It averages gradient history, damping oscillations and accelerating in directions with consistent gradients. |
| Momentum versus Adam? | Momentum smooths gradients; Adam also adapts each parameter's step using a second-moment estimate. |
| Debug follow-up: loss diverges after raising momentum. | Inspect update norms and learning rate together, then lower effective step size and verify optimizer state handling. |

## Related Concepts

- [Optimization](optimization.md), [learning-rate schedules](learning-rate-schedules.md), and [gradient accumulation](gradient-accumulation.md)
- [Matching notebook](../notebooks/25-momentum.ipynb)
- [Interview practice: gradient descent variants](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Sutskever et al., [On the Importance of Initialization and Momentum in Deep Learning](https://www.cs.toronto.edu/~hinton/absps/momentum.pdf).
- PyTorch, [SGD optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).
