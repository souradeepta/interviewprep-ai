# Learning-Rate Schedules

**TL;DR:** A learning-rate schedule changes optimizer step size over training so updates are large enough to make progress early and small enough to refine a solution later.

## Prerequisites

Prerequisites: [optimization](optimization.md), gradient descent, and training/validation curves. You can interpret learning rate as update scale, compare warmup, step decay, cosine decay, and adaptive optimizers, and choose a schedule based on update budget rather than epoch folklore. This ML lesson focuses on optimization control; the AI track's model overview provides broader architecture context.

## Learning Outcomes

- Explain the central assumptions and mechanism of learning rate schedules.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Walking toward a valley with long strides covers ground quickly but may overshoot the bottom. Tiny steps are stable but slow. A schedule changes stride length as training proceeds: begin with a workable step, then shrink it as the solution becomes more precise. Warmup is the gentle acceleration from nearly zero that can prevent unstable early updates in large models whose initial gradient scale is poorly behaved.

The schedule is part of the experiment. Changing it while comparing models changes the optimization budget and can invalidate conclusions. A cosine curve is not inherently better than a well-tuned constant rate on every problem.

## How It Works

Gradient descent updates θₜ₊₁=θₜ−ηₜgₜ, where ηₜ is the learning rate at step t. Step decay multiplies η by γ at milestones; exponential decay applies γ each step; cosine decay smoothly lowers η from η₀ toward η_min over T updates: ηₜ=η_min+½(η₀−η_min)(1+cos(πt/T)). Linear warmup sets ηₜ≈η_max·t/W for the first W steps, optionally followed by decay.

Schedulers may advance per optimizer update or per epoch. When gradient accumulation is used, schedule by actual optimizer steps, not microbatches. With variable-length training or early stopping, define whether total steps are fixed; otherwise the decay phase can end too early or never reach its intended minimum. Adam adapts coordinates individually but still has a global learning-rate multiplier and often benefits from scheduling.

Warmup and decay address different phases. Warmup gradually increases the rate from a small value to the target peak to avoid large early updates before gradients and normalization statistics stabilize. Decay gradually reduces the peak as the optimizer approaches a useful region. A schedule can use warmup followed by linear or cosine decay, but its milestones should be expressed in actual updates and examples processed. If training stops early, the model may never reach the low-rate refinement phase; if training continues past the planned endpoint, the rate may remain at the floor and waste compute.

Scheduler libraries differ in when they step relative to optimizer updates and whether the first rate is applied before or after the first batch. A resumed checkpoint must restore scheduler state and global step. Off-by-one errors can shift warmup or decay substantially in short runs. Plot the planned and observed learning rate against optimizer-step count, and include that plot in experiment diagnostics. For distributed workers, only one synchronized schedule should govern the shared update; rank-local data iteration must not cause scheduler divergence.

Schedule selection interacts with optimizer and batch size. AdamW may benefit from a different peak and decay pattern than momentum SGD; increasing global batch changes updates per epoch and gradient noise. A cosine curve with a fixed total duration assumes an estimate of the training budget, while reduce-on-plateau adapts to a chosen validation metric and can become unstable if that metric is noisy. Compare schedules at equal total examples and compute, and avoid attributing a better result to the scheduler if it simply received more updates.

Learning-rate range tests can provide a quick initial scale. Increase rate over a short run and observe where loss begins to decrease and where it becomes unstable, then choose a conservative peak. This is a heuristic, not a final search, and should be run on representative training data without contaminating evaluation. If loss is flat at every rate, inspect data and gradients before assuming a more elaborate schedule will help.

## Trade-offs and Selection Boundary

| Schedule | Useful when | Boundary |
|---|---|---|
| Constant | Short baselines and stable convex-like objectives | Needs a rate that remains suitable throughout training |
| Step / multistep | Known phases or reproducible epoch budgets | Milestones create abrupt changes and require domain tuning |
| Cosine | Fixed update budget and smooth annealing desired | Requires estimating total steps; restarts change the trajectory |
| Warmup then decay | Large batches, transformers, or unstable first updates | Too much warmup wastes budget; too little may not stabilize |
| Reduce on plateau | Validation metric has meaningful plateaus | Noisy validation can trigger repeated inappropriate reductions |

Select schedule and peak rate jointly on training-only/validation procedures. For pretrained fine-tuning, use a smaller peak rate than training from scratch and consider layer-wise rates. Compare equal update budgets and record the schedule in experiment metadata.

## Implementation

```python
import math

def warmup_cosine(step, total_steps, warmup_steps, peak=0.1, floor=0.0):
    if not 0 <= step < total_steps or not 0 <= warmup_steps < total_steps:
        raise ValueError("invalid schedule bounds")
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps - 1)
    return floor + 0.5 * (peak-floor) * (1 + math.cos(math.pi * progress))

rates = [warmup_cosine(s, 10, 2, peak=.1, floor=.01) for s in range(10)]
assert rates[0] < rates[1]
assert rates[1] == .1 and rates[-1] == .01
assert all(a >= b for a, b in zip(rates[1:], rates[2:]))
```

## Failure Modes and Diagnostics

1. **Scheduler steps on batches while optimizer updates are accumulated:** effective decay runs too fast. Log optimizer-step count and align scheduler advancement.
2. **Warmup/decay endpoint mismatch:** rate is still high at the final step or collapses early. Plot η against actual update count before a long run.
3. **Peak rate too high:** loss spikes or becomes non-finite. Run a short learning-rate range test and inspect gradient norms and loss.
4. **Plateau reduction reacts to metric noise:** rates shrink repeatedly without meaningful progress. Increase patience/minimum delta or use a predetermined schedule.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why decay learning rate? | Smaller later steps reduce overshooting and refine the solution after rapid early progress. |
| Why warm up? | Early updates in large or deep models can be unstable; ramping reduces abrupt parameter movement. |
| Debug follow-up: training stagnates after adding accumulation. | Check whether scheduler steps per microbatch instead of per optimizer update and verify total-step calculation. |

## Related Concepts

- [Optimization](optimization.md), [momentum](momentum-and-acceleration.md), and [gradient accumulation](gradient-accumulation.md)
- [Matching notebook](../notebooks/19-learning-rate-schedules.ipynb)
- [Interview practice: gradient descent variants](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Loshchilov and Hutter, [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983).
- Goyal et al., [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
