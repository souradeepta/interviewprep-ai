# Gradient Accumulation

**TL;DR:** Gradient accumulation sums or averages gradients across microbatches before an optimizer step, approximating a larger batch when device memory is limited.

## Prerequisites

Prerequisites: [optimization](optimization.md), minibatch gradients, and [distributed training](distributed-training.md). You can derive the effective batch size, correctly scale loss across microbatches, distinguish accumulation from more data-parallel devices, and align optimizer/scheduler steps. This ML lesson concerns memory-constrained training; the AI overview discusses training concepts more broadly.

## Learning Outcomes

- Explain the central assumptions and mechanism of gradient accumulation.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

One device may not fit a batch of 64 examples. Instead, process four batches of 16, keep each batch's gradient, add them together, and update weights once. The optimizer then sees an approximation to the gradient of all 64 examples. Updating after every small batch is different: weights change between microbatches, so gradients no longer correspond to one large batch.

Accumulation saves activation memory but does not reduce the total forward/backward arithmetic. It also does not automatically improve training or make normalization behave as if the full batch were present.

## How It Works

If microbatch i has loss Lᵢ averaged over b examples, the large-batch objective is L=(1/K)ΣᵢLᵢ for K equal-sized microbatches. Backpropagate Lᵢ/K (or sum example losses then divide by total examples) and call optimizer.step only after K microbatches. Effective batch B=bK, or bK·N for N synchronized data-parallel workers. For unequal microbatch sizes, weight each mean loss by bᵢ/Σbᵢ. Clear gradients after each optimizer update.

Memory for activations roughly follows microbatch size; optimizer state and parameters remain unchanged. Batch normalization still sees each microbatch's statistics, so gradients may differ from a physically large batch. Schedulers, weight decay, gradient clipping, and EMA updates should generally advance once per optimizer step. Mixed precision may require scaler state to persist across accumulation steps.

Loss scaling across an accumulation window is easy to get wrong. If each microbatch loss is already a mean over its examples, divide by the number of microbatches only when they are equally sized. If the final microbatch is smaller, weight its mean by its actual sample count before dividing by the total. Alternatively, use a summed loss and divide the accumulated gradient by total examples. This ensures the result corresponds to the average gradient of the effective batch. A reference implementation can concatenate a tiny set of examples and compare its single-batch gradient with accumulated gradients to verify semantics.

The accumulation boundary defines an optimizer update. Optimizer moments, decoupled weight decay, learning-rate scheduler, gradient scaler growth tracker, and exponential moving average should generally be updated once at that boundary, not once per microbatch. Gradient clipping should operate on the accumulated, correctly averaged gradient; clipping each microbatch separately is a different algorithm because several individually clipped gradients can still sum to a large vector. Distributed training often all-reduces each microbatch gradient, so communication can increase unless synchronization is delayed safely.

Accumulation affects batch-dependent layers and randomness. Batch normalization computes statistics from each microbatch rather than the accumulated set, so the effective batch is not physically equivalent. Dropout samples a separate mask per microbatch, which also changes stochastic details relative to a single forward pass. Data augmentation is applied independently. These differences are usually acceptable if the goal is a larger gradient batch, but they should be considered when exact equivalence matters. Group norm or layer norm avoids batch-statistic mismatch for small microbatches.

There are performance limits. Accumulation reduces activation memory, but processes the microbatches sequentially and may add Python or kernel-launch overhead. It does not reduce parameter or optimizer-state memory, nor does it increase device parallelism. If each microbatch is too small to use the accelerator efficiently, throughput may fall. Profile memory and step time, and compare accumulation with activation checkpointing, smaller sequence length, or distributed training. The right option depends on whether memory is consumed by activations, optimizer state, or model parameters.

## Trade-offs and Selection Boundary

| Approach | Benefit | Boundary |
|---|---|---|
| Gradient accumulation | Larger effective batch without larger activation memory | More serial microbatch work; batch norm differs |
| Data parallelism | Adds throughput and aggregate batch capacity | Requires devices and gradient communication |
| Smaller batch | Lower memory and more frequent updates | Different optimization noise and learning-rate needs |
| Checkpointing | Reduces activation memory by recomputing | Adds compute and implementation complexity |

Use accumulation when a target batch improves optimization behavior but a device cannot fit it. For latency/throughput, accumulation may reduce update frequency without increasing examples/second; profile. Do not increase effective batch without retuning learning rate or schedule assumptions.

## Implementation

```python
def accumulated_gradient(microbatch_means, sizes):
    total = sum(sizes)
    width = len(microbatch_means[0])
    return [sum(size*g[j] for size,g in zip(sizes,microbatch_means))/total
            for j in range(width)]

grads = [[1., 3.], [3., 1.], [5., 2.]]
sizes = [2, 2, 1]  # last microbatch is smaller
g = accumulated_gradient(grads, sizes)
assert g == [2.6, 2.0]
# Equal-sized accumulation is the ordinary arithmetic mean.
assert accumulated_gradient(grads[:2], [4, 4]) == [2., 2.]
```

## Failure Modes and Diagnostics

1. **Loss not divided by accumulation count:** effective gradient and learning rate are multiplied by K. Compare one accumulated update to a concatenated batch reference.
2. **Scheduler advances each microbatch:** decay happens K times too quickly. Log microsteps and optimizer steps separately.
3. **Unequal final microbatch treated equally:** examples receive incorrect weights. Scale by actual example count.
4. **Assuming batch norm sees effective batch:** running statistics still reflect microbatches. Use normalization suited to small batches or synchronize statistics where appropriate.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Effective batch size? | Microbatch size × accumulation steps × data-parallel workers, assuming synchronized averaging. |
| Why not update weights after every microbatch? | Then later gradients are evaluated at changed parameters and no longer equal one large-batch update. |
| Debug follow-up: loss explodes after enabling accumulation. | Check loss scaling, clipping position, optimizer/scheduler step count, and distributed averaging. |

## Related Concepts

- [Distributed training](distributed-training.md), [learning-rate schedules](learning-rate-schedules.md), and [mixed precision](mixed-precision-training.md)
- [Matching notebook](../notebooks/15-gradient-accumulation.ipynb)
- [Interview practice: gradient descent](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- PyTorch, [Zeroing out gradients in PyTorch](https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html).
- Hugging Face, [Gradient accumulation](https://huggingface.co/docs/transformers/grad_accumulation).
