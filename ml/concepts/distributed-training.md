# Distributed Training

**TL;DR:** Distributed training partitions model computation or data across devices and synchronizes updates, reducing wall time or fitting models that exceed one device.

## Prerequisites

Prerequisites: [optimization](optimization.md), minibatch gradients, and basic parallelism. You can distinguish data and model parallelism, derive the global batch gradient, estimate communication costs, and diagnose scaling inefficiency or divergent replicas. This ML lesson covers training-system choices; the MLOps track extends the operational lifecycle.

## Learning Outcomes

- Explain the central assumptions and mechanism of distributed training.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Data parallelism is like giving several workers different batches of examples, then averaging their gradient suggestions before making one shared update. If each worker computes an unbiased gradient, averaging gives the gradient of the combined batch. Model parallelism splits a single large computation across devices because one device cannot hold or process the model. Both move tensors over a network, and communication can erase the benefit of extra hardware.

## How It Works

With N workers each processing local batch b, synchronous data parallelism averages g=(1/N)Σᵢgᵢ and applies one optimizer step using global batch B=Nb. An all-reduce communicates gradients; ring all-reduce is bandwidth-efficient but costs roughly proportional to gradient size and grows with worker count. Strong scaling holds total work fixed and adds devices; weak scaling increases work with devices. As synchronization cost dominates, throughput gains flatten. Larger global batches may require learning-rate retuning and affect generalization.

Model/tensor parallelism splits layers, matrices, or attention heads. Pipeline parallelism assigns layer stages to devices, but bubbles occur when stages wait; microbatches improve utilization. ZeRO-style sharding reduces replicated optimizer state, gradients, or parameters but adds communication and complexity. Checkpointing trades recomputation for activation memory. Failure recovery requires consistent optimizer and random-number state, not only model weights.

Data-parallel correctness depends on sampler and reduction semantics. Each worker should receive a distinct shard for a synchronized step, and the collective should average gradients consistently. If the final batch sizes differ, equal averaging across workers may overweight examples on a smaller worker batch; drop or pad the final batch, or use sample-weighted reduction. Distributed samplers may repeat examples to equalize worker step counts, so evaluation metrics must avoid counting padded duplicates. These details can create subtle metric shifts even when the model implementation is unchanged.

The strongest scaling opportunities occur when computation per synchronized update is large relative to communication. Model size, accelerator interconnect, gradient tensor count, and batch size all affect that ratio. Gradient bucketing overlaps communication with backpropagation, but many tiny tensors or synchronization barriers can limit overlap. Input decoding and augmentation may become the bottleneck after model compute speeds up. Measure step-time components and accelerator utilization before choosing a parallel strategy; adding workers without a bottleneck model can increase cost while lowering throughput.

Global batch changes optimization. If worker count grows while local batch remains fixed, global batch grows and there are fewer updates per epoch. Large-batch training may need a learning-rate rule, warmup, or longer schedule, but linear scaling is not universally safe. Compare models by examples seen and total compute, and check quality at fixed wall-clock budget as a distinct objective. Gradient accumulation changes update frequency but does not increase local batch statistics for layers such as batch normalization.

Checkpointing and restart behavior become more important in long distributed jobs. A valid resume checkpoint includes model parameters, optimizer moments, scheduler position, loss-scaler state, random seeds/state, data sampler position, and any gradient accumulation buffer if saving mid-update. If a process fails during an all-reduce or checkpoint write, all workers must agree on a consistent recovery point. Validate recovery by terminating a small run and confirming that resumed training follows the expected update trajectory within floating-point tolerance.

Communication topology can favor different schemes. Tensor parallelism communicates within layers and often requires fast local links; pipeline parallelism sends activations between stage devices and needs enough microbatches to fill the pipeline; data parallelism communicates gradients after backward. Hybrid parallelism combines them. Choose based on which resource is constrained—model memory, compute, or synchronization—and estimate aggregate communication before scaling a production training job.

## Trade-offs and Selection Boundary

| Strategy | Strength | Boundary |
|---|---|---|
| Data parallel | Simple and effective when model fits on each device | Replicates model/optimizer state; sync can dominate |
| Tensor/model parallel | Fits large layers or models | Frequent communication and placement constraints |
| Pipeline parallel | Distributes deep networks by layer | Pipeline bubbles and scheduling complexity |
| State sharding | Reduces per-device optimizer/model memory | More communication and checkpoint complexity |

Start with one device and a measured baseline. Add data parallelism when compute dominates communication; shard model state when memory is the limiting resource. Profile before scaling and compare throughput per dollar, not device count alone.

## Implementation

```python
def mean_gradients(worker_grads):
    if not worker_grads: raise ValueError("at least one worker required")
    width = len(worker_grads[0])
    if any(len(g) != width for g in worker_grads): raise ValueError("shape mismatch")
    return [sum(g[j] for g in worker_grads)/len(worker_grads) for j in range(width)]

g1, g2 = [1.0, 3.0], [3.0, 1.0]
global_gradient = mean_gradients([g1, g2])
assert global_gradient == [2.0, 2.0]
theta = [0.0, 0.0]; lr = .1
updated = [w-lr*g for w,g in zip(theta, global_gradient)]
assert updated == [-.2, -.2]
```

## Failure Modes and Diagnostics

1. **Poor scaling efficiency:** utilization falls as workers increase. Profile all-reduce time, input pipeline, and step time; compare communication to compute.
2. **Wrong gradient averaging:** sum instead of mean silently multiplies effective learning rate by worker count. Validate one update against a single-device concatenated batch.
3. **Unequal local batches:** naive worker averaging weights workers equally, not examples equally. Aggregate sample-weighted gradients or ensure equal batch sizes.
4. **Non-reproducible resume:** model weights load but optimizer/scheduler/RNG state is absent. Test checkpoint-resume equivalence at a fixed step.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Data versus model parallelism? | Data parallel replicates model and partitions examples; model parallel partitions a model that is too large for one device. |
| Why does adding GPUs stop helping? | Communication, input, synchronization, or memory bottlenecks dominate compute. |
| Debug follow-up: two GPUs learn differently than one. | Check global batch and gradient normalization, optimizer step counts, sampler sharding, and random state. |

## Related Concepts

- [Gradient accumulation](gradient-accumulation.md), [mixed precision](mixed-precision-training.md), and [learning-rate schedules](learning-rate-schedules.md)
- [Matching notebook](../notebooks/08-distributed-training.ipynb)
- [Interview practice: gradient descent](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Goyal et al., [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677).
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
