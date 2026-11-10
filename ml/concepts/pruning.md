# Pruning

**TL;DR:** Pruning removes selected weights, channels, or structures to reduce model size or compute, then often fine-tunes to recover accuracy.

## Prerequisites

Prerequisites: [neural networks](neural-networks.md), [regularization](regularization.md), and deployment profiling. You can distinguish unstructured from structured pruning, compare magnitude and sensitivity-based criteria, design a prune/fine-tune schedule, and explain why sparsity does not automatically accelerate inference. This ML lesson covers pruning mechanics; [model compression](model-compression.md) compares pruning with other compression options.

## Learning Outcomes

- Explain the central assumptions and mechanism of pruning.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A trained model may contain parameters whose removal barely changes predictions. Pruning tests this redundancy by removing low-value connections or groups and measuring the resulting quality loss. It resembles simplifying a map by deleting details that do not affect navigation at the required scale. However, low weight magnitude alone does not mean a connection is unimportant: the contribution depends on its inputs, interactions, and downstream pathways.

Pruning is a trade between representation capacity and resource use. A sparse checkpoint can be smaller when zeros are encoded efficiently, but dense hardware may still multiply by zero. The deployment runtime must support the sparsity pattern to turn parameter removal into lower latency or energy.

## How It Works

Unstructured pruning masks individual weights. Magnitude pruning removes weights with small |w|, globally or per layer. Structured pruning removes groups such as channels, attention heads, blocks, or filters, producing smaller dense tensors that conventional kernels can exploit. One-shot pruning removes a fraction at once; iterative pruning alternates removal and fine-tuning. Lottery-ticket-style approaches rewind surviving parameters to earlier values, while gradual schedules can let the model adapt as sparsity increases.

Let mask mⱼ∈{0,1}; the pruned parameter is w′ⱼ=mⱼwⱼ. Sparsity is the fraction of masked parameters, but it is not a complete efficiency metric. Unstructured sparse storage also includes index metadata and irregular memory access. For structured channel pruning, removing input/output channels can shrink following convolution operations, but residual connections, normalization, and grouped convolutions impose dependency constraints. A calibration set or validation set evaluates damage; final test data should remain untouched.

Pruning may be viewed as capacity control, but it differs from ordinary [regularization](regularization.md): it imposes a discrete topology change. Fine-tuning after pruning can restore quality but consumes additional compute. The pruning criterion and schedule should be selected based on layer sensitivity and target hardware rather than a global sparsity target alone.

Importance estimates are imperfect because parameters interact. Magnitude-based pruning is cheap and often surprisingly effective, but it assumes small values contribute little. First-order sensitivity methods estimate a loss change from removing a parameter using its weight and gradient; second-order approximations account for curvature but are more expensive. Activation-aware approaches consider how frequently a weight participates in actual inputs. These signals can guide a per-layer allocation, yet none substitutes for evaluation on representative held-out data. A practical experiment starts with a global baseline, plots quality against sparsity by layer, and then tests whether reallocating a fixed sparsity budget helps.

Pruning schedules must also be reproducible. Record whether the threshold is global or layerwise, whether exact zeros are stored sparsely, whether masks can regrow, which checkpoint is fine-tuned, and how many recovery updates are allowed. If structured pruning changes tensor dimensions, export and downstream layers must use the transformed architecture; a mask alone may leave all dense operations intact. Keep an unpruned artifact to serve as a rollback and comparison point.

## Trade-offs and Selection Boundary

| Method | Advantage | Boundary |
|---|---|---|
| Unstructured magnitude | Simple and can reach high parameter sparsity | Requires sparse kernels and metadata-aware format for speed/storage benefit |
| Structured channel/filter | Shrinks dense tensor shapes and often maps to regular kernels | May damage representation sharply; dependencies complicate removal |
| Block sparsity | Regular pattern can exploit specialized accelerators | Hardware-supported block sizes constrain pattern |
| Gradual iterative pruning | Allows recovery between pruning steps | Adds training and schedule-selection cost |
| Sensitivity-based pruning | Allocates pruning according to quality impact | Requires extra evaluations and reliable importance estimates |

Prefer structured pruning when latency on ordinary dense kernels is the goal. Use unstructured sparsity when storage or a supported sparse accelerator is the actual target. If a model is already small or its cost comes from runtime overhead rather than matrix operations, pruning is unlikely to solve the bottleneck. Compare with quantization and distillation under the same quality and resource constraints.

## Implementation

```python
def magnitude_prune(weights, fraction):
    if not 0 <= fraction < 1: raise ValueError("fraction must be in [0, 1)")
    count = int(len(weights)*fraction)
    threshold = sorted(abs(w) for w in weights)[count-1] if count else -1
    mask = [abs(w) > threshold for w in weights]
    # Deterministic tie handling can prune more than requested at equal magnitudes.
    return [w if keep else 0.0 for w,keep in zip(weights,mask)], mask

w = [0.01, -0.2, 0.4, 0.8]
pruned, mask = magnitude_prune(w, .25)
assert pruned[0] == 0.0 and pruned[2:] == w[2:]
assert sum(v == 0 for v in pruned) >= 1
```

## Failure Modes and Diagnostics

1. **No latency reduction despite high sparsity:** deployed kernels remain dense or sparse indices dominate. Profile the target runtime and inspect whether its kernel supports the exact pattern.
2. **One layer loses most quality:** global threshold removes a sensitive layer's weights disproportionately. Plot per-layer sparsity and ablate pruning by layer or use sensitivity-aware budgets.
3. **Fine-tuning does not recover:** learning rate, schedule, or data differs from original training. Compare unpruned and pruned recovery with identical data and monitor per-class errors.
4. **Reported model size excludes metadata:** sparse representation is larger than dense storage. Measure serialized artifact bytes including indices, scales, and runtime wrappers.
5. **Validation overfit during pruning search:** repeated sparsity choices adapt to the same set. Keep a final untouched test and use cross-validation for small datasets.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why doesn't 90% sparsity guarantee 10× speedup? | Dense kernels may still process zeros, and sparse indexing/access has overhead. |
| Structured versus unstructured pruning? | Unstructured removes individual weights; structured removes groups that shrink dense operations more directly. |
| Design follow-up: prune a CNN for mobile. | Profile layers, test channel/block pruning supported by mobile kernels, fine-tune, and benchmark end-to-end on the device. |

## Related Concepts

- [Model compression](model-compression.md), [quantization](quantization.md), and [knowledge distillation](knowledge-distillation.md)
- [Matching notebook](../notebooks/31-pruning.ipynb)
- [Interview practice: model compression](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Han et al., [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626).
- Liu et al., [Network Slimming](https://arxiv.org/abs/1708.06519).
