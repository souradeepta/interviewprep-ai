# Model Compression

**TL;DR:** Model compression reduces storage, memory use, or inference cost while preserving task quality within a measured deployment tolerance.

## Prerequisites

Prerequisites: neural-network inference, latency measurement, and model evaluation. By the end, you can distinguish parameter count from runtime cost, compare quantization, pruning, distillation, and factorization, define a compression objective, and design a target-hardware benchmark. This ML note focuses on methods and decision boundaries; [Modern AI inference lessons](../../modern-ai/README.md) address system-level serving considerations.

## Learning Outcomes

- Explain the central assumptions and mechanism of model compression.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A large model may carry redundant weights or use arithmetic more precise than a task requires. Compression is the process of removing or simplifying that excess while preserving behavior that matters. A smaller file is not necessarily a faster model: a sparse matrix may store fewer numbers but run slowly if hardware cannot exploit sparsity; a low-bit model may incur conversion overhead or unsupported kernels. Measure the quantity the deployment actually constrains.

Compression is a constrained engineering optimization. The candidate is acceptable only if it meets a quality floor on the intended population while satisfying memory, throughput, latency, energy, or bandwidth limits. A useful comparison is a Pareto frontier: a model is dominated when another option is at least as accurate and no more costly on every relevant axis.

## How It Works

Quantization maps floating-point values to fewer bits, often with scale and zero-point parameters. It lowers weight and sometimes activation bandwidth, with error depending on range, distribution, granularity, and kernels. [Pruning](pruning.md) sets selected weights or structures to zero. Unstructured zeros need sparse kernels to accelerate execution; structured channel or block removal can shrink dense operations but may need fine-tuning. [Distillation](knowledge-distillation.md) trains a smaller student to approximate a teacher, often using softened outputs. Low-rank factorization replaces a matrix W∈Rᵐˣⁿ by factors U∈Rᵐˣʳ and V∈Rʳˣⁿ when r is below min(m,n), reducing parameters and multiply-adds at the cost of extra operations and approximation error.

Methods can be combined, but order matters. Quantization-aware training exposes the model to simulated rounding during training; post-training quantization is simpler but can fail for sensitive activations. Pruning followed by fine-tuning can recover quality. Distillation can reduce a model family or architecture, while quantization can further reduce its precision. Any combination should be benchmarked after export because graph transformations and runtime kernels determine actual execution.

The metric for compression is multidimensional. Parameter count predicts checkpoint size only when storage format is known. Peak memory includes activations, temporary workspace, optimizer state during training, and runtime caches during generation. Latency depends on batch size, input length, sequence length, hardware, concurrency, and warmup. Throughput and tail latency can move in opposite directions. Measure end-to-end preprocessing and postprocessing if those stages are on the serving path.

Compression experiments should begin with an explicit acceptance envelope. For example, “no more than a one-point reduction in macro-F1 on the labeled holdout, while p95 single-request latency stays under a specified budget and peak device memory falls below a deployment limit” is testable. “Make the model smaller” is underspecified because it might refer to serialized bytes, parameter count, resident memory, or network transfer size. If the service batches requests, measure both low-load and saturated operation because batching can mask the per-request latency cost. Tail behavior matters when long inputs or unusual examples consume more compute than the median.

Quality comparisons also require uncertainty and slice coverage. A tiny validation set can make two compressed candidates appear separated by a score difference that is sampling noise. Reuse the same held-out examples for paired comparison, calculate per-example outcome changes, and examine rare classes or critical groups. Quantization may disproportionately affect values near a decision threshold; pruning may remove a feature detector used by one rare category. An acceptable average result does not imply every operationally important slice remains acceptable.

Artifact lifecycle is another part of the choice. Conversion can alter graph structure, operator names, and supported hardware, while quantization scales or sparse metadata must be stored with weights. Record the source checkpoint hash, conversion code and configuration, calibration sample identifiers, runtime version, target device, and exported artifact hash. This makes it possible to reproduce a result and roll back when a runtime upgrade changes kernel behavior. Compression is complete only when the deployed artifact has been measured and can be rebuilt.

## Trade-offs and Selection Boundary

| Method | Main benefit | Best fit / boundary |
|---|---|---|
| Quantization | Lower weight bandwidth and memory; supported integer kernels may speed inference | Validate sensitive layers and target hardware; fewer bits do not guarantee latency gain |
| Structured pruning | Removes channels/blocks to reduce dense dimensions | Requires architecture support and often fine-tuning |
| Unstructured pruning | High sparsity with small accuracy change is possible | Useful speedup needs hardware/runtime sparse support |
| Distillation | Smaller student may preserve behavior and fit a different architecture | Requires teacher data/inference and a student training pipeline |
| Low-rank factorization | Replaces large matrices with smaller factors | Rank must be low enough to save compute; approximation may hurt |

Start with an uncompressed reference and profile where cost lies. Choose one method based on the bottleneck: weights/bandwidth suggest quantization; oversized dense layers may suit structured pruning or factorization; architecture size or capability transfer may call for distillation. Keep the compression decision separate from unrelated quality changes so the effect can be attributed.

## Implementation

```python
def symmetric_quantize(values, levels=127):
    scale = max(abs(x) for x in values) / levels
    if scale == 0: return [0]*len(values), 1.0
    q = [max(-levels, min(levels, round(x/scale))) for x in values]
    return q, scale

weights = [-1.2, -0.4, 0.0, 0.7, 1.1]
q, scale = symmetric_quantize(weights, levels=7)
restored = [x*scale for x in q]
assert len(q) == len(weights) and max(abs(a-b) for a,b in zip(weights,restored)) <= scale/2
assert len(q)*1 < len(weights)*4  # idealized int8 vs FP32 weight bytes; metadata excluded
```

## Failure Modes and Diagnostics

1. **Artifact shrinks but latency does not:** runtime falls back to dense or dequantization-heavy kernels. Profile operator-level latency on deployment hardware and inspect exported graph support.
2. **Rare subgroup quality collapses:** aggregate validation hides sensitivity to quantization or pruning. Evaluate class, demographic, sequence-length, and rare-feature slices.
3. **Benchmark uses unrealistic batch size:** larger batches improve throughput but may violate request latency SLOs. Measure the serving concurrency and batch policy, including p95/p99 latency.
4. **Quality comparison uses different preprocessing:** apparent compression loss comes from pipeline drift. Run the same examples through both complete artifacts and verify input/output parity.
5. **Compressed weights are not reproducible:** conversion settings or scales are omitted. Version the source checkpoint, calibration data, conversion config, runtime, and artifact hash.

## Interview Quick Reference

| Question | Answer |
|---|---|
| How choose compression method? | Profile the bottleneck, select a method that targets it, then evaluate quality and real serving cost on target hardware. |
| Why can pruning fail to speed a model? | Unstructured sparsity may not map to efficient sparse kernels; fewer nonzeros do not ensure lower wall-clock time. |
| Design follow-up: need 30% lower p99 under a fixed accuracy floor. | Establish an end-to-end baseline, sweep supported compression options, test the production request mix, and choose a Pareto-feasible artifact. |

## Related Concepts

- [Quantization](quantization.md), [pruning](pruning.md), and [knowledge distillation](knowledge-distillation.md)
- [Matching notebook](../notebooks/23-model-compression.ipynb)
- [Interview practice: model compression](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Cheng et al., [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/abs/1710.09282).
- Han, Mao, and Dally, [Deep Compression](https://arxiv.org/abs/1510.00149).
