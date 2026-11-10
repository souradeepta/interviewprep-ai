# Mixed-Precision Training

**TL;DR:** Mixed-precision training uses lower-precision arithmetic for bandwidth- and compute-heavy operations while retaining enough precision in sensitive calculations and parameter updates.

## Prerequisites

Prerequisites: floating-point representation, [neural networks](neural-networks.md), and gradient descent. You can distinguish FP32, FP16, and BF16 behavior; explain autocasting and loss scaling; estimate memory and throughput trade-offs; and diagnose overflow or underflow. This ML lesson focuses on numeric behavior and training stability; the broader AI track covers system context.

## Learning Outcomes

- Explain the central assumptions and mechanism of mixed precision training.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Floating-point formats trade range and precision for storage and arithmetic speed. A short ruler marked in centimeters is lighter and faster to carry than one marked in micrometers, but it cannot resolve tiny distances. Neural-network matrix products can often tolerate coarser representation, while small gradients and accumulated optimizer state may not. Mixed precision uses the compact ruler where it is adequate and a more precise one for sensitive calculations.

Lower-precision values reduce memory traffic and can enable faster hardware kernels. That does not guarantee a faster training run: operations may fall back to slower kernels, data conversion may cost time, or input loading may already be the bottleneck. Benchmark a complete training step on the target device.

One helpful way to reason about the formats is to separate range from resolution. Range answers how large or small a magnitude can be before overflow or underflow. Resolution answers how finely nearby values can be distinguished at a given magnitude. FP16 has more fraction bits than BF16, so its representable values are locally finer, but its narrower exponent range is a liability when activation magnitudes vary widely. BF16 can represent values closer to FP32's magnitude range, but the coarser significand may erase small changes. Which failure matters depends on the layer and the distribution of values it sees.

The optimizer update deserves separate attention. Many training libraries keep a full-precision copy of parameters and maintain full-precision first and second moments for Adam-like optimizers. Activations, which dominate memory for long sequences or large batches, may use a compact format while optimizer state does not. Thus the memory benefit may be large for activation-bound workloads and smaller for parameter-state-bound workloads. Gradient checkpointing can change the dominant memory category by discarding and recomputing activations, so the two techniques should be measured together instead of assuming their savings simply add.

Precision policy is also architecture-specific. Softmax, normalization statistics, exponentials, and reductions can be sensitive because they aggregate many terms or magnify small denominators. A framework may automatically accumulate some operations in FP32 even when inputs are low precision. Developers should learn the framework's policy and inspect the actual operation dtypes when a run behaves unexpectedly. Manually casting every tensor down can defeat those safeguards and create instability that autocasting was designed to avoid.

## How It Works

FP32 has 8 exponent bits and 23 fraction bits; IEEE FP16 has 5 exponent and 10 fraction bits, giving fine precision but limited dynamic range; BF16 has 8 exponent and 7 fraction bits, retaining FP32-like range with coarser precision. Hardware and library details matter, so these bit counts do not themselves predict end-to-end speed. Autocasting chooses operation dtypes: matrix multiplications may use low precision while reductions, normalization, or loss calculations use FP32.

In FP16 training, loss scaling multiplies loss by S before backpropagation so small gradients are representable. Before the optimizer update, gradients are unscaled; if non-finite values appear, the update is skipped and S is reduced. Successful steps may increase S. BF16 often needs no loss scaling because of its wider exponent range, though model-specific instability remains possible. Master weights and optimizer states may stay in FP32, so total memory savings are less than the raw two-byte-versus-four-byte ratio. Gradient accumulation and distributed all-reduce must agree on when to unscale and clip.

Loss scaling must be integrated at the optimizer boundary. If a training loop accumulates gradients for several microbatches, it should keep the same scale for the accumulation window, unscale the accumulated gradient once, then perform finite checks, clipping, and the optimizer step in the library-prescribed order. In distributed training, every worker must make a consistent decision about whether a step is skipped. A rank that updates while another skips will cause model replicas to diverge. Checkpointing a run should preserve scaler state alongside optimizer and scheduler state so resumed training does not unexpectedly jump to an unsuitable scale.

Mixed precision also changes reproducibility details. Different kernels can use different reduction orders, and floating-point addition is not associative, so bitwise equality with an FP32 run is not a realistic expectation. The useful question is whether repeated runs remain within an acceptable quality and stability range. Record device type, framework version, precision mode, scaler policy, seed, and global batch size. Compare curves and task-level metrics over more than one seed when a small score difference would change a product or research decision.

## Trade-offs and Selection Boundary

| Choice | Strength | Boundary |
|---|---|---|
| FP32 | Broad numerical stability and compatibility | Higher memory and often lower accelerator throughput |
| FP16 + dynamic loss scale | Fast on supported accelerators; lower activation/matmul memory | Narrow exponent range can overflow/underflow |
| BF16 | Wide range often simplifies scaling | Coarser mantissa can affect sensitive computations |
| Selective FP32 ops | Protects reductions and numerically sensitive layers | Requires framework support and profiling |

Use mixed precision when supported kernels improve measured throughput or allow a useful larger batch/model. Compare convergence, final task metric, and peak memory against FP32. Keep numerically sensitive reductions in higher precision when needed; changing dtype is not a replacement for stable loss formulas or sensible initialization.

## Implementation

This dependency-light simulation illustrates why scaling preserves tiny values in a simplified finite-resolution arithmetic; it is not a hardware emulator.

```python
def quantize_step(x, step):
    return round(x / step) * step

tiny_gradient = 0.00004
coarse = quantize_step(tiny_gradient, 0.001)
scale = 1024
scaled_then_unscaled = quantize_step(tiny_gradient * scale, 0.001) / scale
assert coarse == 0.0
assert scaled_then_unscaled != 0.0

def finite_all(values):
    return all(abs(v) < float("inf") for v in values)
assert finite_all([0.1, -3.0]) and not finite_all([float("inf")])
```

## Failure Modes and Diagnostics

1. **Overflowing activations or gradients:** loss scaler repeatedly backs off or steps are skipped. Inspect non-finite tensor locations, learning rate, initialization, and operation dtype.
2. **Underflowed small gradients:** learning stalls in FP16. Verify dynamic loss scaling, compare BF16/FP32, and log gradient norms before and after unscaling.
3. **Clipping scaled gradients:** clipping before unscale makes the threshold meaningless. Unscale first, clip, then step according to framework guidance.
4. **No actual speedup:** utilization is unchanged or lower. Profile kernel dtype coverage, conversion overhead, input pipeline, and synchronization on the intended hardware.
5. **Silent quality regression:** training loss looks normal but final score degrades. Compare fixed-seed FP32 and mixed runs on the same split and inspect sensitive layers and calibration.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why use mixed precision? | Lower precision can improve compute throughput and reduce activation/matmul memory while selected calculations retain precision. |
| Why loss scaling? | Multiplying loss makes tiny FP16 gradients representable; gradients are unscaled before the update. |
| Debug follow-up: FP16 training skips many updates. | Inspect overflow source, lower scale or learning rate, consider BF16/FP32 for sensitive ops, and check unscale/clip order. |

## Related Concepts

- [Gradient accumulation](gradient-accumulation.md), [distributed training](distributed-training.md), and [weight initialization](weight-initialization.md)
- [Matching notebook](../notebooks/22-mixed-precision.ipynb)
- [Interview practice: gradient descent](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740).
- NVIDIA, [Automatic Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html).
