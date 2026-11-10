# Quantization

**TL;DR:** Quantization represents model values with fewer bits, reducing storage and often memory bandwidth, while introducing bounded or distribution-dependent numerical error.

## Prerequisites

Prerequisites: floating-point arithmetic, [neural networks](neural-networks.md), and [model compression](model-compression.md). You can derive affine quantization and dequantization, compare post-training and quantization-aware training, distinguish weight-only from activation quantization, and evaluate quality and latency on target hardware. This ML lesson covers representation and numerical effects; the Modern AI inference track discusses serving-system use cases.

## Learning Outcomes

- Explain the central assumptions and mechanism of quantization.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Imagine replacing a precise measuring instrument with one that has fewer tick marks. Most readings remain close, but small distinctions disappear and boundary cases round differently. Quantization maps a range of floating-point values to a limited set of integer levels. It can reduce model storage and the movement of weights from memory. The impact depends on how the model uses each value, how the range is chosen, and whether the runtime has efficient kernels for the chosen integer format.

Quantization is not simply “round every number to int8.” Zero-point offsets, scales, clipping ranges, per-channel variation, activation distributions, and accumulation precision all matter. A lower-bit artifact does not guarantee faster inference if conversion overhead or unsupported operators dominate.

## How It Works

An affine quantizer maps real x to integer q=clip(round(x/s)+z, qmin, qmax), where s>0 is scale and z is zero point. Dequantization approximates x̂=s(q−z). Symmetric quantization sets z=0 and balances the range around zero; asymmetric quantization can use the integer range more efficiently when values are skewed. With b bits, q has 2ᵇ discrete levels. The step size is approximately (xmax−xmin)/(2ᵇ−1); clipping values outside the chosen range introduces saturation error, while rounding introduces quantization error within the range.

Per-tensor quantization uses one scale for a tensor. Per-channel weight quantization uses separate scales, often reducing error because output channels can have different ranges, but requires runtime support. Weights are static and can be quantized once. Activations vary by input, so post-training activation quantization needs representative calibration data. Quantization-aware training inserts simulated quantize/dequantize operations during training so parameters adapt to rounding. Weights may use 8 or fewer bits while accumulations use wider integer or floating-point types to prevent overflow.

Weight-only quantization reduces parameter bandwidth, especially for large models, but may not accelerate compute-bound operations. Integer activation and weight kernels can improve throughput when hardware supports them. Mixed precision preserves higher precision for sensitive operations. Error can propagate through layers, so compare outputs and downstream metrics, not just average weight reconstruction error.

Calibration selects scales and clipping ranges from observed tensors. Min/max calibration preserves outliers but can allocate most code levels to a few extreme values; percentile or histogram-based calibration may improve resolution for common activations while clipping tails. The right range depends on the application: clipping a rare sensor spike may be acceptable noise for one task and a catastrophic miss for another. Calibration data should therefore include realistic edge cases and important subgroups, while remaining separate from the final evaluation set. If deployment input distribution shifts, the chosen activation ranges may no longer be appropriate.

Quantization granularity affects both quality and deployment complexity. Per-tensor scales minimize metadata and simplify kernels; per-channel or group scales represent heterogeneous weight ranges more accurately at the cost of extra scale loads and conversion logic. Dynamic quantization computes some activation scales at runtime, which adapts to each input but adds overhead. Static quantization calibrates ranges ahead of time and can enable faster integer kernels, but depends on representative calibration examples. Check the exact runtime's supported operators and formats before training or converting a full model.

Numerical comparison should happen at several levels. First compare reconstructed weights or layer outputs to identify large local errors. Then compare full-model logits on fixed examples, followed by task metrics and operational decisions. Similar logits do not guarantee identical decisions for examples near a threshold, and similar average metrics can conceal rare-class failures. Treat per-example disagreement as a useful diagnostic dataset rather than relying on one aggregate accuracy number.

## Trade-offs and Selection Boundary

| Method | Best fit | Boundary |
|---|---|---|
| FP16/BF16 | Hardware-supported dense operations with modest precision loss | Still uses more storage than int8; numerical range/precision differs |
| INT8 post-training | Mature accelerator support and representative calibration set | Sensitive models/layers may lose quality |
| Quantization-aware training | Tight quality tolerance justifies retraining | Requires training pipeline and simulation support |
| Weight-only low-bit | Large weight-bound models with limited memory bandwidth | May not reduce latency if compute or dequantization dominates |
| Per-channel scales | Weight channels have widely different ranges | More metadata and kernel complexity |

Try post-training quantization first when conversion simplicity matters and calibration data represents deployment. Use quantization-aware training when quality degradation exceeds tolerance or activation quantization is necessary. Compare FP32/FP16, quantized, and optionally [distilled](knowledge-distillation.md) variants under the same data and hardware conditions. If the runtime lacks suitable kernels, optimize another bottleneck instead.

## Implementation

```python
def quantize(values, bits=8, symmetric=True):
    qmin, qmax = (-(2**(bits-1)), 2**(bits-1)-1) if symmetric else (0, 2**bits-1)
    lo, hi = min(values), max(values)
    if symmetric:
        scale = max(abs(lo), abs(hi)) / qmax if max(abs(lo),abs(hi)) else 1.0
        zero = 0
    else:
        scale = (hi-lo)/(qmax-qmin) if hi != lo else 1.0
        zero = round(qmin-lo/scale)
    q = [max(qmin,min(qmax,round(x/scale)+zero)) for x in values]
    restored = [scale*(v-zero) for v in q]
    return q, restored, scale, zero

values = [-1.0, -.25, 0.0, .5, 1.0]
codes, restored, scale, zero = quantize(values, bits=8)
assert len(set(codes)) == len(values) and max(abs(a-b) for a,b in zip(values,restored)) <= scale/2
```

## Failure Modes and Diagnostics

1. **Calibration data misses rare activations:** chosen range clips production inputs. Compare activation histograms across representative traffic slices and retain rare but important cases.
2. **Outliers consume most integer levels:** ordinary values collapse into a narrow interval. Inspect per-layer ranges; use per-channel scales or justified clipping and validate tail examples.
3. **Zero point or scale mismatch after export:** outputs shift dramatically. Verify quantization parameters, signedness, rounding mode, and operator conventions end to end.
4. **Accuracy holds but latency worsens:** runtime inserts casts or lacks a fast kernel. Profile the exported graph on target hardware and inspect actual operator dtypes.
5. **Average score hides rare-class regression:** quantization affects a minority slice disproportionately. Report per-class and subgroup metrics at the chosen operating point.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What do scale and zero point do? | They map integer codes to an approximate real-valued range and align integer zero with real zero. |
| PTQ versus QAT? | Post-training quantization converts a trained model with calibration; QAT simulates quantization during training so the model can adapt. |
| Design follow-up: INT8 reduces accuracy on a small critical class. | Inspect its activation ranges and confusion slice, try per-channel or selective higher precision/QAT, and measure on representative target data. |

## Related Concepts

- [Mixed-precision training](mixed-precision-training.md), [pruning](pruning.md), and [model compression](model-compression.md)
- [Matching notebook](../notebooks/32-quantization.ipynb)
- [Interview practice: model compression](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Jacob et al., [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877).
- Nagel et al., [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295).
