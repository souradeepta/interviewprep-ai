# Quantization (for LLMs)

TL;DR: LLM quantization represents weights, activations, or KV state with fewer bits to reduce memory traffic and storage, trading implementation complexity and numerical error against quality and latency.

## Prerequisites

This lesson assumes matrix multiplication, floating-point precision, and the [token optimization](29-token-optimization.md) context. It focuses on model-weight and inference-state quantization; quantization-aware training is a distinct training-time process and is not implied by loading a low-bit checkpoint.

## Learning Outcomes

- Distinguish post-training quantization from quantization-aware training and identify which tensors are quantized.
- Calculate idealized weight-storage savings from bit width while identifying omitted runtime overhead.
- Explain affine quantization and why group scales, outliers, and calibration data matter.
- Design deterministic numerical checks and task-level evaluations for a quantized model.

## Core Intuition

Suppose a model stores a weight as a 32-bit floating-point number even though inference does not need all of those bits to preserve useful behavior. Quantization maps a range of values to a smaller set of representable codes. A scale and possibly a zero point map codes back to approximate real values. Fewer bits can reduce checkpoint size and memory traffic; the actual speedup depends on kernels and hardware support. A small numeric reconstruction error can also compound through many layers or distort sensitive directions, so bit width alone does not certify quality.

Quantization is not simply “round weights to integers.” A practical scheme chooses which weights or activations to quantize, whether scaling is per tensor, channel, or group, how outliers are handled, and what format the runtime kernel consumes. Post-training quantization (PTQ) applies a scheme to an already trained checkpoint, sometimes using representative calibration inputs. Quantization-aware training (QAT) simulates quantization effects during optimization so the model can adapt, at additional training cost. Weight-only schemes can reduce model storage while leaving activations in higher precision; weight-and-activation schemes can reduce more arithmetic and bandwidth but have different calibration and kernel requirements.

## How It Works

For a tensor x with scale s > 0 and integer zero point z, a simple affine quantizer is `q = clip(round(x / s) + z, q_min, q_max)`. Dequantization is `x_hat = s * (q - z)`. For a signed symmetric range, z is often zero and s is chosen from the magnitude range. The quantization error is `e = x_hat - x`; clipping creates larger errors for values outside the representable range, while rounding error remains within approximately half a quantization step when no clipping occurs.

With b bits per value, idealized raw storage is `N*b/8` bytes for N values, before scales, zero points, alignment, and metadata. Going from 16-bit to 4-bit values suggests a fourfold reduction in raw value storage, not necessarily fourfold lower end-to-end memory or latency. Quantized matrix multiplication can unpack values or use specialized low-bit kernels. If the hardware/runtime falls back to dequantizing into a wider format, performance may be flat or worse. Smaller weights may still help when the workload is memory-bandwidth-bound.

Scaling granularity matters. Per-tensor scaling is compact but may let a few outliers set a range that wastes resolution for most values. Per-channel or group-wise scales adapt better at the cost of metadata and extra loads. Calibration inputs help estimate activation ranges for methods that quantize activations; calibration data must represent deployment behavior without leaking evaluation examples. Some algorithms reorder or transform weights to handle outliers, and some formats support asymmetric ranges. These are method-specific assumptions, not universal properties of “4-bit.”

Quality evaluation must use the actual target checkpoint, tokenizer, quantization implementation, kernel, and workload. Test not only average benchmark scores but also long-context behavior, rare tokens, numerically sensitive tasks, and output distributions. For fine-tuning, quantized base weights may be frozen while trainable low-rank adapters remain in higher precision, as in QLoRA; this is different from quantizing all trainable parameters.

## Trade-offs

| Scheme or choice | Benefit | Cost or assumption | Evaluation signal |
|---|---|---|---|
| FP16/BF16 baseline | Broad runtime support and relatively low numerical error | More weight memory than low-bit formats | Quality, speed, and memory baseline |
| 8-bit PTQ | Lower storage with often manageable error | Method and layer sensitivity still matter | Quality regression by task and outlier layer |
| 4-bit weight-only PTQ | Large idealized weight reduction | Specialized kernels/scales; activation storage remains | Peak memory and latency on supported hardware |
| Weight-and-activation quantization | Can reduce activation traffic and enable integer execution | Calibration sensitivity and additional numerical error | Calibration drift and task metrics by input domain |
| Per-group scales | Better local range fit than one tensor scale | Scale metadata and kernel complexity | Error distribution and actual kernel throughput |
| QAT | Model can adapt to simulated quantization | Training cost and reproducibility burden | Held-out quality and convergence versus PTQ |

For weight memory, a useful first estimate is `parameters * bits_per_weight / 8`. It omits embeddings or tied-weight details only if the parameter count is wrong, and it also omits scales, temporary buffers, activations, and the KV cache. Therefore, distinguish “the checkpoint fits” from “the model plus runtime state fits.” For autoregressive serving, KV memory can dominate at long contexts or high concurrency even after weight quantization.

## Implementation

This compact symmetric quantizer demonstrates rounding, clipping, dequantization, and error accounting using only the Python standard library. Its 4-bit signed code range is deliberately specified as [-7, 7] to keep a symmetric zero-centered codebook; real formats may use different ranges, group scales, or packing. The snippet does not model a production low-bit kernel.

```python
def quantize_symmetric(values, bits, scale=None):
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")
    if not values:
        raise ValueError("values must be nonempty")
    limit = 2 ** (bits - 1) - 1
    if scale is None:
        peak = max(abs(value) for value in values)
        scale = peak / limit if peak else 1.0
    if scale <= 0:
        raise ValueError("scale must be positive")
    codes = [max(-limit, min(limit, round(value / scale))) for value in values]
    restored = [code * scale for code in codes]
    return scale, codes, restored


values = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
scale, codes, restored = quantize_symmetric(values, bits=4, scale=0.25)
assert codes == [-4, -2, 0, 2, 4, 7]
assert restored[0] == -1.0 and restored[-1] == 1.75
assert restored[-1] != values[-1]  # fixed calibration range clips the outlier
assert len(codes) * 4 == 24  # ideal packed bits, excluding scale metadata
```

The example's Python integers are not actually packed into four-bit storage. For a real model, run a supported loader and kernel, record exact package and hardware versions, and compare against the same model in the reference precision. Fix the evaluation set before tuning; report per-task results and uncertainty rather than choosing only the benchmark that improved.

## Failure Modes

- **Assuming bit width equals model memory:** Scales, zero points, padding, unquantized layers, and runtime buffers add overhead. Compare measured peak device memory with the ideal `N*b/8` estimate and inspect tensor dtypes.
- **Outlier-driven clipping or scale:** A few extreme values can consume representable range. Plot error by channel/group, inspect clipping rates, and compare finer-grained scales or an outlier-aware method.
- **Calibration mismatch:** Activation ranges from one domain may not cover deployment inputs. Measure saturation on a representative held-out set and slice by domain, sequence length, and language.
- **Quality regression hidden by an aggregate score:** Quantization may disproportionately harm rare capabilities. Use a fixed task suite with per-category metrics, long-context cases, and paired examples; roll back when a critical slice regresses.
- **Slow fallback kernel:** A runtime may unpack or dequantize weights inefficiently. Profile kernel dispatch and end-to-end latency on target hardware; do not infer speed from file size.
- **Double quantization or incorrect checkpoint metadata:** Loading weights with the wrong scale layout or applying conversion twice corrupts values. Validate known tensor shapes, quantization configuration, and logits against a trusted loader.
- **Ignoring KV-cache memory:** Quantized weights can fit while concurrent requests still exceed memory due to cached tokens. Budget weights, workspace, and KV state separately under peak load.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| What does a quantizer store? | Low-bit codes plus scale/zero-point metadata that approximate original values. | Derive ideal weight storage for a given parameter count. |
| PTQ versus QAT? | PTQ converts after training, sometimes with calibration; QAT simulates quantization during training so weights adapt. | Choose one for a strict quality target and limited training budget. |
| Why can 4-bit be slower? | The runtime may lack an efficient kernel or spend time unpacking/dequantizing values. | Design a hardware-specific benchmark. |
| How do you validate quantization? | Compare a fixed suite and representative slices to a reference checkpoint, with memory and latency measured on target hardware. | Debug a regression isolated to long contexts. |

## Related Concepts

- [KV Cache](23-kv-cache.md) covers a separate inference-memory component that may dominate after weight compression.
- [Inference Optimization](28-inference-optimization.md) discusses deployment latency and throughput as system outcomes.
- [Model Compression](../../ml/concepts/model-compression.md) places quantization among broader compression methods.

## Practice

- [Quantization interview practice](../interview-prep/curriculum-practice.md#quantization-for-llms) works through fixed-scale clipping, reconstruction error, and deployment validation.
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) presents a weight-only PTQ method.

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) describes low-bit frozen weights and higher-precision adapters for fine-tuning.
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) studies activation/weight range migration for quantization.
