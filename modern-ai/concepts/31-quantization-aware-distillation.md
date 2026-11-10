# Quantization and Distillation Trade-offs

**TL;DR:** Quantization reduces numeric precision and storage; distillation teaches a smaller student to approximate a teacher. Their errors can compound, so compare quality and resource changes together.

## Prerequisites

Neural network inference and [model evaluation](01-llm-evaluation-harness.md).

## Learning Outcomes

- Estimate nominal weight storage from parameter count and bit width.
- Separate quantization error from student-to-teacher mismatch in an ablation.
- Compare calibration and quantization-aware training assumptions.
- Explain why a scalar teaching demo cannot establish model quality or latency.

## Core Intuition

Quantization is like recording measurements with fewer decimal places. Distillation is like teaching a smaller apprentice from a specialist's answers. Either can reduce deployment cost, but rounding a teacher and then approximating it with a student introduces distinct error sources. Measure each separately before deciding that a smaller artifact is acceptable.

## How It Works

Uniform affine quantization maps a floating value x into an integer grid using scale s and zero point z, approximately \(q=round(x/s)+z\), then dequantizes to \(\hat{x}=s(q-z)\). Symmetric and asymmetric mappings, clipping, per-channel scales, calibration data and hardware kernels all matter. Nominal storage for N values at b bits is Nb/8 bytes, but packed storage has metadata and alignment overhead.

Distillation trains student parameters to match teacher labels, logits or intermediate features. A temperature-softened distribution can transfer relative class probabilities; for language models, token-level KL or sequence objectives are common. Quantization-aware training simulates quantization during training so updates can adapt to rounding. Post-training quantization instead calibrates a trained model without full retraining. Combining approaches may help or hurt depending on calibration and task distributions.

Quantization error depends on the mapping granularity. A single scale for an entire tensor is easy to store but can waste levels when a few outliers set a wide range. Per-channel or group-wise scales fit local ranges better and add metadata. Symmetric quantization is convenient when values center near zero; asymmetric quantization can use the representable range more efficiently when activations have an offset. Clipping removes extreme values to protect resolution for the bulk, but those extremes may encode rare, important features. Calibration should include representative prompt and activation distributions, not only convenient public examples.

Distillation and quantization answer different questions. A student can learn a compact architecture from a high-quality teacher, whereas quantization keeps the architecture and changes numeric representation. To understand combined behavior, compare at least teacher FP, teacher quantized, student FP and student quantized. This four-way design separates architecture loss, rounding loss and any interaction. Evaluate with the same tokenizer, decoding policy, prompt suite and serving constraints. If only the final compressed model is measured, a regression cannot be attributed to its source.

The loss used for distillation determines what knowledge is transferred. Hard labels teach the selected output; soft teacher probabilities convey the relative ordering of alternatives. A higher temperature makes those probabilities less peaked and emphasizes dark knowledge, but the loss scale and gradient magnitude change too. Intermediate hidden-state matching can preserve representations yet requires corresponding layers and dimensions. For generative tasks, teacher-forced token losses do not fully capture errors after the student's own earlier token choices, so sequence-level or rollout evaluation may be needed.

Hardware is part of the method. A nominal four-bit checkpoint may unpack to a wider representation if the accelerator lacks a native low-bit kernel. Group scales, alignment and dequantization can dominate memory traffic. Similarly, structured sparsity helps only if the runtime uses a supported sparse kernel. Report serialized size, peak resident memory, throughput and tail latency separately; “compressed” is a storage statement until execution is measured.

Use staged acceptance gates. First check numerical integrity and layer-wise error; next validate language or task quality on held-out sets; then benchmark the target engine with the intended batch and context length. Add safety, multilingual, long-context and rare-token slices when those are in scope. A small aggregate loss may still damage a narrow but critical skill. Keep the uncompressed artifact and a rollback path until shadow evaluation and canary monitoring show stable behavior.

## Trade-offs

| Method | Resource change | Main risk |
|---|---|---|
| Lower-bit post-training quantization | Smaller weights, potentially faster supported kernels | Outlier/clipping error and hardware dependence |
| Quantization-aware training | Learns around simulated rounding | Extra training complexity and simulation mismatch |
| Distillation | Smaller student architecture | Capacity and teacher-label limitations |
| Quantized distilled student | Compounded compression | Need separate quality ablations to attribute error |

## Implementation with Python

Run `python3 modern-ai/implementations/31-quantization-aware-distillation.py`. A fixed scalar vector compares bit-grid approximation, student mismatch and nominal bytes. It does not measure model perplexity or latency; those require real weights, representative evaluation and a target runtime.

```python
import runpy
m = runpy.run_path("modern-ai/implementations/31-quantization-aware-distillation.py")
quantize = m.get("quantize")
quantized = quantize(list((-.8, -.1, .2, .9)), bits=4)
assert len(quantized) == 4
```

## Failure Modes

| Failure | Diagnostic | Mitigation |
|---|---|---|
| Rare outliers saturate | Per-channel error and activation-range histograms | Use robust calibration or finer-grained scales |
| Aggregate metric hides task regressions | Slice-level evaluation and error analysis | Define task-critical slices before compression |
| Smaller file fails to improve latency | Measure target kernel throughput and memory behavior | Benchmark actual deployment hardware and batch shape |

- Outliers saturate the grid: inspect per-layer and per-channel range/error histograms, then use appropriate scale granularity or clipping.
- Average quality hides a critical regression: evaluate task and safety slices separately and gate on predeclared thresholds.
- Storage shrinks but runtime does not: profile native low-bit kernel support, dequantization and memory traffic on the target engine.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| What differs between quantization and distillation? | Numeric representation versus learned smaller model; ask how to attribute error in a combined system. |
| Why can a 4-bit model not be exactly one quarter the size of FP16? | Scales, zero points, packing and unquantized components add overhead; ask how you measure actual footprint. |
| What is calibration data for? | It estimates ranges or activation behavior; follow up on distribution shift. |

## Related Concepts

See [mixed-bit quantization](42-mixed-bit-quantization.md), [ML model compression](../../ml/concepts/model-compression.md), [notebook 31](../notebooks/31-quantization-aware-distillation.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#quantization-and-distillation).

Compression comparisons need a precise deployment target. A server with large batches may amortize dequantization differently from a mobile device limited by memory. Include expected concurrency, maximum context, generation cap and power constraints in the experiment. Warm up the runtime, repeat measurements to report variability, and separate prompt processing from token generation when their kernels behave differently. These measurements complement model quality; one cannot substitute for the other.

## References

[LLM.int8()](https://arxiv.org/abs/2208.07339), [GPTQ](https://arxiv.org/abs/2210.17323), and [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
