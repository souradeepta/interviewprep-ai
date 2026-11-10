# Adapters
TL;DR: Adapters specialize a pretrained network by training small bottleneck modules inserted into its layers while keeping the shared base weights frozen.

## Prerequisites

Know the transformer residual stream and the basic idea of fine-tuning; [LoRA](08-lora.md) is a useful comparison. This lesson is specifically about adapter modules, rather than the broader family of parameter-efficient fine-tuning methods.

## Learning Outcomes

- Derive the approximate trainable parameter count of a bottleneck adapter from hidden size and bottleneck width.
- Distinguish adapter updates from LoRA updates and full fine-tuning.
- Choose an adapter design using measured task quality, serving constraints, and task-sharing requirements.
- Diagnose underfitting, unintended base-weight updates, or serving overhead with a concrete check.

## Core Intuition

Imagine keeping a common reference manual fixed while attaching a small, task-specific insert at selected pages. Each task gets its own insert, and all tasks reuse the manual. A transformer adapter follows this pattern: the pretrained network provides general representations, while a small learned module changes how hidden states are transformed for a particular domain or task.

This is not equivalent to placing examples in a prompt. Adapter parameters are learned during a training step and saved; at inference, the selected adapter participates in the forward pass. It is also distinct from LoRA: LoRA adds a low-rank update to selected weight matrices, while a bottleneck adapter adds a small neural transformation in the network's layer computation. Both can reduce per-task trainable parameters, but their runtime and composition behavior differ.

Adapters are useful when many related tasks share an expensive base model, such as separate support-intent models for different products. They are not automatically cheaper end to end. The shared base still has to be loaded, each adapter needs storage and evaluation, and the inserted computation can add latency. The benefit depends on sharing the base across tasks and on the adapter-enabled runtime supporting the chosen architecture.

## How It Works

For a hidden vector `h` with width `d`, a common bottleneck adapter computes a down projection, nonlinearity, and up projection, then adds the result to the residual stream:

```text
z = h + W_up f(W_down h + b_down) + b_up
```

Here `W_down` maps `d` dimensions to bottleneck width `r`, `W_up` maps `r` back to `d`, and `f` is an activation such as ReLU. The residual connection preserves a direct path for the original representation. Biases and normalization details vary by implementation. In a simple two-linear-layer module, the trainable count is approximately `2dr + d + r`; omitting biases gives `2dr`. For example, `d=768` and `r=32` gives 49,952 parameters with biases, per adapter module. A transformer with modules at multiple locations multiplies that count accordingly.

The original Houlsby design inserts adapters around transformer sublayers; other adapter families change placement, activation, normalization, or parameterization. Thus “an adapter” does not uniquely imply one topology or one parameter percentage. The implementation and configuration determine the actual parameter count and compute. A reliable report counts trainable parameters from the instantiated model rather than quoting a rule of thumb.

During training, the base parameters are frozen and gradients update adapter parameters (and any explicitly chosen task head). The optimizer sees only the trainable set. At inference, the application loads the base and the matching adapter, then routes a request to the appropriate task configuration. Storing a shared base once can reduce duplicated checkpoint storage, while adapter switching requires careful isolation so that requests do not accidentally use another task's weights.

Adapter composition is architecture-dependent. Sequential composition applies one module after another and can increase depth and runtime. Parallel or fusion methods combine modules differently and may require a trained composition mechanism. Do not assume that arbitrary independently trained adapters can be stacked without validation: they may conflict, and order can affect outputs. For a single task, compare the adapter against a full fine-tuning or LoRA baseline on the same held-out split, training budget, and deployment hardware.

## Trade-offs

| Choice | Trainable state | Inference behavior | Strong fit | Main cost or risk |
|---|---|---|---|---|
| Bottleneck adapter | Small inserted modules, plus optional task head | Adds module computation; base can be shared | Many task-specific variants sharing one base | Adapter routing, storage, and extra latency |
| LoRA | Low-rank factors on selected matrices | Some implementations merge updates into base weights | Efficient updates where mergeable serving matters | Target-module choice and merge/quantization compatibility |
| Full fine-tuning | Most or all model weights | No adapter branch at serving | Maximum freedom when data and compute support it | Separate large checkpoints and higher training cost |
| Prompt or prefix tuning | Learned prompt/prefix parameters | Adds or lengthens input context | Adaptation through supported prompt interfaces | Extra context cost and architecture-specific behavior |

The decision should be made against a fixed evaluation set and operational target. Compare quality at equal data and tuning effort, then measure peak training memory, checkpoint size, tokens per second, and p95 latency with the actual serving stack. An adapter can win on task isolation or checkpoint reuse while losing on latency; LoRA can be easier to merge, but merge support is not universal. Full tuning may be justified when the task needs broad representation changes that the selected parameterization cannot express. No method guarantees a particular accuracy or percentage of saved memory across model families.

## Implementation

This dependency-light example computes the size of a two-projection adapter. It demonstrates parameter accounting only; it does not train a model or represent every adapter variant.

```python
def bottleneck_adapter_parameters(hidden_size: int, bottleneck: int, bias: bool = True) -> int:
    if hidden_size <= 0 or bottleneck <= 0:
        raise ValueError("dimensions must be positive")
    weights = hidden_size * bottleneck + bottleneck * hidden_size
    biases = hidden_size + bottleneck if bias else 0
    return weights + biases

assert bottleneck_adapter_parameters(768, 32) == 49_952
assert bottleneck_adapter_parameters(768, 32, bias=False) == 49_152
try:
    bottleneck_adapter_parameters(768, 0)
except ValueError:
    pass
else:
    raise AssertionError("zero-width bottleneck must be rejected")
```

For a real training run, first verify that the library supports the exact model architecture and adapter type. Freeze the base, inspect the named trainable parameters, and confirm gradients reach the adapter after a backward pass. Save the adapter configuration with the weights, pin the base-model revision, and test loading in a fresh process. These checks catch an apparently successful run that accidentally tuned the whole model or saved an adapter that cannot be paired with the intended base.

## Failure Modes

- **Capacity is too small for the domain shift.** Training loss and held-out task metrics both plateau below a full-tuning baseline. Sweep bottleneck width or placement with a fixed validation protocol; if gains stop while the gap remains, compare LoRA or full fine-tuning.
- **Base weights are not actually frozen.** Trainable-parameter counts exceed the intended adapter and head, or base tensors change after an optimizer step. Assert the trainable parameter names and compare a base-weight checksum before and after a short update.
- **Adapter and base revisions are mismatched.** Loading may fail, or outputs degrade despite apparently valid tensors. Store the model identifier and revision with the adapter, then run a known input/output regression test after loading.
- **Task routing leaks state across requests.** A multi-tenant service can return results using the previously active adapter. Include adapter identity in request-scoped configuration and run interleaved A/B requests to verify deterministic isolation.
- **Serving cost is assumed away.** Offline quality is acceptable but p95 latency or throughput misses the service objective. Benchmark the deployed adapter runtime against the same base without adapters, including adapter switching and batching.
- **Adapters are stacked without compatibility evidence.** Quality becomes order-dependent or unstable. Evaluate each composition and ordering on held-out examples; use a supported fusion method or keep adapters separate when composition has no validated benefit.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What does a bottleneck adapter learn? | A down projection, nonlinearity, and up projection inserted into a frozen network's computation, typically with a residual path. |
| How does it differ from LoRA? | Adapters add a module to the forward path; LoRA parameterizes a low-rank update to selected weight matrices. Compare the actual target modules and serving implementation. |
| How do you estimate adapter size? | For two biased linear layers, approximately `2dr + d + r` parameters per module; multiply by the number of insertion sites and count any task head. |
| When would you prefer full fine-tuning? | When representative validation shows the small update cannot meet the quality target and the added training/storage cost is acceptable. |
| Design follow-up: how would you validate a multi-task adapter service? | Pin the shared base, test each adapter independently, interleave requests to test routing isolation, and report task quality, checkpoint size, throughput, and tail latency. |

## Related Concepts

- [LoRA](08-lora.md) — low-rank weight updates, often a closer comparison for parameter-efficient adaptation.
- [Parameter-Efficient Fine-Tuning](11-parameter-efficient-finetuning.md) — broader family that includes adapters and LoRA.
- [Prefix Tuning](10-prefix-tuning.md) — adaptation through learned prefix representations.
- [Fine-Tuning](04-finetuning.md) — full-model baseline and its data and compute trade-offs.
- [Adapters practice](../interview-prep/curriculum-practice.md#adapters) — the matching manifest-owned assessment.
- [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html) — original adapter study.
- [Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning](https://arxiv.org/abs/2311.11077) — adapter methods and their software ecosystem.

## Real-World Examples

Consider a support platform that serves a shared multilingual encoder with one adapter per product line. This can avoid storing a full copy of the encoder for every product. Before adopting it, the team should measure each product's classification quality, adapter storage, cold-load behavior, and routing latency; the architecture alone does not establish a particular savings figure. If product data has different label definitions, evaluation must also be product-specific rather than averaged into one misleading score.
