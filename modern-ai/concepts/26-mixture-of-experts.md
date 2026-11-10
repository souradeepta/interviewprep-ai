# Mixture of Experts Routing

**TL;DR:** A router activates a small selected set of expert feed-forward networks for each token; capacity and balancing determine whether sparse compute remains useful.

## Prerequisites

[Transformer architecture](11-flash-attention.md) and softmax classification.

## Learning Outcomes

- Describe top-k routing and weighted expert aggregation.
- Calculate per-expert capacity from token count, expert count, top-k and capacity factor.
- Interpret a load-balance proxy alongside per-expert counts.
- Diagnose overflow from measured routing counts.

## Core Intuition

Think of a help desk with specialists. Each request is sent to its most relevant specialists, which saves work compared with asking every specialist to answer. But if every request goes to the same popular specialist, the queue erases the compute savings. An MoE layer has the same tension: sparse activation reduces expert computation while routing imbalance and all-to-all communication can dominate.

## How It Works

For token representation \(x_t\), a learned gate computes logits \(z_t=W_gx_t\) and probabilities \(p_{t,i}=softmax(z_t)_i\). Top-k selects experts \(S_t\); selected weights are commonly normalized to sum to one, and the output is \(y_t=\sum_{i\in S_t}w_{t,i}E_i(x_t)\). The number of stored parameters can be large while only k expert networks are active per token. This is not a guarantee of lower end-to-end latency: dispatch, communication, padding and small batches matter.

With T tokens, E experts and top-k, a common token capacity is \(C=\lceil cTk/E\rceil\), where c is the capacity factor. Once an expert reaches C, additional assignments may be dropped, rerouted, or handled by a fallback. The notebook uses a deterministic greedy cap and counts each rejected assignment; production implementations differ in overflow policy. One auxiliary balance proxy is \(E\sum_i f_iP_i\), where f is the observed assignment fraction and P is mean router probability. It is useful as a training signal, not a standalone quality metric.

Routing statistics should be read as a distribution, not one scalar. The assignment histogram reveals which experts execute; the probability histogram reveals what the gate preferred before capacity enforcement. These can disagree: a popular expert can be clipped by capacity, making accepted counts look balanced while many tokens were actually diverted. Record both pre-cap choices and final execution counts, along with dropped or fallback assignments. Report the metrics by layer and by batch because averaging layers can conceal one collapsed router.

The load-balance proxy is derived from the relationship between hard assignments and soft gate probabilities. f_i summarizes discrete traffic while P_i summarizes router preference. When both concentrate on the same expert, their product contributes more loss. This auxiliary objective is added to the main task objective with a weight; increasing that weight can distribute tokens but may also oppose useful specialization. Tune against validation quality and routing health rather than assuming uniform traffic is always optimal. Different domains can legitimately favor different experts.

Capacity is a batch-level constraint. If tokens are unevenly distributed, the expected average T·k/E is not a safe per-expert maximum. A capacity factor provides slack, but slack may increase padding, memory reservation and communication. Small batches have especially noisy routing: one extra token can materially alter an expert's load. At inference, request composition changes over time, so capacity policy should be exercised on bursty, skewed batches instead of only a balanced synthetic batch.

Top-k routing also changes the gradient path. A hard top-k decision is discrete, while selected gate weights remain differentiable. Training therefore relies on the selected-expert path plus auxiliary signals and may have weak feedback for experts that never get selected. Router initialization, noise, warm-up and expert-specific learning dynamics can affect whether specialization emerges. When training is unstable, inspect router-logit scale and entropy as well as loss curves; clipping gradients alone will not repair a gate that routes nearly every token identically.

Finally, sparse arithmetic does not imply sparse system work. Tokens must be grouped by destination, transported to expert devices when model parallelism is used, evaluated, then returned to their original order. This dispatch path can dominate for short sequences or modest batches. Measure token counts and bytes per expert, padding fraction, all-to-all time and expert kernel utilization on the intended accelerator configuration. A dense baseline with fewer parameters can win on latency even while doing more nominal multiply-adds.

## Trade-offs

| Choice | Benefit | Cost or boundary |
|---|---|---|
| Top-1 | Fewer expert evaluations | Less routing redundancy; sensitive to a bad top expert |
| Top-2 | More chance to combine useful specialists | More compute and dispatch traffic |
| Larger capacity factor | Fewer overflow assignments | More padding or reserved memory |
| Auxiliary balance loss | Encourages utilization | Can conflict with specialization if over-weighted |

Use MoE when sparse expert computation justifies its memory and communication complexity. A dense layer may be preferable for small models, low batch sizes, or hardware without efficient expert dispatch.

## Implementation with Python

Run `python3 modern-ai/implementations/26-mixture-of-experts.py`. The fixed gate logits route six tokens over three experts; the program prints capacity, accepted counts, overflow rate and a load-balance proxy for several capacity factors. This is accounting, not a learned router or benchmark.

```python
from pathlib import Path
import runpy
route = runpy.run_path("modern-ai/implementations/26-mixture-of-experts.py")["route"]
result = route([[4, 1, 0], [3, 2, 0], [0, 2, 4]], k=2, capacity_factor=1)
assert result["dropped_token_rate"] >= 0
print(result["counts"], result["load_balance_loss"])
```

## Failure Modes

| Failure | Signal | Response |
|---|---|---|
| Router collapse | One expert receives most assignments | Inspect per-expert counts and gate entropy; adjust balancing or initialization |
| Silent overflow | Nonzero dropped-assignment rate | Measure drops by layer and test a larger capacity or alternate overflow policy |
| Communication dominates | High expert-parallel transfer time despite low FLOPs | Profile dispatch and compare against dense execution at actual batch sizes |

- Router collapse: per-expert counts concentrate on one destination; inspect logits and entropy by layer, then test balancing and initialization changes.
- Silent overflow: accepted token counts hide rejected assignments; log pre-cap routes and dropped rate, then tune capacity or fallback policy.
- Communication dominates: low arithmetic does not translate to lower latency; profile dispatch bytes, all-to-all time and expert utilization at target batch sizes.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| Why can MoE increase parameter count without proportional active compute? | Only top-k experts run per token; ask how router and dispatch costs change latency. |
| What does a capacity factor control? | The per-expert assignment ceiling; follow up with the quality impact of dropping tokens. |
| How would you detect routing collapse? | Per-expert assignment/probability histograms and overflow by layer; do not infer balance from aggregate throughput. |

## Related Concepts

Next: [Router learning](39-router-learning.md), [conditional computation](53-conditional-computation.md). The matching exercise is [notebook 26](../notebooks/26-mixture-of-experts.ipynb) and [inference interview practice](../interview-prep/inference-systems.md#mixture-of-experts).

## References

[GShard](https://arxiv.org/abs/2006.16668) and [Switch Transformers](https://arxiv.org/abs/2101.03961).
