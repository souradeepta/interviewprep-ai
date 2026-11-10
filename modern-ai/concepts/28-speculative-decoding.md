# Speculative Decoding

**TL;DR:** A cheap draft proposes several tokens and a target model verifies them; acceptance sampling can preserve the target distribution, while speed depends on acceptance and verification cost.

## Prerequisites

Autoregressive decoding and probability distributions.

## Learning Outcomes

- Explain a draft/verify cycle and its acceptance-correction behavior.
- Account for accepted and rejected prefixes over a decoding cycle.
- Derive a symbolic break-even condition from draft, verify and target costs.
- Distinguish a theoretical work estimate from measured latency.

## Core Intuition

A fast junior writer proposes a few words and an expert checks them together. If the first words agree, the expert saves repeated sequential decisions; at the first disagreement, the expert supplies the correction and the process restarts. Draft tokens are useful only when the verification work is cheaper than asking the target model to generate each token separately.

## How It Works

Let q be the draft distribution and p the target distribution. For a proposed token x, rejection sampling accepts with probability \(\min(1,p(x)/q(x))\). If rejected, a correction drawn from the residual distribution proportional to \(\max(0,p-q)\) preserves the target distribution under the algorithm's assumptions. Practical systems may use different speculative methods; exactness claims must state the method and sampling setup.

For a draft length gamma and prefix acceptance A, one cycle yields A draft tokens and typically a correction token when rejection occurs. If draft cost per token is d, target verification cost is v, and ordinary target cost per token is t, a rough cycle ratio is \(t(A+I[A<gamma])/(gamma d+v)\). This ignores scheduling, memory traffic, batching, and variance. It can be below one. Acceptance alone is not speedup.

The expected accepted prefix depends on agreement at successive positions, not only a single average token match rate. If the conditional probability of accepting each proposed token were a constant a and positions were independent, the probability of accepting at least j proposals would be approximately a^j. Real generation violates this simplifying assumption: uncertainty and draft-target disagreement vary with position, prompt, temperature and token type. Measure the full accepted-prefix histogram, including zero acceptance, rather than reporting only mean acceptance. A high average can hide a long tail of cycles that waste draft work.

Verification's parallelism is algorithm-dependent. A target transformer can score a block of candidate tokens in one forward pass, but causal dependencies and attention-mask construction still matter. A batched pass may use more memory and produce a different kernel profile than sequential decoding. The symbolic cost v should therefore be measured under the actual block length, batch shape and precision. If target verification is close to gamma sequential steps in cost, the draft stage has little room to win. If verification is efficient, one accepted block can amortize its fixed cost.

Exactness also requires carefully defined probabilities. The rejection acceptance ratio is based on the target and draft distributions over the same next-token support. Numerical underflow, temperature or top-p transformations applied inconsistently, and mismatched tokenizers can invalidate the simple argument. If draft and target use different tokenizations, a method needs an explicit alignment mechanism. Correct sampling may use a residual distribution after rejection; simply replacing the rejected token with the target argmax is a different decoding policy and can change output distribution.

Operationally, speculation is most attractive when prompts lead to predictable continuations, such as structured output or code completion, but these are hypotheses to verify on real traces. It may be weaker on open-ended reasoning where the draft diverges quickly. Evaluate separately by request class and measure quality parity alongside latency. Include draft memory in the deployment budget, cache synchronization behavior, timeout handling and fallback to ordinary target decoding. The fallback matters because a draft model outage must not make the service unavailable.

An online controller can adjust draft length from recent acceptance and verify cost, but the signal is delayed and workload-dependent. Avoid chasing short-lived fluctuations with a large controller gain. Begin with a bounded sweep over a held-out workload, select stable operating points, then add conservative adaptation only if its effect is measurable. Report confidence intervals and tail latency; a small mean improvement accompanied by worse p99 may violate the service objective.

## Trade-offs

| Choice | Advantage | Cost / boundary |
|---|---|---|
| Larger draft block | More possible tokens per verify | More wasted draft work after early rejection |
| Better draft model | Higher acceptance probability | More draft compute and memory |
| Greedy verification | Simple equality checks | Does not model stochastic distribution-preserving acceptance |
| Adaptive draft length | Can respond to acceptance history | Adds control complexity and needs workload evaluation |

Use speculation when target verification batches candidates efficiently and draft agreement is high. It is unattractive when the target is already cheap, acceptance is poor, or verification serializes on memory.

## Implementation with Python

Run `python3 modern-ai/implementations/28-speculative-decoding.py`. A fixed cost model sweeps every accepted-prefix length for a four-token draft block and prints accepted/rejected counts and a speedup bound. Those are formula outputs, not measured production gains.

```python
import runpy
account = runpy.run_path("modern-ai/implementations/28-speculative-decoding.py")["account"]
cycle = account(4, 3, draft_cost=.15, verify_cost=.8, target_step_cost=1)
assert cycle["acceptance_rate"] == .75
print(cycle["speedup_bound"])  # symbolic cost ratio, not a benchmark
```

## Failure Modes

| Failure | Signal | Response |
|---|---|---|
| Draft mismatch | Low accepted-prefix distribution | Measure acceptance by prompt and generation phase; use a closer draft or disable speculation |
| Verification overhead dominates | Cycle cost exceeds target-only token cost | Profile target verify kernel and sweep draft length |
| Distribution silently changes | Output statistics differ under sampling | Test the acceptance/correction algorithm against the target distribution |

- Draft mismatch wastes proposals: inspect accepted-prefix histograms by task and disable or replace the draft for low-agreement classes.
- Verification overhead removes the gain: compare block verification cost with sequential target cost and sweep block size on representative shapes.
- Sampling behavior drifts: run target-distribution parity checks and verify that acceptance and residual correction use consistent probabilities.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| Why can speculative decoding be lossless? | Correct acceptance and residual sampling recover the target distribution; ask what changes under greedy decoding. |
| What controls speedup? | Acceptance, draft cost, verification cost and batching; follow up with break-even inequality. |
| What do you monitor? | Acceptance prefix, cycle latency, target work and output parity; ask how you would disable it safely. |

## Related Concepts

See [KV cache optimization](29-kv-cache-optimization.md), [serving frameworks](13-llm-serving-frameworks.md), [notebook 28](../notebooks/28-speculative-decoding.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#speculative-decoding).

## References

[Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192) and [Speculative Sampling](https://arxiv.org/abs/2302.01318).
