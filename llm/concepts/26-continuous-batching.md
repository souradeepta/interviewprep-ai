# Continuous Batching

TL;DR: Continuous batching admits new generation requests and retires completed ones at iteration boundaries, keeping accelerator work occupied despite variable prompt and output lengths.

## Prerequisites

This lesson assumes a decoder-only generation loop, prompt prefill versus token decode, and [context-window](25-context-window.md) limits. A serving system also needs per-request state such as generated-token count and a [KV cache](23-kv-cache.md); continuous batching governs the scheduling of that state, not the attention math itself.

## Learning Outcomes

- Contrast static batches with iteration-level admission and completion in a generation server.
- Explain how request-length variation, KV capacity, and queueing affect throughput and latency.
- Compute useful scheduling metrics from a deterministic request trace.
- Diagnose starvation, head-of-line delay, or memory pressure using per-request and aggregate measurements.

## Core Intuition

Imagine a shuttle that waits for every passenger to finish a long trip before accepting anyone else. A passenger with a short trip leaves an empty seat for the remainder of that batch. Language generation has the same mismatch: requests finish after different numbers of output tokens. Static batching commonly keeps completed slots idle until the slowest sequence finishes. Continuous batching refills available slots as soon as individual requests complete, so the device can continue serving work.

The term is sometimes used broadly for dynamic batching, but the important serving pattern here is iteration-level scheduling: each decode iteration advances active sequences by one token, removes finished sequences, and may admit waiting requests subject to resource constraints. It does not mean requests with arbitrary shapes are merged without coordination. A scheduler still needs compatible model, adapter, sampling, and parallelism settings, and it must preserve request isolation and output order.

## How It Works

A typical server receives a request, tokenizes and validates its prompt, queues it, runs prefill, then schedules decode iterations. At a scheduling boundary, completed or cancelled requests release their KV blocks and sequence slots. The scheduler selects eligible waiting requests, allocates cache capacity, and forms the next iteration's work. Each active sequence advances according to its own state, and finished sequences are streamed and retired independently.

Static batching groups a fixed set of requests and often runs until the longest output completes. If request i produces S_i tokens, useful decode slot work is approximately `sum_i S_i`, while allocated slot work is `B * max_i(S_i)`. A simple slot utilization ratio for that batch is `sum_i S_i / (B * max_i(S_i))`. This describes ragged sequence slots only; it ignores prefill, kernel efficiency, queueing, and memory. Iteration-level replacement can reduce this padding waste, but throughput is also bounded by accelerator compute, memory bandwidth, KV capacity, and scheduler overhead.

Admission is a constrained decision. A long prompt consumes prefill compute and creates more KV state; a long generation keeps that state resident. If the scheduler admits too many requests, cache allocation or temporary buffers can fail. If it admits too few, accelerator capacity is underused. Fairness also matters: admitting only short jobs may improve average completion time while starving long prompts. Common controls include token budgets, maximum active sequences, priority or aging, per-tenant quotas, and separate prefill/decode policies.

Metrics should retain distributions rather than only averages. Time to first token (TTFT) includes queueing, prefill, and scheduling. Inter-token latency measures decode responsiveness. Request latency and tokens per second describe different objectives. Track queued requests, active sequences, prompt and output token distributions, KV blocks in use, preemption/recompute events, and cancellation rates. Report p50 and tail percentiles by request length and tenant, because a global average can hide starvation or long-context regressions.

## Trade-offs

| Scheduling choice | Benefit | Cost or assumption | Diagnostic |
|---|---|---|---|
| Static batching | Simple scheduling and predictable group shape | Finished sequences can leave unused slots until the batch ends | Slot utilization versus output-length spread |
| Continuous/iteration-level batching | Reuses slots as requests finish; often improves utilization under varied lengths | More scheduler bookkeeping and variable batch shape | Throughput, queue time, and scheduler time by active batch size |
| Aggressive admission | More concurrent work and potentially higher throughput | KV pressure, preemption, OOM risk, and tail-latency increases | Cache occupancy, allocation failures, and p99 latency |
| Conservative admission | More memory headroom and predictable latency | Device underutilization and longer queues | GPU utilization alongside queue length and TTFT |
| Length-aware/fair scheduling | Can reduce padding or protect service classes | May disadvantage long jobs or add policy complexity | Per-length completion time and starvation age |

Continuous batching is most helpful when request arrivals overlap and output lengths vary. With a single request, highly uniform short outputs, expensive scheduler overhead, or a workload dominated by prefill, gains may be small. Paged KV allocation can make dynamic admission more practical by reducing external fragmentation, but paging does not eliminate the underlying byte budget.

## Implementation

This standard-library-only trace illustrates slot replacement at a fixed capacity. It compares elapsed decode iterations and slot utilization for a static batch versus a scheduler that admits queued work when a request finishes. Each request is represented by its remaining decode steps; the trace is deterministic and does not model GPU execution, prompt prefill, or KV memory.

```python
def static_schedule(lengths, capacity):
    if not lengths or capacity <= 0 or any(n <= 0 for n in lengths):
        raise ValueError("lengths and capacity must be positive")
    batches = [lengths[i:i + capacity] for i in range(0, len(lengths), capacity)]
    ticks = sum(max(batch) for batch in batches)
    return ticks, sum(len(batch) * max(batch) for batch in batches)


def continuous_schedule(lengths, capacity):
    if capacity <= 0 or any(n <= 0 for n in lengths):
        raise ValueError("capacity and request lengths must be positive")
    waiting = list(lengths)
    active = []
    ticks = 0
    useful = 0
    while waiting or active:
        while waiting and len(active) < capacity:
            active.append(waiting.pop(0))
        ticks += 1
        useful += len(active)
        active = [remaining - 1 for remaining in active if remaining > 1]
    return ticks, useful


lengths = [1, 2, 8]
static_ticks, static_allocated = static_schedule(lengths, capacity=2)
continuous_ticks, continuous_useful = continuous_schedule(lengths, capacity=2)
assert static_ticks == 10
assert static_allocated == 12
assert (continuous_ticks, continuous_useful) == (8, 11)
assert continuous_ticks < static_ticks  # queued C replaces completed A/B work
assert continuous_useful == sum(lengths)
```

In a real engine, request arrivals, prefill, cancellation, and KV allocation affect the trace. Evaluate a representative arrival distribution with fixed hardware and model; compare static and continuous scheduling under the same admission limits. Include warm and cold prompts, short and long generations, and bursty load. A toy scheduler's slot-work result is not evidence of production throughput.

## Failure Modes

- **Admission exceeds cache capacity:** New requests fail or trigger preemption and recomputation. Estimate KV bytes per active token, cap admission with safety margin, and alert on allocation failures and preemptions.
- **Long requests starve:** A strict short-job-first policy can indefinitely defer large prompts. Track wait age by length and tenant, use aging or quotas, and test adversarial arrivals.
- **Tail latency rises despite higher throughput:** Higher occupancy can increase queueing and memory contention. Monitor TTFT and inter-token p95/p99 separately from aggregate tokens per second.
- **Scheduler overhead dominates small batches:** Frequent bookkeeping can cost more than saved idle slots. Profile scheduler CPU time and compare against a simple baseline at realistic batch sizes.
- **Cancellation leaks state:** A client disconnect may leave cache blocks allocated. Test cancellation at prefill and decode boundaries; observe allocation release time and orphaned sequence counts.
- **Mixed request settings are combined incorrectly:** Different adapters or sampling constraints may not share a compatible execution batch. Validate compatibility before admission and test output isolation across heterogeneous requests.
- **Aggregate metrics hide unfairness:** Averages can improve while one request class degrades. Break down queue time, completion time, and rejection rate by prompt length, output length, and service class.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| What is continuous batching? | A scheduler adds and removes sequences between decode iterations instead of holding fixed slots until a whole batch ends. | Debug idle slots when output lengths vary. |
| Why can throughput improve? | Completed requests release execution slots, which can be filled by queued work. | Explain why higher throughput may still worsen p99 latency. |
| What limits admission? | KV-cache capacity, compute, memory bandwidth, temporary buffers, and service policy. | Design a safe admission controller from token-level memory estimates. |
| How do you evaluate fairness? | Measure wait and completion distributions by length and tenant, and test starvation cases. | Choose an aging policy and state its trade-offs. |

## Related Concepts

- [KV Cache](23-kv-cache.md) explains the per-sequence state whose allocation constrains admission.
- [Attention Optimization](24-attention-optimization.md) covers attention kernel and architecture choices that affect iteration cost.
- [Inference Optimization](28-inference-optimization.md) combines serving-level latency and throughput techniques.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) describes paged KV management and a serving system with iteration-level scheduling.

## Practice

- [Continuous batching interview practice](../interview-prep/curriculum-practice.md#continuous-batching) walks through fixed-capacity queued admissions and scheduling metrics.

## References

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) introduces iteration-level scheduling for autoregressive generation.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) evaluates KV memory management and serving throughput.
