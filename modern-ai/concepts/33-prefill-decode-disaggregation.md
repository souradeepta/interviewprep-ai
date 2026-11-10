# Prefill and Decode Scheduling

**TL;DR:** Prompt prefill and token-by-token decode have different compute and memory patterns; separating their queues can improve resource control while adding transfer and scheduling costs.

## Prerequisites

[KV cache optimization](29-kv-cache-optimization.md) and serving latency metrics.

## Learning Outcomes

- Define TTFT, time per output token and end-to-end latency.
- Model a request through prefill and decode queues.
- Explain cache-transfer and scheduling costs introduced by disaggregation.
- Choose percentile metrics that expose queueing regressions.

## Core Intuition

Prefill is preparing a whole case file; decode is answering one turn at a time while retaining state. A long prompt can occupy compute that would otherwise advance many ongoing responses. Separate pools let each stage receive different resources and scheduling, but moving cache state between pools has a price.

## How It Works

Prefill processes prompt tokens in parallel through the model and produces the initial KV cache. Decode consumes that cache and repeatedly generates tokens, often constrained by memory bandwidth and per-step dependencies. Time-to-first-token (TTFT) ends after prefill and scheduling delay; time-per-output-token and end-to-end latency describe decode and total service. User experience also depends on queueing percentiles and output length.

A disaggregated architecture routes prefill work to one worker pool and decode work to another. If cache transfer takes X bytes over effective bandwidth B, transfer time is at least X/B before protocol overhead. Separation can reduce head-of-line blocking and allow distinct batch policies. It can also increase network traffic, cache copies, coordination and failure modes. A simple FCFS pipeline model exposes queue coupling but omits parallel workers, preemption, batching and transfer time.

Queueing behavior depends on utilization and service-time variance, not only average work. When a stage approaches full utilization, a small burst can sharply increase waiting time. Long prefills may delay short prompts in a single FCFS queue, while variable decode lengths can hold a decode slot for very different durations. Report queue wait separately from service time and track p50, p95 and p99 by prompt-length and output-length buckets. A model with stable mean latency can still miss an interactive SLA when its tail is heavy.

Disaggregation introduces a handoff boundary for the KV cache. The handoff may use local interconnect, remote memory or network transfer, each with different bandwidth, contention and failure behavior. Cache blocks need ownership and version metadata during migration. A worker restart between prefill and decode must either recover state, rerun prefill or fail the request predictably. Measure transferred bytes and time under concurrent traffic; an isolated bandwidth number will not capture shared fabric contention.

Scheduling policies should account for both fairness and utilization. Continuous batching admits new requests into decode iterations, improving hardware occupancy but potentially delaying a request behind incompatible shapes. Chunked prefill splits a long prompt into smaller pieces, allowing decode work to interleave, while adding scheduling overhead and potentially extending that prompt's TTFT. Priority classes can protect interactive traffic but starve background work. Define maximum wait and aging rules, then test mixed workloads rather than a single homogeneous request set.

Admission control is a resource safeguard. Before accepting a request, estimate prompt prefill cost and expected cache reservation. If predictions are uncertain, use conservative bounds or allow a queue with a deadline. Over-admitting requests can make every request slow and increase memory pressure; rejecting too aggressively wastes capacity. Track accepted, queued, rejected and timed-out requests alongside token throughput so a throughput gain is not achieved by excluding difficult traffic.

The simulator in this lesson deliberately uses one worker for each stage and deterministic service demands. It captures the recurrence that a stage can begin only after both the request is ready and the worker is free. Extend it by adding parallel workers, cache-transfer delay and a bursty arrival process before using it to compare scheduling ideas. Even then, use it to form hypotheses; calibrate with traces and a deployment-like load test because accelerator kernels and communication are absent from the model.

Before splitting pools, estimate both utilization and the cache handoff. Let λ be arrival rate and E[S] the stage service time; utilization ρ = λE[S]/c for c equivalent workers. As ρ approaches one, queue wait becomes sensitive to bursts and service-time variance, so average latency is not enough to choose a design. Estimate prefill and decode separately using representative prompt and output length distributions, then include bytes transferred, transfer concurrency, retry probability, and reserved memory. Compare co-located and disaggregated runs at the same offered load and acceptance policy. If the split helps only after rejecting long prompts, report that selection effect explicitly rather than attributing it to scheduling efficiency.

## Trade-offs

| Design | Benefit | Cost / boundary |
|---|---|---|
| Co-located stages | No cache handoff | Competing resource needs and possible head-of-line blocking |
| Separate pools | Independent scaling and scheduling | Cache movement and orchestration overhead |
| Continuous batching | Better utilization across requests | More complex fairness and latency tails |
| Priority scheduling | Protects latency-sensitive work | Starvation risk for long or low-priority requests |

## Implementation with Python

Run `python3 modern-ai/implementations/33-prefill-decode-disaggregation.py`. Fixed synthetic service demands traverse prefill and decode queues; the output includes TTFT, finish time and latency quantiles. It illustrates scheduling only and does not predict machine performance.

```python
import runpy
m = runpy.run_path("modern-ai/implementations/33-prefill-decode-disaggregation.py")
simulate = m.get("simulate")
rows = simulate(list(((0, 3, 2),)))
assert rows[0]["first_token"] == 3 and rows[0]["finish"] == 5
```

## Failure Modes

| Failure | Diagnostic | Mitigation |
|---|---|---|
| Transfer dominates saved queue time | Bytes moved, transfer latency and TTFT | Place pools close, reduce copies or co-locate small jobs |
| Long decode jobs block short ones | Queue wait by output-length class | Use chunked preemption or fair scheduling |
| Mean looks healthy but users see stalls | p95/p99 TTFT and completion latency | Load-test bursts and report tail quantiles |

- Cache-transfer contention erases stage isolation gains: correlate handoff bytes and transfer queue time with TTFT, then compare co-location under identical offered load.
- Variable output lengths starve short responses: bucket queue wait by remaining decode tokens and add aging or preemption with a bounded fairness test.
- Admission overcommits KV memory: compare reserved blocks with observed peak use and reject or defer requests before workers enter an OOM cascade.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| Why split prefill and decode? | Their compute patterns differ; follow up with cache transfer costs. |
| Which metric captures prompt responsiveness? | TTFT, separated from per-token decode and total latency; ask how you decompose queue delay. |
| When would you avoid disaggregation? | Small workload or transfer/coordination cost exceeds contention benefit; follow up on experiment design. |

## Related Concepts

See [dynamic batching](47-dynamic-batching.md), [cache-aware scheduling](50-cache-aware-scheduling.md), [notebook 33](../notebooks/33-prefill-decode-disaggregation.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#prefill-and-decode-scheduling).

Observability should preserve a per-request timeline: admission, queue entry, prefill start/end, cache transfer start/end, decode start, first token and completion. Decompose total latency into queue, compute and handoff components. Aggregate metrics alone cannot show whether a regression came from a slower prefill kernel or a growing decode backlog. Record request class, prompt length, generated tokens and scheduling policy while controlling cardinality and protecting sensitive content.

## References

[DistServe](https://arxiv.org/abs/2401.09670) and [Sarathi-Serve](https://arxiv.org/abs/2403.02310).
