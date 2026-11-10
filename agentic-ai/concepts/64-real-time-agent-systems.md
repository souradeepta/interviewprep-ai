# Real-Time Agent Systems
**TL;DR:** A real-time agent system must produce a valid response or safe fallback within a deadline under burst load and partial failure.

## Prerequisites
Async services, latency percentiles, and agent safety controls.

## Learning Outcomes
- Allocate an end-to-end deadline across model and tool stages.
- Design degradation behavior for timeout and overload.
- Measure deadline success and safety under bursty load.

## Core Intuition
An emergency operator acts on the best available information before a deadline, then updates when more evidence arrives. A real-time agent has the same constraint: a perfect answer after the decision window is a failed answer. The safety policy defines which quick fallback is acceptable.

## How It Works
The operating flow identifies the task state, selects a bounded strategy, gathers only the evidence needed for that decision, validates the result, and records the outcome. Deadline behavior must be designed at the workflow level. Set a monotonic deadline at request arrival and derive per-stage timeouts from its remaining budget; independent fixed timeouts can add up to far more than the promised limit. Cancellation must propagate to tools and model calls where possible, and any result arriving after expiry must be ignored for side effects. For streaming, distinguish transport responsiveness from verified completion and expose status transitions so clients can handle partial or fallback responses safely.
Set a deadline at ingress and pass the remaining budget to every model and tool call. Reserve time for validation and response delivery rather than allowing the final dependency to consume the whole budget. Streaming can improve perceived responsiveness, but partial output must be clearly provisional if it has not passed validation. For control tasks, separate fast deterministic safeguards from slower model recommendations. Queue policies should reject or degrade work before overload causes all requests to miss deadlines.
Deadline behavior must be designed at the workflow level. Set a monotonic deadline at request arrival and derive per-stage timeouts from its remaining budget; independent fixed timeouts can add up to far more than the promised limit. Cancellation must propagate to tools and model calls where possible, and any result arriving after expiry must be ignored for side effects. For streaming, distinguish transport responsiveness from verified completion and expose status transitions so clients can handle partial or fallback responses safely.
Measure deadline success rate, p50/p95/p99 end-to-end latency, queue wait, model time, tool time, fallback rate, and correctness at each latency tier. Load tests need burst arrivals, slow dependencies, cancellation, and retry storms. A low mean does not establish a real-time guarantee; define a service-level objective and load envelope. In safety-sensitive settings, verify that timeout behavior is safe and that late responses cannot execute after cancellation.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Wait for complete result | Validated response | May miss deadline |
| Stream provisional output | Fast perceived response | Partial content may mislead |
| Fast fallback | Bounded response time | Lower task capability |
| Queue under load | Preserves requests | Deadline expiration and memory growth |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def remaining_budget(deadline, now, reserve):
    return max(0, deadline - now - reserve)

assert remaining_budget(10.0, 7.0, 1.0) == 2.0
```

## Failure Modes
- Tool call consumes the whole deadline: propagate cancellation and reserve validation time.
- Queue overload causes tail collapse: bound queue length and shed or degrade work early.
- Late result triggers stale action: bind actions to deadlines and reject expired completions.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| How do you design around a strict deadline? | Budget each stage, propagate cancellation, and define a safe fallback before implementation. Design follow-up: what happens when a tool returns late? |
| Why report p99 rather than mean? | Averages hide the users whose work misses its deadline; tail behavior determines deadline reliability. Debug follow-up: split queue and service time. |
| When is streaming unsafe? | When partial output could cause action or be mistaken for a validated result; label provisional content and gate side effects. |

## Related Concepts
See [Real-time agents](50-real-time-agents.md), [latency optimization](33-latency-optimization-agents.md), [agent monitoring](30-agent-monitoring.md).

## References
- [sre.google](https://sre.google/sre-book/service-level-objectives/)
- [www.nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)

## Further Topic Detail
## Detailed Explanation

Real-time agent systems must respond to continuous streams of data and make decisions within strict latency constraints—think autonomous vehicles deciding how to respond to traffic, trading bots reacting to market movements, or live customer support systems. Real-time creates constraints that fundamentally change system design: computation must be fast (milliseconds or milliseconds), decisions must be robust despite incomplete information (can't wait for perfect data), and failures must be recoverable (no ability to pause and debug).

Key challenges: (1) Latency budgets (every millisecond matters, cutting model inference in half cuts agent latency in half), (2) Streaming reasoning (decisions made with incomplete information, updated as new data arrives), (3) State consistency (ensuring distributed components stay coordinated), (4) Failure recovery (graceful degradation, fallback strategies), (5) Monitoring (understanding system behavior in production). Design choices include: edge deployment (running agent logic locally for latency), batching decisions (grouping updates to reduce overhead), approximate reasoning (fast decisions over perfect decisions), and hierarchical response (immediate reaction + longer-term optimization).

Real-time agent systems represent the frontier of agent applications—autonomous systems that must perform in the real world, not just offline. Understanding them requires systems and operations thinking, and appreciation that 'real-time' isn't about AI capability but about infrastructure and system design.


## Core Intuition

A chess player thinking deeply can beat a fast player. But a chess player in a lightning round (seconds per move) must think fast rather than deeply. Real-time agent systems are like lightning chess: decisions must happen immediately even if imperfect. Infrastructure and algorithm efficiency matter more than raw capability.


## How It Works

1. Streaming: return response tokens as they're generated (don't wait for full response)
2. Concurrency: handle multiple requests simultaneously (not sequential)
3. Latency: first-token latency (how fast first token appears) and total latency
4. Buffering: buffer tokens to avoid excessive I/O, stream in chunks
5. Inference optimization: quantization, caching, batching to reduce latency
6. Infrastructure: use GPU, high-bandwidth connections, optimized serving (vLLM)
7. Monitoring: track latency distribution, percentiles (p50, p99)




## Interview Q&A


**Q: How do you stream responses from agents?**
A: Use streaming API: GPT-4 streaming, LLaMA server with streaming. Get tokens one at a time, send to user as they arrive. UX: user sees incremental response (feels responsive). Implementation: WebSocket or Server-Sent Events (SSE).

**Q: What causes agent latency and how do you reduce it?**
A: Sources: model inference (biggest), tokenization, tool calls, API latency. Reduce: (1) faster models (7B vs 70B), (2) quantization (4-bit vs 16-bit), (3) batching (process multiple requests together), (4) caching (reuse computations).

**Q: How do you handle concurrent agent requests?**
A: Load balancing: distribute to multiple replicas. Queue: if more requests than capacity, queue and process in order. Async: use async/await, don't block on slow requests. Graceful degradation: prioritize critical requests, drop non-critical if overwhelmed.

**Q: What is first-token latency and why does it matter?**
A: First-token: how fast user sees first response token. Matters: affects perceived responsiveness. Optimize: reduce prompt processing, use smaller models, cache embeddings. Typical: 100-500ms first-token (good), >1s (feels slow).

**Q: How do you optimize for different latency SLAs?**
A: Compute SLA: p50 latency (median), p99 (99th percentile). Different tiers: standard (p50<1s), fast (p50<100ms), urgent (p50<10ms). Cost increases: faster tier = more resources. Choose based on use case.
