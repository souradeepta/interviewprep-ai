# Cost, Capacity, and Reliability Planning

**TL;DR:** Plan ML service capacity from peak demand, measured per-worker capacity, utilization target and headroom, then validate the cost and reliability policy under bursts and failures.

## Prerequisites

[Model serving](14-model-serving.md), [deployment strategies](16-deployment-strategies.md), and basic rate/cost arithmetic.

## Learning Outcomes

- Estimate worker count from demand, tested capacity, utilization target and headroom.
- Calculate a simple monthly inference cost from request volume and unit cost.
- Apply a budget reserve before declaring a plan acceptable.
- Explain how redundancy, autoscaling lag, batching and workload mix affect reliability and cost.

## Core Intuition

Sizing a model service from average traffic is like staffing a help desk for the quietest hour. Peaks, retries, cold starts and worker failures determine whether users wait or requests fail. Overprovisioning reduces risk but consumes budget; underprovisioning causes queue growth and can create a retry storm that makes the original overload worse. Capacity planning turns observed workload and service objectives into a measurable resource plan, with explicit uncertainty and fallback choices.

## How It Works

Estimate arrival demand by route and time period, then benchmark model capacity on the actual hardware, model version, precision, batch shape and latency objective. A worker’s sustainable capacity is not its maximum throughput at 100% utilization; queueing latency rises sharply near saturation. This exercise computes workers as demand × headroom divided by per-worker capacity × target utilization, rounded up. For 100 requests per second, 25 requests/second measured capacity, 70% target utilization and 1.3 headroom, the plan is eight workers. The values must come from load tests and an explicit service objective.

Capacity should include failure tolerance. If the service must tolerate one worker loss, size N+1 or apply the expected failure fraction to peak demand. Region loss may require multi-zone or multi-region capacity and replicated artifacts. Autoscaling can respond to demand, but it has detection, provisioning and model-load delays; keep enough warm capacity for the burst that arrives before new workers are ready. Scale on queue depth, latency or concurrency in addition to CPU/GPU utilization because accelerator saturation and queue pressure may not align with host CPU.

Cost models should separate variable and fixed components. Variable cost may be requests, input/output tokens, accelerator-seconds or storage; fixed cost includes minimum replicas, registry, monitoring and network charges. The provided function estimates monthly variable inference cost from daily requests and cost per thousand, then adds fixed cost. For generative systems, input and output token distributions differ, so a single request price can obscure expensive long-context cohorts. Include retries, shadow traffic, evaluation jobs, cache capacity, idle warm workers and data egress where relevant.

Budget decisions should include a reserve. This lesson uses a 10% reserve, so projected cost must remain at or below 90% of the budget for an automatic “within reserve” result. A plan above that line may still be viable, but requires a review that compares quality, latency, reliability and scaling choices. Cost reduction should not silently degrade a safety or latency objective. Potential levers include batching, model quantization, caching, routing, output limits, right-sizing and low-priority scheduling; each needs a quality and tail-latency evaluation.

Reliability is not only replica count. Define SLOs for availability, latency and successful predictions; model upstream data, feature store, registry, network and deployment dependencies. Use error budgets to balance release speed and reliability: repeated SLO misses can pause risky releases while teams address systemic causes. Provide degraded modes, such as a simpler model or deterministic rule, but measure their quality and ensure they are safe for the use case. Circuit breakers and load shedding can protect the service during overload, with explicit criteria for which requests are deferred or rejected.

Capacity tests should resemble production. Replay realistic arrival bursts, prompt lengths, output lengths, request mix and retry behavior. Test warm and cold starts, one-worker failure, dependency slowdown and a traffic increase beyond forecast. Observe p50/p95/p99 latency, queue delay, error rate, throughput, cost per successful request and GPU memory. Averages conceal overload tails. Record the tested image, weights, feature schema and runtime so later releases can be compared consistently.

Use the plan as a decision record: demand forecast source, uncertainty range, hardware benchmark, target utilization, headroom, reserve, redundancy target, budget owner and scale triggers. Example action: when projected cost exceeds 90% of budget or p95 latency exceeds its SLO for two windows, review batching/routing/capacity before scaling further. A cost-per-request target alone can encourage unsafe batching or low-quality models; pair financial guardrails with service and model-quality constraints.

## Trade-offs

| Choice | Reliability/cost effect | Boundary |
|---|---|---|
| High utilization | Lower idle spend | Queue latency and failure sensitivity increase near saturation |
| Spare headroom | Absorbs bursts and worker loss | Pays for idle capacity |
| Autoscaling | Matches resources to demand | Provisioning and model warm-up lag |
| Batching | Improves accelerator utilization | Can add wait time and mix incompatible requests |
| Smaller/quantized model | Can reduce memory and inference cost | Quality and native-kernel performance may change |

## Implementation with Python

Run `python3 mlops/implementations/24-cost-capacity-reliability.py`. It estimates workers with 30% demand headroom and a 70% utilization target, computes a monthly cost approximation, and reserves 10% of budget. These are sample thresholds, not measured capacity claims; benchmark the production shape before using them.

```python
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("capacity", "mlops/implementations/24-cost-capacity-reliability.py")
capacity = module_from_spec(spec); spec.loader.exec_module(capacity)
plan = capacity.capacity_plan(100, 25)
assert plan["workers"] == 8
```

## Failure Modes

- Average demand underestimates bursts: use peak/seasonal traces and retain measured headroom.
- Maximum benchmark is treated as sustainable capacity: benchmark at target latency and utilization, then leave reserve.
- Autoscaling reacts too late: prewarm minimum capacity and include model-load delay in burst tests.
- Cost optimization breaks service quality: gate changes on latency, reliability and model-quality metrics.

## Interview Quick Reference

| Question | Expected answer | Follow-up |
|---|---|---|
| How many model workers do you provision? | Use peak demand, measured capacity under latency SLO, target utilization and failure headroom. | Design a load test to validate a worker’s sustainable RPS. |
| What do you do when cost exceeds budget? | Decompose cost, compare safe levers and require quality/latency gates. | Debug a cost drop accompanied by a rising timeout rate. |
| Why not run at 95% utilization? | Queueing and failure headroom can cause nonlinear latency and overload cascades. | Design a reserve policy for a predictable daily burst. |

## Related Concepts

See [model serving](14-model-serving.md), [deployment strategies](16-deployment-strategies.md), [monitoring](17-monitoring-alert-design.md), and [MLOps interview practice](../interview-prep/mlops-interview-questions.md#reliability-governance-and-cost).

## References

[Google SRE: Capacity Planning](https://sre.google/sre-book/software-engineering-in-sre/) and [AWS Well-Architected Framework: Cost Optimization](https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/welcome.html).
