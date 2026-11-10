# ML Pipeline Orchestration and Backfills

**TL;DR:** Orchestration makes dependencies, retries and data intervals explicit; safe backfills require bounded scope, idempotent outputs and clear publication semantics.

## Prerequisites

[Data pipelines](01-data-pipelines.md), [data versioning](04-data-versioning.md), and basic DAG concepts.

## Learning Outcomes

- Represent a training workflow as a dependency graph and topologically order its tasks.
- Explain idempotency and why retries must not duplicate side effects.
- Bound a backfill by partition range, concurrency and resource budget.
- Define when outputs become visible and how partial runs are recovered.

## Core Intuition

A pipeline DAG is a recipe with prerequisites: evaluation should not run before training produces a model, and publication should not happen until validation passes. An orchestrator tracks this order and records what happened. Re-running a task after a worker failure is safe only if the task's output is deterministic or its side effects are idempotent. A historical backfill is simply the same recipe applied to old partitions, but it can overload systems or publish inconsistent results if its bounds and commit rules are unclear.

## How It Works

Represent tasks as vertices and dependencies as directed edges. A valid workflow is a directed acyclic graph (DAG); topological sorting yields an order in which every dependency precedes its consumer. Cycles indicate an invalid plan and should fail before doing work. In a production scheduler, ready tasks may run concurrently subject to worker pools, quotas and priority. The small implementation returns a stable ordering and simulates completion for one data partition.

Partition identity must be explicit. A daily training-data partition might be keyed by event date, source version and transformation version. A run key such as (task, partition, code/config version) can make retries idempotent: if the key is already committed, a retry reuses its result rather than writing a duplicate. Do not equate “task process exited zero” with “output is correct”; validate row counts, schema, checksums and freshness before marking a partition publishable.

Writes should use a transactional or staged pattern. Write results to a temporary location, validate them, then atomically promote a manifest or pointer. If the job dies before promotion, readers continue using the previous complete output. If it dies after promotion, the commit record identifies the new version. Where object storage lacks multi-file transactions, an immutable manifest can make a set of files appear as one logical commit. Keep lineage linking output partitions to input snapshots and task versions.

Backfills need a bounded execution plan. Define inclusive/exclusive date ranges, maximum partition count, concurrency, rate limits, expected compute and a stop condition. Prefer a dry-run that lists partitions and dependencies before launching. Historical jobs can compete with live freshness work, so use a separate queue or low-priority resource pool. If the number of partitions exceeds a safe threshold, split the work into reviewable batches rather than launching an unbounded backlog.

Retries should distinguish transient and permanent failures. Network timeouts may be retried with exponential backoff and jitter; schema errors or missing required files should fail fast and page the owning team. Repeatedly retrying deterministic invalid input burns capacity and can hide the original error. Track attempt count, failure class, run key and elapsed time. Preserve logs and input version identifiers so a failed partition can be reproduced.

The operational decision in the implementation is a maximum backfill partition count. The example allows a run of at most three partitions and rejects larger ranges. A real limit should be set from measured worker capacity and live workload headroom. Before backfill, estimate resource use, monitor live queue delay and set an abort threshold, such as stopping if live freshness lag exceeds its SLO for two consecutive windows.

Scheduling and data time are related but not identical. A task may run at 02:00 for the previous day's partition; the scheduler's wall clock should not determine which rows belong in that partition. Use explicit data intervals and deterministic input snapshots. For late-arriving events, choose a correction policy: reopen a bounded partition, apply a compensating update, or accept a documented lateness window. Record the interval and source watermark in the output manifest so downstream training can reproduce the same data view.

Backfill order matters when outputs are cumulative. Recomputing a daily aggregate for one historical day should not accidentally overwrite later state with a stale snapshot. Prefer partition-local outputs and a materialization step that builds a new version from a coherent set of partitions. If a backfill is interrupted, maintain per-partition completion and validation state, then resume only missing work. A single global “job succeeded” flag is insufficient when a multi-day run can partially commit.

Resource controls should be explicit in both scheduler configuration and the run plan. Cap concurrency, assign a low-priority queue, reserve capacity for live jobs and watch database/source limits. A retry storm can amplify a transient outage; use bounded attempts, exponential backoff with jitter and a circuit breaker for repeated dependency failures. Failed permanent validation should stop the dependent branch and preserve the bad input sample for diagnosis rather than retrying indefinitely.

Security and reproducibility apply to orchestration metadata too. Run logs can contain identifiers or feature values, so redact or hash sensitive fields and enforce retention. Record actor, code revision, configuration, input manifest and output checksum. Backfills may use updated code against old data; label that run as a new transformation version so its output is not confused with the historical artifact originally consumed by a model.

## Trade-offs

| Choice | Benefit | Cost or risk |
|---|---|---|
| Task-level retry | Recovers transient worker failures | Requires idempotent side effects |
| Whole-DAG retry | Simple to reason about | Recomputes successful expensive tasks |
| Atomic publish pointer | Prevents readers seeing partial output | Requires immutable versioned artifacts |
| Large backfill batch | Completes history quickly | Competes with live traffic and broadens failure blast radius |
| Small bounded batch | Easier to observe and stop | Takes longer and needs orchestration overhead |

## Implementation with Python

Run `python3 mlops/implementations/19-pipeline-orchestration.py`. The dependency-light module detects cycles, returns task order, treats committed task-partition pairs as completed, and bounds an inclusive integer partition range. The example backfill guard is three partitions; raise it only after capacity testing.

```python
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("orch", "mlops/implementations/19-pipeline-orchestration.py")
orch = module_from_spec(spec); spec.loader.exec_module(orch)
assert orch.topological_order(["a", "b"], {"b": ["a"]}) == ["a", "b"]
```

## Failure Modes

- Retry duplicates a write: include a stable run key and atomically commit output manifests.
- Backfill overloads live processing: bound partition count and concurrency; abort when live freshness lag exceeds its SLO.
- Partial output is consumed: stage outputs and publish only after validation and commit.
- Dependency cycle stalls execution: validate DAG acyclicity before scheduling any task.

## Interview Quick Reference

| Question | Expected answer | Follow-up |
|---|---|---|
| How do you make a training task safe to retry? | Use deterministic/versioned inputs, idempotency keys and atomic output publication. | Debug a retry that produced two model artifacts. |
| How should a large historical backfill run? | Dry-run the bounded partitions, throttle separately from live traffic and monitor freshness. | Design an abort rule during a backfill. |
| What does orchestration provide beyond cron? | Dependency state, retries, scheduling, observability and recovery semantics. | Design a DAG that publishes only validated model artifacts. |

## Related Concepts

See [data pipelines](01-data-pipelines.md), [data versioning](04-data-versioning.md), [reproducibility](07-reproducibility.md), and [MLOps interview practice](../interview-prep/mlops-interview-questions.md#data-contracts-and-feature-reliability).

## References

[Apache Airflow: DAGs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html) and [Google Cloud: Idempotent Data Processing](https://cloud.google.com/architecture/idempotent-data-processing-pipelines).
