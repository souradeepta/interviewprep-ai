# Agent Interview Questions

For every 5–10 minute prompt, state assumptions, tool/state design, success
metric, failure modes, budget, observability, safety boundary, and one follow-up.

## Tool contracts and state

1. Define a typed tool contract that makes invalid arguments and side effects explicit.
2. Make retries safe with idempotency keys and a bounded retry policy.
3. Choose memory boundaries between conversation, task, user, and durable state.
4. Recover a workflow from a stale checkpoint without repeating a side effect.

## Planning and orchestration

5. Compare ReAct, a fixed workflow, and planner/executor decomposition for a support task.
6. Route tasks to specialists while bounding fan-out, cost, and context duplication.
7. Design multi-agent handoffs with a typed result and provenance contract.
8. Cancel a plan when a deadline or budget is exceeded and preserve a useful partial result.

## Evaluation and observability

9. Define task success separately from tool success and language quality.
10. Build trace-level debugging for prompt, tool, state, latency, tokens, and outcome.
11. Create a regression set from failed trajectories and prevent reward hacking.
12. Compare agent variants when success improves but cost and latency worsen.

## Reliability and human oversight

13. Handle a failing tool with backoff, fallback, and user-visible uncertainty.
14. Design approval checkpoints for irreversible or high-impact actions.
15. Detect loops, repeated tool calls, and unproductive reflection.
16. Resume a multi-turn task after process restart or partial tool completion.

## Security

17. Treat tool output and retrieved text as untrusted input against prompt injection.
18. Enforce per-tool permissions and tenant isolation across delegated agents.
19. Prevent secret exfiltration and unsafe output from becoming a tool argument.
20. Design a kill switch, audit trail, and incident response for unsafe autonomy.

Evaluation follow-up: connect Q9–Q12 and Q20 to the
[cross-domain evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

The required themes are retries/idempotency, memory, permissioning, recovery,
tracing, task success, budgets, unsafe output, and handoffs. Deeper references
are in `../concepts/26-error-recovery.md`, `../concepts/29-agent-evals.md`,
`../concepts/31-observability-for-agents.md`, and `../concepts/60-agent-security-sandboxing.md`.
