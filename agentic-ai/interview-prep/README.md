# Agent Interview Practice

Use the [20-question scenario bank](agent-interview-questions.md) for 5–10
minute design/debugging rounds and the [coding exercises](agent-coding-exercises.md)
for deterministic reliability practice. All tasks use fake tools, injected
clocks, and bounded state; no network or credentials are required.

Each answer should cover tool contracts, state and invariants, retries and
idempotency, quality/task success, cost/latency, observability, human
oversight, and security. Read the linked concept only after attempting the
prompt.

Focused remediation: [retry and idempotency](retry-idempotency-remediation.md)
for Q2 and [state recovery](state-recovery-remediation.md) for Q4, including
the `unknown` state after an ambiguous side effect.
