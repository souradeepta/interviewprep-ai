# Agent Interview Questions

For every 5–10 minute prompt, state assumptions, tool/state design, success
metric, failure modes, budget, observability, safety boundary, and one follow-up.

## Tool contracts and state

### Q1 — Define a typed tool contract that makes invalid arguments and side effects explicit.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Define a typed tool contract that makes invalid arguments and side effects explicit. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q2 — Make retries safe with idempotency keys and a bounded retry policy.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Make retries safe with idempotency keys and a bounded retry policy. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [retry and idempotency remediation](retry-idempotency-remediation.md), [agent concepts](../concepts/), and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q3 — Choose memory boundaries between conversation, task, user, and durable state.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Choose memory boundaries between conversation, task, user, and durable state. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q4 — Recover a workflow from a stale checkpoint without repeating a side effect.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Recover a workflow from a stale checkpoint without repeating a side effect. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [state recovery remediation](state-recovery-remediation.md), [agent concepts](../concepts/), and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


## Planning and orchestration

### Q5 — Compare ReAct, a fixed workflow, and planner/executor decomposition for a support task.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Compare ReAct, a fixed workflow, and planner/executor decomposition for a support task. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q6 — Route tasks to specialists while bounding fan-out, cost, and context duplication.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Route tasks to specialists while bounding fan-out, cost, and context duplication. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q7 — Design multi-agent handoffs with a typed result and provenance contract.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design multi-agent handoffs with a typed result and provenance contract. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q8 — Cancel a plan when a deadline or budget is exceeded and preserve a useful partial result.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Cancel a plan when a deadline or budget is exceeded and preserve a useful partial result. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


## Evaluation and observability

### Q9 — Define task success separately from tool success and language quality.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Define task success separately from tool success and language quality. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q10 — Build trace-level debugging for prompt, tool, state, latency, tokens, and outcome.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Build trace-level debugging for prompt, tool, state, latency, tokens, and outcome. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q11 — Create a regression set from failed trajectories and prevent reward hacking.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Create a regression set from failed trajectories and prevent reward hacking. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q12 — Compare agent variants when success improves but cost and latency worsen.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Compare agent variants when success improves but cost and latency worsen. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


## Reliability and human oversight

### Q13 — Handle a failing tool with backoff, fallback, and user-visible uncertainty.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Handle a failing tool with backoff, fallback, and user-visible uncertainty. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q14 — Design approval checkpoints for irreversible or high-impact actions.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design approval checkpoints for irreversible or high-impact actions. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q15 — Detect loops, repeated tool calls, and unproductive reflection.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Detect loops, repeated tool calls, and unproductive reflection. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q16 — Resume a multi-turn task after process restart or partial tool completion.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Resume a multi-turn task after process restart or partial tool completion. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


## Security

### Q17 — Treat tool output and retrieved text as untrusted input against prompt injection.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Treat tool output and retrieved text as untrusted input against prompt injection. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q18 — Enforce per-tool permissions and tenant isolation across delegated agents.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Enforce per-tool permissions and tenant isolation across delegated agents. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q19 — Prevent secret exfiltration and unsafe output from becoming a tool argument.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Prevent secret exfiltration and unsafe output from becoming a tool argument. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q20 — Design a kill switch, audit trail, and incident response for unsafe autonomy.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design a kill switch, audit trail, and incident response for unsafe autonomy. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [agent concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


Evaluation follow-up: connect Q9–Q12 and Q20 to the
[cross-domain evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

The required themes are retries/idempotency, memory, permissioning, recovery,
tracing, task success, budgets, unsafe output, and handoffs. Deeper references
are in `../concepts/26-error-recovery.md`, `../concepts/29-agent-evals.md`,
`../concepts/31-observability-for-agents.md`, and `../concepts/60-agent-security-sandboxing.md`.
