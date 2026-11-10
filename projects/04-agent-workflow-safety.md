# Project 04: Agent Workflow Safety

**Timebox:** 8 hours. **Role path:** Applied AI / agent engineer. **Prerequisites:** [Agent loops](../agentic-ai/concepts/02-agent-loops.md), [tool use](../agentic-ai/concepts/03-tool-use.md), [agent testing](../agentic-ai/concepts/28-agent-testing.md), [agent security sandboxing](../agentic-ai/concepts/60-agent-security-sandboxing.md), [human-agent collaboration](../agentic-ai/concepts/37-human-agent-collaboration.md).

## Problem brief

Implement a bounded local assistant workflow that can look up fictional support policy and prepare (but never execute) a refund request. It must distinguish read-only lookup from a proposed side effect, require approval for the latter, handle duplicate requests idempotently, and retain an audit trace without sensitive payloads.

## Data contract

Use synthetic tickets, fictional policy text, and a local mock refund ledger. Inputs have request ID, policy topic, and a fictional amount. Include unknown topics, malformed amounts, duplicate request IDs, and adversarial text that asks the agent to bypass approval. No network calls or real orders/payments are allowed.

## Deliverables

1. A state machine with typed tool input/output and a maximum-step limit.
2. Deterministic tests for policy lookup, abstention, duplicate handling, approval gating, and adversarial input.
3. Structured audit events with tool name, request ID, decision, and outcome; exclude raw ticket text.
4. A safety note describing what the mock cannot establish and the human authority boundary.

## Local success criterion

`python3 solution.py` exits zero and its assertions verify: unknown policy leads to escalation; a refund remains `proposed` until approved; replaying one request ID causes at most one ledger effect; prompt text cannot bypass approval; and the trace omits ticket text. The mock must not contact external services.

## Evaluation rubric — 100 points

| Dimension | Points | Evidence |
|---|---:|---|
| Engineering and bounded workflow | 20 | Typed interfaces, step bound, deterministic state transitions, local-only tools. |
| Measurement and evaluation | 15 | Explicit success/safety metrics with appropriate denominators and test coverage. |
| Failure handling and safety | 30 | Abstention, approval gate, idempotency, input validation, prompt-injection case. |
| Communication and auditability | 20 | Reconstructable trace without unnecessary sensitive content; honest mock limits. |
| Interview debrief | 15 | Explains authority boundary, failure mode, and production controls still required. |

## Extension prompt

Simulate a crash between recording approval and applying the local ledger update. Propose an atomic persistence design and write a test that detects duplicate or lost effects.
