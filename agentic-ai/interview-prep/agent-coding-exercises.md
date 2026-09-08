# Agent Coding Exercises

Each exercise is API-free and deterministic. Use fake tools and an injected
clock; do not sleep or call a model.

1. **Bounded retry:** retry transient failures with exponential backoff,
   idempotency keys, and a maximum attempt count.
2. **Tool-result validation:** validate a typed result, reject unsafe or stale
   values, and preserve the trace of the rejection.
3. **Checkpoint/restore:** serialize state after each side effect and resume
   without duplicating completed work.
4. **Conversation state:** enforce message and memory budgets while retaining
   required task state across turns.
5. **Cancellation:** stop a plan at a deterministic deadline and return a
   partial result with cleanup actions.
6. **Budget accounting:** aggregate tokens, tool cost, and latency and reject
   the next action when any budget is exhausted.

Reference implementations for the reliability behaviors are the three files
under `../implementations/26-error-recovery.py`, `61-multi-turn-conversation.py`,
and `64-real-time-agent-systems.py`.
