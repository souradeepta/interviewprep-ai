# Agent State-Recovery Remediation

Use this page after **Q4: “Recover a workflow from a stale checkpoint.”** A
checkpoint is a recovery boundary, not proof that an external side effect did
or did not happen.

## Durable state contract

Persist a schema/version, task identity, checkpoint time, bounded conversation
messages, durable task state, completed action IDs, and any reconciliation
records. Keep message trimming separate from task progress so context limits
cannot erase completed actions. Restore into a deep copy and validate roles,
content types, state shape, and version before execution.

Reject stale checkpoints using an explicit maximum age. Do not silently restore
an old plan against changed permissions, tools, or business data. A migration
must be versioned and observable; otherwise require a fresh task.

## Resume protocol

1. Load and validate the checkpoint schema and task owner.
2. Compare checkpoint age, tool-contract versions, authorization, and deadline.
3. For each planned action, consult durable completion records by action ID.
4. Skip actions marked `completed` and return their stored results.
5. Reconcile actions in `unknown` state with the downstream system; never blind
   retry an irreversible action.
6. Execute only pending actions within the remaining budget, checkpointing
   durable progress after each successful side effect.
7. Return a structured partial result if cancellation, expiry, or a safety
   boundary prevents completion.

## Tests and observability

Test malformed roles/content, non-dictionary task state, stale timestamps,
version mismatch, deep-copy isolation, trimmed messages with retained action
IDs, duplicate resume calls, and post-side-effect exceptions. Trace checkpoint
ID, schema version, action state transitions, lease owner, reconciliation
result, and the reason for any skipped or cancelled action.

## Failure boundaries

- A valid checkpoint does not make an arbitrary external API exactly once.
- Conversation history is context, not the source of truth for side effects.
- Checkpoint writes must be durable before reporting a completed action.
- Recovery must re-check authorization and current tool contracts.

## References

- [Multi-turn conversation concept](../concepts/61-multi-turn-conversation.md)
- [Reference implementation](../implementations/61-multi-turn-conversation.py)
- [Agent coding exercises](agent-coding-exercises.md)
