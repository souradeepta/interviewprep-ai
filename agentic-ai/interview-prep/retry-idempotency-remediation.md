# Retry and Idempotency Remediation

Use this page after **Q2: “Make retries safe with idempotency keys.”** A retry
policy controls attempts; it does not make an arbitrary external side effect
exactly once.

## State machine

Persist one record per operation key:

| State | Meaning | Safe next action |
|---|---|---|
| `pending` | No terminal result is known | Claim/execute once under a lease |
| `completed` | Durable result is recorded | Return the stored result; do not execute |
| `unknown` | The request may have reached the side effect, but the response was lost | Reconcile by provider lookup or human workflow; never blind-retry |

The operation key must cover the business identity and an input hash. Reusing a
key with different arguments is a conflict, not a second attempt.

## Safe attempt sequence

1. Validate the tool input and derive `(operation_key, input_hash)`.
2. Atomically create or read the durable record. Return a stored `completed`
   result immediately.
3. Mark `pending` with an owner/lease and attempt number.
4. Call the downstream service with its own idempotency key when supported.
5. On a confirmed success, persist the result and mark `completed` before
   replying to the caller.
6. On a classified transient failure with no evidence of side effect, retry
   with bounded exponential backoff and jitter.
7. On timeout, connection loss, or an exception after the provider may have
   acted, mark `unknown` and require reconciliation.

Exactly-once behavior requires downstream idempotency or a transactionally
coupled outbox/consumer. A local lock alone cannot prevent duplication after a
process crash.

## Validation and observability

Test success, retry exhaustion, permanent-error propagation, duplicate calls,
input-hash conflicts, and post-side-effect exceptions. Record operation key,
attempt, state transition, provider request ID, retry reason, and reconciliation
result. Alert on growth in `unknown`, lease expiry, and repeated conflicts.

## Failure boundaries

- Retry only errors classified as transient; do not retry validation, auth, or
  permission failures.
- Bound attempts, elapsed time, and total cost. Backoff must not exceed the
  caller’s deadline.
- Do not report success before the durable completion record is written.
- Do not automatically rerun an `unknown` payment, deletion, email, or other
  irreversible side effect.

## Related references

- [Error recovery concept](../concepts/26-error-recovery.md)
- [Reference implementation](../implementations/26-error-recovery.py)
- [Agent coding exercises](agent-coding-exercises.md)
