# Simulation, Log Parsing, and State Machines

Many current coding questions are not a named textbook algorithm. They parse events, update state, enforce ordering, and ask for a final or intermediate result. Treat the input as a stream and make the state explicit. Correctness comes from a precise event contract and a small set of invariants, not from clever control flow.

## Recognition criteria

Use simulation when the prompt describes a sequence of operations and asks what remains afterward: account balances, active sessions, parking allocation, task queues, rate limits, inventory, or a parser. Use a state machine when each event changes which transitions are valid. Use a heap or queue when the next action depends on event time or priority; use a map when state is keyed by user, object, or identifier; use a stack when events are nested. Sort events only if the prompt says input order is arbitrary and provides timestamps that define the true order.

Before coding, document the record schema, ordering rule, duplicate policy, invalid-event behavior, and timestamp tie semantics. Distinguish input order from event time. If records can arrive late, a one-pass chronological simulation may be invalid; the system may need a watermark, reorder buffer, or correction mechanism. A coding-interview solution can state the simpler assumption, but should make it explicit.

```python
def active_sessions(events):
    active = set()
    completed = []
    for timestamp, user, action in sorted(events):
        if action == "start":
            if user in active:
                raise ValueError(f"duplicate start: {user}")
            active.add(user)
        elif action == "stop" and user in active:
            active.remove(user)
            completed.append((user, timestamp))
        else:
            raise ValueError(f"invalid event: {timestamp, user, action}")
    return completed, active
```

This example rejects a stop without an active session and a duplicate start. Another product could choose idempotent duplicate handling, but that rule must be deliberate. If event ordering requires a custom tie-break, include it in the sort key; Python’s default tuple ordering is not an adequate substitute for a stated domain rule.

## Worked trace

Take ordered events `(1, A, start)`, `(2, B, start)`, `(3, A, stop)`, `(4, B, stop)`. Initially `active = {}`. After the first event it is `{A}`; after the second `{A,B}`; the third removes A and emits `(A,3)`; the final event removes B and emits `(B,4)`. The invariant is that `active` contains exactly the users with a start event not yet paired with a valid stop. If the third event were another stop for A, the simulator should reject or explicitly ignore it according to contract.

## Complete problem 1: transaction reconciliation

**Prompt:** Given ledger events `(event_id, account, delta_cents)`, return each account’s ending balance while applying an event ID at most once. **Approach:** Iterate in defined order, validate the identifier and integer amount, skip already-seen IDs, and add each accepted delta to a per-account map. Use integer cents to avoid floating point. **Example:** Events `(x,A,+500)`, `(y,A,-125)`, `(x,A,+500)` yield A=375 because x is a duplicate. **Correctness:** The `seen` set ensures every logical event contributes once; the account map is the sum of accepted deltas. **Complexity:** O(n) expected time and O(n+a) space for n events and a accounts. In production, deduplication and balance update need an atomic durable transaction; an in-memory set only models the logic.

## Complete problem 2: parking lot allocation

**Prompt:** Cars arrive and depart at specified times; compute the minimum number of spaces needed, with a space reusable when a departure time equals another car’s arrival. **Approach:** Sort arrivals and departures separately. For each arrival, remove all departures `<= arrival` from the active count, then add one and track the maximum. **Example:** `[1,4]` and `[4,6]` use one space under this reusable half-open boundary rule; changing the departure comparison to `<` makes the boundary closed and requires two. **Correctness:** Active count represents the cars occupying spaces at the current event time, so its maximum is exactly the capacity required. **Complexity:** O(n log n) time and O(n) space for sorting. If each car needs a specific space assignment, maintain a min-heap of free space IDs as well as end times.

## State machines and event logs

Model legal phases explicitly. A job may transition from `queued` to `running` to `succeeded` or `failed`; reject transitions such as `succeeded -> running` unless retries are an explicit feature. For small state spaces, a transition table makes policy inspectable. For nested events such as function-call logs or markup, a stack captures the most recent unmatched opener. For rate limiting, a per-key queue of timestamps supports a sliding time window, while a token bucket stores both token count and last-refill time.

Decide whether the state is a snapshot or can be reconstructed by replaying events. Replay is useful for audit and debugging but needs idempotency, stable ordering, and schema versioning. If events can be delivered more than once, define an idempotency key. If they can be late, choose whether to reject, buffer, or compensate. These requirements often matter more than the map or queue selected in an interview answer.

## Common failure cases and diagnostics

- **Sorting meaningful sequence:** Sorting by timestamp can reorder a stream where input order is authoritative or timestamps tie. Preserve input order or add a sequence number; test equal-time events.
- **Duplicate delivery:** A retry can double-apply a payment, counter, or inventory change. Deduplicate by stable event key and test repeated input; explain persistence and atomicity limits.
- **Missing counterpart:** A stop/end event may arrive without a start, or a start may remain open at end of input. Decide whether to reject, report incomplete state, or allow it; test both cases.
- **Transition outside the state machine:** Updating the state without checking its prior phase permits impossible sequences. Centralize allowed transitions and assert state invariants after each event.
- **Timestamp boundary ambiguity:** Processing completion before arrival versus arrival before completion changes peak capacity. State closed/half-open semantics and test a pair touching at exactly one timestamp.
- **Floating-point currency or time:** Repeated decimal addition can create comparison drift. Use integer minor units or integer ticks where the domain permits.
- **Late or out-of-order events:** A simple sorted batch hides online arrival behavior. State whether the input is a complete batch; for streams, describe buffering, watermarks, and correction policy.
- **Unbounded logs or state:** Keeping every event in memory may violate constraints. Aggregate incrementally when ordering permits, expire state by time, or use external persistence while maintaining required invariants.

## Complexity and selection summary

For n events, a single ordered pass is O(n); sorting unordered events costs O(n log n). Maps and sets provide expected O(1) lookup/update, queues support FIFO work, heaps support O(log n) next-event selection, and stacks support O(1) push/pop. Space is proportional to the live state plus any required output or deduplication history. State explicitly whether that history can be bounded. Choose a richer structure only for an operation the prompt needs; a sorted array plus scan is often clearer when all events are static.

## Practice list

Practice logger rate limiting, transaction reconciliation, parking-lot allocation, task scheduling, nested path parsing, event deduplication, and session reconstruction. Related LeetCode themes include 1603 (design parking system), 359 (logger rate limiter), 933 (number of recent calls), 981 (time-based key-value store), 1094 (car pooling), and 1353 (maximum number of events). For every prompt, write the state variables, legal transitions, ordering and duplicate policy, then test empty input, ties, invalid transitions, and unfinished state.
