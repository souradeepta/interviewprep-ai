# Simulation, Log Parsing, and State Machines

Many current OA questions are not a named textbook algorithm. They parse
events, update state, enforce ordering, and ask for a final or intermediate
result. Treat the input as a stream and make the state explicit.

## Reliable workflow

1. Define the event schema and invalid-input policy.
2. Sort only when the prompt says events are unordered; preserve input order
   when it is meaningful.
3. Keep independent state in named maps, queues, or counters.
4. Write invariants, such as “balance never goes negative” or “each job is
   active at most once.”
5. Test duplicate events, missing close events, ties, empty input, and the
   largest timestamp or identifier.

```python
def active_sessions(events):
    active = set()
    completed = []
    for timestamp, user, action in sorted(events):
        if action == "start":
            active.add(user)
        elif action == "stop" and user in active:
            active.remove(user)
            completed.append((user, timestamp))
        else:
            raise ValueError(f"invalid event: {timestamp, user, action}")
    return completed, active
```

For billing and rate limits, prefer integer units and monotonic timestamps to
avoid floating-point boundary bugs. For queues, decide whether work arriving
at time `t` is eligible before work completing at `t`. For hierarchical logs,
use a stack or parent map rather than repeatedly scanning all previous lines.

Practice themes include logger rate limiting, transaction reconciliation,
parking-lot allocation, task scheduling, nested path parsing, and event
deduplication. These test correctness, complexity, and edge-case discipline
as much as algorithm selection.
