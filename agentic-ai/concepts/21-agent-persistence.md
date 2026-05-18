# Agent Persistence

## Detailed Explanation

Agent persistence enables agents to save and restore state across sessions. Mechanisms: save agent state (memory, learned lessons, configuration) to storage, reload on restart. Advantages: agent continues learning across sessions, survives crashes, enables agent handoff (one agent passes state to another). Challenges: state consistency (save/load might have bugs), versioning (old states may be incompatible with new agent), storage overhead (some agents have large state). Use cases: long-running agents (days/weeks), agents learning over time, agent recovery (restart after crash), agent migration (move between servers). Trade-offs: save frequently (safety, performance cost) vs infrequently (performance, loss risk).

## Core Intuition

Save game in video game. Quit game, resume later, continue from where you left off. Agent persistence is the same—save agent progress, reload to continue.

## How It Works

1. **State Definition** — Define what to persist (memory, learned patterns, config)
2. **Serialization** — Convert to JSON/pickle
3. **Storage** — Save to file/database
4. **Versioning** — Track state versions for compatibility
5. **Loading** — Reload state on restart
6. **Migration** — Handle schema changes across versions

## Interview Q&A

**Q: What is the difference between agent memory and agent state, and how do you manage each?**
A: State is operational context needed for the current task: current step in a workflow, intermediate results, open tool calls. Memory is accumulated knowledge that should persist across tasks: user preferences, learned facts, historical decisions. State should be stored in fast in-memory or key-value stores (Redis) with short TTL. Memory should be stored in durable persistent stores (PostgreSQL, vector DB) with explicit retention policies. Never conflate them—losing state during a task is a bug; losing memory between sessions is a design choice.

**Q: How do you implement checkpointing for long-running agent tasks?**
A: Checkpoint after each major step in the workflow, not just at the end. The checkpoint must include: current state (which step completed, what outputs were produced), the original task specification, any acquired context (fetched data, tool results), and enough information to resume without re-running completed steps. Store checkpoints in durable storage. On resume, validate checkpoint integrity (hash check) and verify the environment state still matches assumptions (e.g., documents haven't changed). Design checkpoints as complete re-entrant points, not just progress markers.

**Q: What retention and cleanup policies should govern agent persistence stores?**
A: Task state: delete after task completion + 7 days (enough for debugging). Short-term memory: rolling window of 30-90 days, or until user requests deletion. Long-term memory: indefinite, but implement user-controlled deletion. Logs: compress and archive after 30 days, delete after regulatory retention period. Set storage alerts: if persistence store exceeds size thresholds, investigate (may indicate memory leak or missing cleanup). For GDPR compliance, implement "right to be forgotten" across all persistence layers.

**Q: How do you handle conflicts between persisted memory and new contradicting information?**
A: Implement a memory update protocol: when the agent encounters information that contradicts stored memory, log the conflict with timestamps, update the memory if the new information is more authoritative (more recent, from a more reliable source), and note the previous value. Don't silently overwrite—maintain a memory version history for high-stakes facts. For user preferences, prefer the most recent statement. For factual information, prefer the most authoritative source. Alert if critical persisted facts are contradicted frequently.

**Q: What are the security risks of agent persistence and how do you mitigate them?**
A: Persistence injection: adversarial inputs that cause the agent to store malicious content in memory, later retrieved and executed. Stale credential exposure: persisted access tokens that have been revoked but the agent still uses. Over-privileged memory access: one user's data accessible to another user's agent session. Mitigate with: input sanitization before storing in memory, credential expiry handling (refresh tokens, not long-lived stored credentials), strict namespace isolation in the persistence layer, and regular security audits of what's being stored.

**Q: How do you implement efficient semantic memory retrieval for agents?**
A: Store memories with embeddings for semantic search. Use metadata indexing for exact-match queries (by date, by type). Implement memory consolidation: periodically summarize and compress old memories to reduce storage and retrieval cost. Use importance scoring to prioritize high-value memories in retrieval (recent + frequently accessed + explicitly flagged as important). Test retrieval quality: for a sample of agent tasks, verify that the relevant memories are being retrieved in the top-k results. Memory retrieval failures are often the root cause of agent behavior regressions.


## Best Practices

1. Define clear state schema
2. Version all saved state
3. Test save/load cycle
4. Encrypt sensitive state
5. Backup before major updates
6. Atomic save (all-or-nothing)
7. Monitor storage size
8. Clean up old states

## Code Examples

### Example 1: Basic State Save/Load

```python
import json
from datetime import datetime

class PersistentAgent:
    def __init__(self, state_file="agent_state.json"):
        self.state_file = state_file
        self.memory = []
        self.learned_lessons = {}
        self.load()

    def save(self):
        state = {
            "memory": self.memory,
            "lessons": self.learned_lessons,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def load(self):
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                self.memory = state["memory"]
                self.learned_lessons = state["lessons"]
        except FileNotFoundError:
            pass
```

### Example 2: Versioned State

```python
class VersionedPersistent:
    SCHEMA_VERSION = "2.0"

    def save_with_version(self):
        state = {
            "schema_version": self.SCHEMA_VERSION,
            "memory": self.memory,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def load_with_migration(self):
        with open(self.state_file, "r") as f:
            state = json.load(f)
            version = state.get("schema_version", "1.0")

            if version == "1.0":
                state = self._migrate_1_0_to_2_0(state)

            self.memory = state["memory"]

    def _migrate_1_0_to_2_0(self, old_state):
        # Handle schema changes
        return old_state
```

### Example 3: Atomic Saves

```python
import shutil
import os

class AtomicPersistent:
    def safe_save(self):
        temp_file = self.state_file + ".tmp"

        # Write to temp file
        with open(temp_file, "w") as f:
            json.dump(self.state, f)

        # Atomic rename
        shutil.move(temp_file, self.state_file)

    def backup_before_update(self):
        backup_file = f"{self.state_file}.backup"
        if os.path.exists(self.state_file):
            shutil.copy(self.state_file, backup_file)
```

## Related Concepts

- Memory Types, Agent Loops, Error Recovery, Observability
