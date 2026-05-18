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
