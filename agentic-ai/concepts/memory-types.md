# Memory Types

## TL;DR
Agents need multiple memory systems: short-term (context window), long-term (vector DB/database), episodic (action logs), semantic (facts/KB). Each serves different purpose; trade-off capacity vs. relevance vs. latency.

## Core Intuition
Humans have: immediate context (current conversation), long-term memories (past events), facts about world (semantic), and logs of what they did. Agents need all to act intelligently.

## How It Works

**Short-Term Memory (Context Window):**
- Capacity: 4K-200K tokens (current LLMs)
- Content: current conversation, recent events
- Latency: immediate
- Use: planning current action

**Long-Term Memory (External Storage):**
- Vector DB or traditional database
- Content: past interactions, embeddings, summaries
- Latency: milliseconds to seconds
- Retrieval: semantic search or SQL queries

**Episodic Memory (Action Logs):**
- Log of past actions + outcomes
- "On 2024-01-15, I called X with params Y, got Z"
- Use: reflection, learning from mistakes
- Supports debugging

**Semantic Memory (Knowledge Base):**
- Facts, rules, world knowledge
- "Paris is capital of France"
- Structured (knowledge graphs) or unstructured (text)
- Use: grounding, validation

## Key Properties / Trade-offs

| Type | Capacity | Latency | Recency | Cost |
|------|----------|---------|---------|------|
| Short-term | Small (4K-200K) | 0ms | Fresh | Included |
| Long-term | Huge (unlimited) | 1-100ms | Stale | $ (storage) |
| Episodic | Medium (logs) | 1-10ms | Complete history | $ (storage) |
| Semantic | Large (KB) | 1-100ms | Static | $ (storage+index) |

## Common Mistakes / Gotchas

- **Exceeding context window:** too much memory → truncation → info loss. Summarize/chunk.
- **Stale long-term memory:** old interactions → outdated info. Re-index periodically.
- **No episodic logs:** can't learn from past. Always log actions.
- **Ungrounded semantic KB:** outdated facts → hallucinations. Validate and update KB.
- **Memory retrieval latency:** semantic search adds 10-100ms. Cache common queries.

## Code Example

```python
from collections import deque

class AgentMemory:
    def __init__(self, context_window_size=100):
        self.short_term = deque(maxlen=context_window_size)  # FIFO
        self.semantic_kb = {"Paris": "capital of France"}
        self.episodic_log = []  # All actions
    
    def add_short_term(self, event):
        """Add to context window."""
        self.short_term.append(event)
    
    def add_episodic(self, action, result):
        """Log action + outcome."""
        self.episodic_log.append({"action": action, "result": result})
    
    def query_semantic(self, fact):
        """Lookup fact."""
        return self.semantic_kb.get(fact, None)
    
    def get_context(self):
        """Build context for LLM."""
        return "\n".join(str(e) for e in self.short_term)

# Usage
memory = AgentMemory()
memory.add_short_term("User asked: What is Paris?")
memory.add_episodic("query_kb(Paris)", "capital of France")
context = memory.get_context()
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Memory types?" | Short-term (context), long-term (vector DB), episodic (logs), semantic (KB). Each serves different need. |
| "Which to use?" | Use all. Short-term for immediate context, long-term for background, episodic for learning. |
| "Context window limit?" | Summarize or chunk old info. Use hierarchical retrieval (coarse → fine). |
| "Memory retrieval latency?" | 1-100ms typical. Cache if needed, optimize queries, use approximate search. |

## Related Topics
- [Agent Memory Management](agent-memory-management.md) — managing multiple memory systems
- [Context Window Management](context-window-management.md) — optimizing limited context
- [Agent Loops](agent-loops.md) — loops update memory

## Resources
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- [Memory in Large Language Models](https://arxiv.org/abs/2310.01738)
