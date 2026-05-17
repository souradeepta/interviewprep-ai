# Context Window Management

## TL;DR
Manage limited context (4K-200K tokens). Strategies: summarize old messages, prune irrelevant info, hierarchical retrieval, sliding window.

## Core Intuition
Context is limited. Long conversations exceed window. Keep recent + important info.

## How It Works
**Strategies:**
1. **Summarize:** compress old messages → summary
2. **Prune:** remove unimportant exchanges
3. **Retrieval:** only fetch relevant memory (not all)
4. **Sliding window:** keep recent k messages, discard old

**Example:**
```
Messages: 100 (tokens exceed window)
Recent: 20 (keep)
Summarize: 80 into summary (10 tokens)
Context: recent + summary fits
```

## Common Mistakes / Gotchas
- **Summarization loss:** compress to summary, lose nuance
- **No ranking:** remove random messages → miss important
- **Always keep all:** exceed window → error

## Interview Quick-Reference
**Context management?** Summarize, prune, hierarchical retrieval, sliding window. Keep important recent.

## Related Topics
- [Memory Types](memory-types.md)
- [Context Window](../llm/concepts/context-window.md) — LLM context

## Resources
- [Generative Agents](https://arxiv.org/abs/2304.03442) — memory management in agents
