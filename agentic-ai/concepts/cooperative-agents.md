# Cooperative Agents

## TL;DR
Peer agents collaborate: bidirectional communication, shared goals, negotiation. No hierarchy; agents are equals.

## Core Intuition
Equals working together: agent A needs help, asks B, both contribute to solution.

## How It Works
```
Agent1 <--> Agent2 <--> Agent3
    ↓         ↓         ↓
Shared blackboard (state, observations)
```

**Example:**
- Agent1: "I need data on user behavior"
- Agent2: "I can provide that, but need product metadata"
- Agent3: "I have metadata, what format do you need?"
- Agreement on format, exchange info, solve together

## Key Properties / Trade-offs
- Flexibility: agents negotiate, adapt
- Communication: more overhead than hierarchy
- Robustness: if one agent fails, others continue

## Common Mistakes / Gotchas
- **Too many agents:** O(N²) communication. Keep to 3-5.
- **Deadlock:** agent A waits for B, B waits for A. Add timeouts.
- **No shared goals:** agents pull in different directions. Align goals first.

## Interview Quick-Reference
**Cooperative?** Peer agents, bidirectional communication, negotiate decisions, shared goals.

## Related Topics
- [Hierarchical Agents](hierarchical-agents.md) — manager-worker alternative
- [Agent Communication](agent-communication.md) — protocols for peer communication

## Resources
- [Multi-Agent Systems Design](https://arxiv.org/abs/1908.03265)
