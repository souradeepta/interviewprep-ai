# Hierarchical Agents

## TL;DR
Manager agent delegates to specialist workers. Each worker expert in domain. Manager breaks task, coordinates, aggregates results.

## Core Intuition
Complex problems benefit from specialization. Manager (coordinator), analysts (data), engineers (implementation). Each expert, working together.

## How It Works
```
User task → Manager
  ↓
Manager: break into subtasks
  ↓
Worker1 (analyst) → task A
Worker2 (engineer) → task B
Worker3 (strategist) → task C
  ↓
Results → Manager aggregates
  ↓
Final response to user
```

**Example:**
- Manager: "Design recommendation system for streaming"
- Workers:
  - Analyst: "User interaction data shows patterns..."
  - Engineer: "We can use collaborative filtering, implement as..."
  - Strategist: "Trade-offs: real-time vs batch, cost vs quality..."
- Manager: Synthesizes into design

## Key Properties / Trade-offs
- Specialization: high quality per task
- Coordination overhead: manager time, communication
- Scaling: add workers as needed

## Common Mistakes / Gotchas
- **Poor delegation:** unclear subtasks → workers confused
- **No coordination:** workers work in isolation → inconsistent
- **Bottleneck:** manager slow → whole system slow

## Interview Quick-Reference
**Hierarchical?** Manager delegates to specialist workers. Good for decomposed, specialized tasks.

## Related Topics
- [Multi-Agent Systems](multi-agent-systems.md) — broader patterns
- [Cooperative Agents](cooperative-agents.md) — peer alternative

## Resources
- [Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296)
