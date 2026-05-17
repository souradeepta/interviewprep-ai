# Agent Monitoring

## TL;DR
Monitor agents in production: success rate, tool accuracy, latency, cost. Metrics: task completion, hallucinations, error rates. Alert on degradation.

## Core Intuition
Agents are fragile. Monitor closely for failures, degradation, unexpected behavior.

## How It Works
**Metrics:**
- Success rate: % of tasks completed
- Tool accuracy: % of valid tool calls
- Latency: time per task
- Cost: $ per task
- Hallucination rate: invalid claims

**Alerting:**
- Success < 90%? Alert
- Latency > 5s? Alert
- Tool errors > 10%? Alert

## Common Mistakes / Gotchas
- **Only success/fail:** doesn't catch partial failures
- **No cost tracking:** agent expensive but success high
- **Ignoring hallucinations:** success but claims are false

## Interview Quick-Reference
**Monitor agents?** Success rate, tool accuracy, latency, cost, hallucination rate.

## Related Topics
- [Agent Evals](agent-evals.md) — testing agents
- [Monitoring & Observability](../system-design/patterns/monitoring-and-observability.md)

## Resources
- [MLOps for Agents](https://arxiv.org/abs/2309.14090)
