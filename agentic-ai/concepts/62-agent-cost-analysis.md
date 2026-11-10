# Agent Cost Analysis
**TL;DR:** Agent cost analysis attributes model, tool, retrieval, infrastructure, retry, and human-review costs to completed outcomes.

## Prerequisites
Token accounting, service observability, and basic unit economics.

## Learning Outcomes
- Calculate full trajectory cost per successful task.
- Separate fixed capacity costs from variable task costs.
- Set budget-aware stopping, routing, and alert thresholds.

## Core Intuition
A taxi fare is not just the fuel used while moving; it also includes empty trips and time waiting. Agent cost per task must count failed attempts, repeated tool calls, context growth, and review work. A cheap call that never completes the task is not an economical system.

## How It Works
The operating flow identifies the task state, selects a bounded strategy, gathers only the evidence needed for that decision, validates the result, and records the outcome. Cost attribution should include the tail, not only expected usage. A rare tool loop or unusually large history can dominate the monthly bill and degrade shared capacity. Define budget envelopes for task classes, send early warnings as a trajectory approaches its limit, and preserve enough budget to validate and report a safe partial result. Cost controls are policy decisions: communicate when the agent stops for budget rather than returning a misleading success. Reconcile trace estimates with provider or infrastructure billing.
Assign a trace identifier to every user task and record token counts by model and direction, tool calls, retrieval operations, retries, elapsed time, and reviewer minutes. Convert usage to cost using a versioned price sheet or internal serving cost model. Allocate shared infrastructure carefully: report both marginal variable cost and fully loaded cost per successful task. For multi-step agents, enforce a budget at the controller so a retry loop cannot consume unbounded spend. Route simple tasks to smaller models only when quality tests support it.
Cost attribution should include the tail, not only expected usage. A rare tool loop or unusually large history can dominate the monthly bill and degrade shared capacity. Define budget envelopes for task classes, send early warnings as a trajectory approaches its limit, and preserve enough budget to validate and report a safe partial result. Cost controls are policy decisions: communicate when the agent stops for budget rather than returning a misleading success. Reconcile trace estimates with provider or infrastructure billing.
Track median and tail cost per attempted and successful task, cost by task class and model, retry waste, tool error rate, and quality at each budget tier. Forecast from observed task mix, not a single average query. Run budget stress scenarios for longer contexts, provider outages, and traffic growth. Report quality-cost frontiers so product owners choose explicitly; do not silently trade away task success to satisfy a token target.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Large model every step | Simple policy and strong capability | Highest variable spend |
| Tiered model routing | Lower cost on easy cases | Router errors and model variance |
| Caching repeated work | Cuts duplicate computation | Staleness and privacy concerns |
| Hard budget cutoff | Bounds worst-case spend | May stop before completion |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def task_cost(model_cost, tool_costs, review_cost):
    return model_cost + sum(tool_costs) + review_cost

assert task_cost(0.02, [0.01, 0.01], 0.05) == 0.09
```

## Failure Modes
- Retries inflate hidden cost: report attempts and cost per successful outcome, not just first-call usage.
- Price assumptions are stale: version pricing inputs and reconcile estimates with invoices.
- Budget cutoff creates silent partial work: return explicit incomplete status and safe resume state.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| What is the right cost denominator? | Cost per successful task, alongside per-attempt and quality metrics, because failed trajectories still consume resources. Design follow-up: how do you allocate shared infrastructure? |
| How do you control runaway agents? | Enforce token, call, time, and monetary budgets in the controller with explicit stop behavior. Debug follow-up: inspect repeated state transitions. |
| When use a smaller model? | When slice-level evaluation shows acceptable quality and savings after routing and fallback costs. |

## Related Concepts
See [Cost optimization](34-agent-cost-optimization.md), [agent evaluation](59-agent-evaluation-metrics.md), [agent monitoring](30-agent-monitoring.md).

## References
- [platform.openai.com](https://platform.openai.com/docs/guides/production-best-practices)
- [www.nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)

## Further Topic Detail
## Detailed Explanation

Agent cost analysis quantifies the expenses of running agent-based systems, which differs from standard inference costs because agents perform variable amounts of computation depending on task complexity. A simple task might complete in one step (minimal cost), while complex tasks might involve multiple tool calls, retries, and long context windows (expensive). Understanding and controlling costs is crucial for viability: an agent system that works perfectly but costs $10 per query isn't commercially viable.

Cost components include: (1) LLM API calls (primary cost, varies by model size and tokens), (2) Tool usage (external APIs, data retrieval), (3) Retrieval operations (searching knowledge bases), (4) Token overhead (prompt structure, examples), (5) Failures requiring retry (wasted computation). Cost optimization strategies include: model selection (use smaller models when sufficient), step reduction (design agents to take fewer steps), tool efficiency (fast tools are cheaper), and caching (avoid redundant computations). Some systems use tiered approaches: cheap fast agents for simple queries, expensive capable agents only for complex queries.

Understanding agent cost analysis requires systems thinking about the economics of AI systems. Many novel agents are technically impressive but economically unviable. Cost considerations should drive architecture decisions, not be an afterthought.


## Core Intuition

Imagine an employee who must make a decision: they can choose to spend 5 minutes thinking and decide, or spend 2 hours researching and decide better. Agents have similar choices: fast but potentially wrong (cheap), or slow and careful (expensive). Cost analysis is choosing the right level of effort for each decision.


## How It Works

1. Token counting: input tokens + output tokens per interaction
2. Pricing: cost-per-1K-tokens varies by model (GPT-4: $0.03, GPT-3.5: $0.002)
3. API calls: each tool call, retrieval, external service adds cost
4. Cost per task: sum of all tokens and API calls for one task
5. Aggregation: multiply by frequency to get daily/monthly/yearly cost
6. Optimization: reduce tokens (shorter prompts), use cheaper models (GPT-3.5 vs GPT-4)
7. Monitoring: track cost in real-time, alert on anomalies




## Interview Q&A


**Q: How do you reduce agent token usage?**
A: Prompt optimization: remove verbose instructions, use examples efficiently. Model selection: GPT-3.5 cheaper than GPT-4 (trade accuracy for cost). Caching: reuse computations (store embeddings, cache prompts). Summarization: compress context. Typical reduction: 30-50% with optimization.

**Q: What's the cost difference between models?**
A: GPT-4: $0.03/1K input, $0.06/1K output tokens. GPT-3.5: $0.002/1K input, $0.004/1K output. Claude: similar to GPT-4. Trade-off: GPT-4 better quality but expensive. Use GPT-3.5 for high-volume low-stakes, GPT-4 for critical tasks.

**Q: How do you handle cost anomalies?**
A: Monitor: track per-user, per-task costs. Alert: if exceeds threshold (e.g., >$1 per task). Investigate: is it legitimate (complex task) or bug (infinite loop, hallucination)? Limits: set per-user caps to prevent runaway spending.

**Q: Should you optimize for cost or quality?**
A: Context: cost-quality trade-off. High-volume: optimize cost (use GPT-3.5, short context). Mission-critical: optimize quality (use GPT-4, long context). Most: balance (use hybrid, optimize both). Measure: cost per unit quality achieved.

**Q: How do you forecast agent costs?**
A: Estimate: average tokens per task × tasks per day × days per month × price-per-token. Sensitivity: how do costs scale with usage, model, context? Budget: plan for growth (2-3x). Monitor: track actual vs. forecast, adjust as needed.
