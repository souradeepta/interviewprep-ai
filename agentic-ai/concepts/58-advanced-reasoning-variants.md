# Advanced Reasoning Variants
**TL;DR:** Reasoning variants spend extra inference to explore, verify, or revise candidate solutions; each should be selected against a task-specific quality gain.

## Prerequisites
Agent loops, evaluation design, and basic search.

## Learning Outcomes
- Compare single-path reasoning, self-consistency, tree search, and tool verification.
- Estimate added calls and latency from a reasoning policy.
- Use an external checker when correctness is decidable.

## Core Intuition
A proofreader, a second independent solver, and a branching search are different kinds of review. Asking the same model to “think harder” is not equivalent to independent verification: it can repeat the same mistaken assumption. Reasoning variants are compute allocation policies, and the right choice depends on whether the task has a verifier or only a subjective quality judgment.

## How It Works
The operating flow identifies the task state, selects a bounded strategy, gathers only the evidence needed for that decision, validates the result, and records the outcome. More reasoning is most valuable when the problem has meaningful alternatives or a reliable way to reject bad candidates. On a calculation task, execution gives a strong check; on a subjective writing task, a second sample may only express a different preference. Decompose evaluation by task type, difficulty, and failure cost. Use a budget router to apply expensive search only to cases where the estimated benefit is supported by held-out measurements. Keep traces concise and evidence-centered for auditability.
For arithmetic or code, generate a candidate and check it with execution or a trusted evaluator. For open-ended planning, branching can preserve alternatives, but the value depends on a scoring function that can distinguish promising branches. Self-consistency samples multiple completions and votes; it helps only when errors are not fully correlated and the answer space can be normalized. Reflection can correct omissions when feedback is grounded, but a model critique without evidence can merely produce a second plausible story. Keep candidate traces for diagnosis and avoid exposing private chain-of-thought; record concise evidence and decisions instead.
More reasoning is most valuable when the problem has meaningful alternatives or a reliable way to reject bad candidates. On a calculation task, execution gives a strong check; on a subjective writing task, a second sample may only express a different preference. Decompose evaluation by task type, difficulty, and failure cost. Use a budget router to apply expensive search only to cases where the estimated benefit is supported by held-out measurements. Keep traces concise and evidence-centered for auditability.
Use paired task runs to compare final correctness, verifier catches, token and tool cost, wall-clock p95, and failure severity. Report quality versus cost curves rather than a single cherry-picked sample. For stochastic strategies, run enough seeds to estimate uncertainty. A variant should be enabled only for task slices where incremental quality exceeds its latency and budget cost. Add a stopping rule based on verified completion or a hard budget, not a subjective sense that the model has reasoned sufficiently.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Single pass | Lowest cost | No recovery from first error |
| Self-consistency | Reduces some sampling variance | Correlated errors and voting cost |
| Tree search | Explores alternatives | Needs a useful evaluator |
| Program/tool verification | Strong for decidable properties | Limited to verifiable claims |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def keep_candidates(candidates, budget):
    return sorted(candidates, key=lambda x: x['value'] / max(x['cost'], 1), reverse=True)[:budget]

assert keep_candidates([{'value': 4, 'cost': 2}, {'value': 3, 'cost': 1}], 1)[0]['value'] == 3
```

## Failure Modes
- More samples repeat one bias: compare disagreement and use independent evidence sources.
- Search score rewards fluent but wrong branches: validate the scorer against known outcomes.
- Reasoning budget exhausts latency: cap branches and use a verifier-triggered escalation policy.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| When does tree search help? | When branching choices matter and a reasonably aligned evaluator can rank partial paths. Design follow-up: how will you calibrate that evaluator? |
| Can self-critique establish correctness? | No; treat it as another proposal unless grounded in external evidence or a checker. Debug follow-up: what evidence is independent? |
| How do you choose a reasoning budget? | Measure the marginal success improvement per token and latency on each task slice, then set an explicit cap. |

## Related Concepts
See [Tree of thought](09-tree-of-thought.md), [MCTS for agents](10-mcts-for-agents.md), [agent evaluation metrics](59-agent-evaluation-metrics.md).

## References
- [arxiv.org](https://arxiv.org/abs/2305.10601)
- [arxiv.org](https://arxiv.org/abs/2203.11171)

## Further Topic Detail
## Detailed Explanation

Advanced reasoning variants extend basic agent frameworks with enhanced decision-making capabilities: Chain-of-Thought (showing reasoning steps), Tree-of-Thought (exploring multiple reasoning paths), Self-Critique (agent reviews and improves its own output), Metacognition (agent thinking about its own thinking), and Debate (multiple agents arguing to reach better conclusions). These variants address the fact that the initial agent output is often suboptimal—iteration and alternative perspectives improve results.

Each variant trades computation for quality: Chain-of-Thought requires longer outputs but improves reasoning clarity. Tree-of-Thought explores multiple paths (expensive) but finds better solutions. Self-Critique adds a review pass. Debate uses multiple agents (N× computation). The key insight is that just like humans solve complex problems by thinking out loud, trying multiple approaches, and reconsidering, AI agents benefit from similar reflection. However, these variants multiply inference cost, creating a trade-off between solution quality and computational expense.

Advanced reasoning variants are becoming standard in complex agent applications because they demonstrably improve outcomes. Understanding them requires appreciating that single-pass inference often produces suboptimal results, and that cost of thinking is worth paying for important decisions.


## Core Intuition

A math student solving a hard problem doesn't just write the first answer—they work through multiple approaches, check their work, reconsider. Advanced reasoning variants let agents work similarly: try multiple solutions, think through their reasoning, critique and improve. This costs more time/compute but gives better answers.


## How It Works

1. Chain-of-thought: step-by-step reasoning improves accuracy
2. Tree-of-thought: generate multiple reasoning paths, evaluate each, select best
3. Self-consistency: sample multiple reasoning chains, take majority answer
4. Program synthesis: generate code, execute to verify correctness
5. Ensemble: combine multiple agent instances or reasoning strategies
6. Iterative refinement: generate answer → check → refine → repeat
7. Backtracking: if path fails, try alternative reasoning path




## Interview Q&A


**Q: How does tree-of-thought improve on standard chain-of-thought?**
A: CoT: single path (might get stuck). ToT: generate multiple paths (3-5), evaluate each, select best. Enables backtracking (if path fails, try other). Slower (need multiple inference passes) but more reliable for hard problems.

**Q: What is self-consistency and when is it useful?**
A: Self-consistency: sample same prompt multiple times (different random seeds), get multiple answers. Vote (majority wins). Improves accuracy especially for reasoning tasks (math, logic). Cost: N×inference cost for sampling N times. Worth it for critical decisions.

**Q: How do you evaluate reasoning quality in agents?**
A: Metrics: (1) final answer correctness, (2) reasoning quality (check steps are logical), (3) confidence (agent's self-assessment), (4) efficiency (how many steps to reach answer). Trace reasoning: log each step for debugging and auditing.

**Q: What is program synthesis and why is it better than natural language reasoning?**
A: Program synthesis: generate executable code instead of text. Benefits: deterministic (no ambiguity), verifiable (run and check), composable (combine programs). Challenge: requires coding capability, works best for specific problem classes.

**Q: How do you handle multiple agents reasoning differently?**
A: Ensemble: run multiple agents with different prompts/models. Aggregate: vote, weighted average, or consensus. Diversity: different agents should use different approaches (some systematic, some intuitive). Better performance than single agent, especially for complex problems.
