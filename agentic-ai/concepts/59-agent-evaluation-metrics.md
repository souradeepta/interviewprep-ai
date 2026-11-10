# Agent Evaluation Metrics
**TL;DR:** Agent evaluation measures goal completion, policy compliance, reliability, and resource use over whole trajectories rather than text similarity alone.

## Prerequisites
Classification metrics, experimental design, and agent traces.

## Learning Outcomes
- Define task success with observable criteria.
- Report trajectory-level safety and efficiency measures.
- Build a representative evaluation set with adjudicated labels.

## Core Intuition
A car is not evaluated only by how polished its dashboard looks; its route, safety, reliability, and fuel use matter. An agent response is likewise one artifact inside a trajectory that may include tool choices, retries, and side effects. Scoring only the final text misses whether the task was actually completed safely.

## How It Works
The operating flow identifies the task state, selects a bounded strategy, gathers only the evidence needed for that decision, validates the result, and records the outcome. Build the evaluation corpus from real task shapes and include the full tool environment contract. A benchmark that omits authentication failures or empty search results will overstate production reliability. Label success criteria before inspecting outputs, and have reviewers adjudicate ambiguous cases with documented rubrics. Preserve raw traces under access controls so failures can be reproduced; remove secrets and minimize sensitive data before wider analysis. Re-evaluate after prompt, model, tool, or policy changes because each can alter the trajectory.
Write task contracts before selecting a metric: expected final state, allowed tools, forbidden actions, required evidence, and acceptable partial completion. A trajectory evaluator can combine deterministic checks with human judgments. Deterministic checks should handle facts such as schema validity, expected database state, and whether an unauthorized tool was called. Human review is appropriate for tone or ambiguous quality, with blinded rubrics and adjudication for disagreement. Keep train/development/test task sets separated by scenario family to avoid prompt tuning on benchmark duplicates.
Build the evaluation corpus from real task shapes and include the full tool environment contract. A benchmark that omits authentication failures or empty search results will overstate production reliability. Label success criteria before inspecting outputs, and have reviewers adjudicate ambiguous cases with documented rubrics. Preserve raw traces under access controls so failures can be reproduced; remove secrets and minimize sensitive data before wider analysis. Re-evaluate after prompt, model, tool, or policy changes because each can alter the trajectory.
Report task success with a confidence interval, severe policy violations, retry rate, tool-call success, cost per successful task, and latency percentiles. Also test robustness across paraphrases, missing fields, tool outages, and distribution slices. Avoid a single weighted score unless weights reflect an explicit product decision: a high completion score must not hide a severe safety failure. Human preference scores need rater agreement and calibration checks. Run regressions by version and preserve failed traces for replay.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Exact state assertion | Repeatable and cheap | Only covers known outcomes |
| Model judge | Scales nuanced review | Judge bias and prompt sensitivity |
| Human rubric | Rich context and nuance | Cost, delay, and rater variance |
| Online experiment | Measures user outcomes | Exposure risk and confounding |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def score_task(success, violation, tokens, token_budget):
    return {'success': bool(success), 'policy_ok': not violation, 'within_budget': tokens <= token_budget}

assert score_task(True, False, 20, 50)['policy_ok']
```

## Failure Modes
- Benchmark leakage inflates scores: deduplicate by scenario and hold out task families.
- Judge accepts plausible but false output: anchor scoring in evidence and adjudicate samples.
- Average success hides rare severe violations: report severity-stratified rates and hard safety gates.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| How do you evaluate an open-ended agent? | Use a rubric tied to task criteria plus deterministic checks for actions and evidence, then measure rater agreement. Design follow-up: what is held out? |
| Why cost per success instead of cost per call? | Agents make variable numbers of calls; unsuccessful loops still consume budget. Debug follow-up: how do you attribute retries? |
| How do you avoid benchmark overfitting? | Hold out scenario families, include perturbations and production samples, and freeze a regression set. |

## Related Concepts
See [Agent evals](29-agent-evals.md), [agent testing](28-agent-testing.md), [agent monitoring](30-agent-monitoring.md).

## References
- [platform.openai.com](https://platform.openai.com/docs/guides/evals)
- [arxiv.org](https://arxiv.org/abs/2309.07864)

## Further Topic Detail
## Detailed Explanation

Evaluating agents is fundamentally different from evaluating models because agents are judged not just on output quality but on whether they achieve goals reliably, efficiently, and safely. Standard metrics (BLEU, ROUGE) measure text quality but don't capture whether an agent successfully completed tasks. Agent evaluation requires: goal achievement (did the agent accomplish the task?), efficiency (how many steps? how much cost?), safety (did it avoid harmful actions?), and robustness (does it work on varied inputs?).

Evaluation approaches include: (1) Task completion rates (what fraction of tasks succeed), (2) Step efficiency (how many agent steps to complete tasks), (3) Token efficiency (how many tokens consumed), (4) Cost metrics (API cost, human review cost), (5) Human evaluation (did humans rate outputs as helpful?), (6) Benchmark datasets (standardized tasks). Challenges include: benchmarks may not reflect real distributions (agents overfit to benchmark patterns), human evaluation is expensive, and metrics can conflict (fast often means less accurate). Developing robust agent evaluation is active research because it's crucial for building trustworthy systems.

Understanding agent evaluation requires systems thinking and appreciation for complex trade-offs. You can't optimize one metric (speed) at the expense of others (accuracy, safety) without careful measurement.


## Core Intuition

Evaluating a student isn't just grading test answers—it's asking: do they solve real problems? Are they efficient? Do they think safely? Evaluating agents similarly requires measuring whether they achieve goals, how many steps they take, whether they waste resources, and whether they make dangerous decisions.


## How It Works

1. Task success: did agent complete task correctly? Binary or continuous score
2. Cost: token usage × cost-per-token, total expense per task
3. Latency: wall-clock time to complete task, end-to-end latency
4. Quality: subjective evaluation (human raters), automated metrics
5. Efficiency: task success / (cost × latency), score per resource
6. Reliability: success rate, failure modes, error distribution
7. User satisfaction: NPS, task satisfaction, willingness to use again
8. Benchmarks: standard datasets, compare across agents and versions




## Interview Q&A


**Q: How do you measure success for open-ended tasks?**
A: Binary success: did agent achieve goal? Limited (doesn't measure partial progress). Continuous: score 0-1 based on closeness to goal. Rubric: define criteria (clarity, accuracy, completeness) and score on each. Human evaluation most reliable but expensive.

**Q: What metrics matter most for production agents?**
A: Prioritize: (1) success rate (core metric), (2) cost (budget constraint), (3) latency (user patience), (4) error rate (reliability). In that order: fast failure is costly, slow success is annoying, errors are worst. Monitor all four.

**Q: How do you compare agents fairly?**
A: Same benchmark: test on identical tasks. Controlled conditions: same model, same prompts, same data. Multiple seeds: run each agent 5-10 times, report mean ± std. Report all metrics (not cherry-picked). Significance testing (statistical).

**Q: What is Pareto frontier for agents?**
A: Trade-off between metrics (accuracy vs cost, accuracy vs speed). Pareto frontier: set of non-dominated solutions (can't improve one without worsening another). Plot accuracy vs cost for all agents, frontier shows Pareto-optimal agents.

**Q: How do you handle metrics for multi-turn interactions?**
A: Per-turn: success per interaction. Cumulative: did agent eventually succeed? Task-centric: did agent achieve high-level goal (may take many turns)? Choose based on use case. For chat: success = user satisfied after conversation.
