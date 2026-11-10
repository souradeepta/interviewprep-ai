# ReAct: Reasoning + Acting
**TL;DR:** A ReAct loop alternates a bounded reasoning decision, a tool action, and an observation so later actions respond to new evidence.

## Prerequisites
Agent loops, tool boundaries, and evaluation.

## Learning Outcomes
- Explain the main react workflow and its state boundary.
- Compare an automated path with review or fallback using measured outcomes.
- Diagnose a representative failure using trace evidence.

## Core Intuition
ReAct: Reasoning + Acting is best understood as a controlled workflow with explicit evidence and a bounded decision. The system should expose what it knows, what it is allowed to do, and when a person or deterministic checker must take over. This framing prevents a fluent answer from being confused with a verified outcome.

## How It Works
For partially observable work, each observation can change the next action; this avoids committing to a brittle long plan. The controller should keep actions typed and validate tool results before placing them back into the model context.
## Detailed Explanation

ReAct (Reasoning + Acting) interleaves reasoning steps with action execution. Instead of planning everything upfront (planning agents) or reacting blindly (reactive agents), ReAct agents think-act-observe-think-act cycles. Example: "I need to book a flight [Reason]. Let me search available flights [Act]. I found 3 options [Observe]. Option 2 is cheapest [Reason]. Let me check user budget [Act]. Budget is sufficient [Observe]. Book flight 2 [Reason+Act]." ReAct balances planning and reactivity: agents plan locally (next 1-2 steps) but adapt to observations immediately. This is more flexible than rigid plans and more intelligent than pure reaction. ReAct excels when: (1) environment is partially observable (need feedback), (2) outcomes are uncertain (observation reveals reality), (3) tasks require reasoning mid-execution. Key advantage: agents can correct mistakes immediately. Key disadvantage: requires frequent reasoning cycles, more LLM calls. Implementation: structure prompts as "Thought: X, Action: Y, Observation: Z" loops, letting LLM generate reasoning and actions simultaneously. The observation (tool result) becomes context for next reasoning step.


## Core Intuition

Imagine learning to play chess. Beginner: "I'll plan 5 moves ahead... oh, that leads to checkmate against me." Better: "I'll think 1-2 moves ahead, move, see opponent's response, reconsider." ReAct is the second approach: think a bit, act, observe, adjust, repeat. More robust than full planning because you adapt to surprises.


## How It Works

ReAct cycles through Think-Act-Observe:

1. **Thought** — LLM reasons about current state and next step
2. **Action** — LLM decides which tool to call with what params
3. **Observation** — Tool returns result
4. **Repeat** — Use observation as context for next thought




## Interview Q&A

**Q: ReAct vs planning agents—difference?**
A: Planning: "Plan everything upfront, execute." ReAct: "Plan 1 step, execute, observe, plan next step." ReAct adapts to surprises; planning is rigid. ReAct uses more LLM calls but is more flexible.

**Q: How many reasoning steps before acting?**
A: Depends on problem. Math: multiple reasoning steps before calculation. Search: reason about what to search for, then act. Rule of thumb: 1-2 reasoning steps per action is good balance.

**Q: How to prevent infinite loops?**
A: Set max steps (e.g., 10 ReAct cycles). Detect repeated states (same action repeated). If stuck, use fallback or escalate to user.

**Q: Cost of ReAct vs simpler approaches?**
A: ReAct costs more (multiple LLM calls per task). Worth it for complex reasoning, not for simple queries. Use simpler approaches for straightforward tasks.

**Q: How do you detect when agent is stuck?**
A: Repeated actions (same action twice in a row), same observation twice (action had no effect), max step limit reached. When stuck, escalate or try fallback approach.
Compare tool success, turns per task, and completion quality against a fixed workflow. ReAct is useful when the next step depends on a tool result; a deterministic sequence is cheaper when the workflow is already known.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Deterministic baseline | Predictable behavior and low cost | Limited coverage of ambiguous cases |
| Model-assisted path | Handles flexible inputs | Requires evidence checks and evaluation |
| Human review | Resolves high-impact uncertainty | Capacity and response-time limits |
| Fallback or abstention | Bounds unsafe behavior | May defer useful work |

## Implementation
This runnable Python example shows a bounded policy gate relevant to the workflow. It models control logic only; it does not claim production quality.

```python
def react_step(observation, choose, tools):
    action = choose(observation)
    result = tools[action['name']](**action['args'])
    return {'action': action, 'observation': result}

assert react_step({'x': 1}, lambda _: {'name': 'lookup', 'args': {'key': 'x'}},
                   {'lookup': lambda key: key})['observation'] == 'x'
```

## Failure Modes
- ReAct: Reasoning + Acting uses stale or incomplete input: log source and timestamp, validate freshness, and request a refresh before acting.
- ReAct: Reasoning + Acting produces an unsupported result: require evidence and a domain-specific validator; route unresolved cases to review.
- ReAct: Reasoning + Acting exceeds its resource or authority boundary: enforce explicit budgets and permissions in the controller, then report the stopped state.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| ReAct vs planning agents—difference? | Planning: "Plan everything upfront, execute." ReAct: "Plan 1 step, execute, observe, plan next step." ReAct adapts to surprises; planning is rigid. ReAct uses more LLM calls but is more flexible. |
| How many reasoning steps before acting? | Depends on problem. Math: multiple reasoning steps before calculation. Search: reason about what to search for, then act. Rule of thumb: 1-2 reasoning steps per action is good balance. |
| How to prevent infinite loops? | Set max steps (e.g., 10 ReAct cycles). Detect repeated states (same action repeated). If stuck, use fallback or escalate to user. |

## Related Concepts
See [agent testing](28-agent-testing.md), [agent evaluation](29-agent-evals.md), and [agent monitoring](30-agent-monitoring.md).

## References
- [arxiv.org](https://arxiv.org/abs/2210.03629)
- [arxiv.org](https://arxiv.org/abs/2305.10601)

## Further Topic Detail
## Detailed Explanation

ReAct (Reasoning + Acting) interleaves reasoning steps with action execution. Instead of planning everything upfront (planning agents) or reacting blindly (reactive agents), ReAct agents think-act-observe-think-act cycles. Example: "I need to book a flight [Reason]. Let me search available flights [Act]. I found 3 options [Observe]. Option 2 is cheapest [Reason]. Let me check user budget [Act]. Budget is sufficient [Observe]. Book flight 2 [Reason+Act]." ReAct balances planning and reactivity: agents plan locally (next 1-2 steps) but adapt to observations immediately. This is more flexible than rigid plans and more intelligent than pure reaction. ReAct excels when: (1) environment is partially observable (need feedback), (2) outcomes are uncertain (observation reveals reality), (3) tasks require reasoning mid-execution. Key advantage: agents can correct mistakes immediately. Key disadvantage: requires frequent reasoning cycles, more LLM calls. Implementation: structure prompts as "Thought: X, Action: Y, Observation: Z" loops, letting LLM generate reasoning and actions simultaneously. The observation (tool result) becomes context for next reasoning step.


## Core Intuition

Imagine learning to play chess. Beginner: "I'll plan 5 moves ahead... oh, that leads to checkmate against me." Better: "I'll think 1-2 moves ahead, move, see opponent's response, reconsider." ReAct is the second approach: think a bit, act, observe, adjust, repeat. More robust than full planning because you adapt to surprises.


## How It Works

ReAct cycles through Think-Act-Observe:

1. **Thought** — LLM reasons about current state and next step
2. **Action** — LLM decides which tool to call with what params
3. **Observation** — Tool returns result
4. **Repeat** — Use observation as context for next thought




## Interview Q&A

**Q: ReAct vs planning agents—difference?**
A: Planning: "Plan everything upfront, execute." ReAct: "Plan 1 step, execute, observe, plan next step." ReAct adapts to surprises; planning is rigid. ReAct uses more LLM calls but is more flexible.

**Q: How many reasoning steps before acting?**
A: Depends on problem. Math: multiple reasoning steps before calculation. Search: reason about what to search for, then act. Rule of thumb: 1-2 reasoning steps per action is good balance.

**Q: How to prevent infinite loops?**
A: Set max steps (e.g., 10 ReAct cycles). Detect repeated states (same action repeated). If stuck, use fallback or escalate to user.

**Q: Cost of ReAct vs simpler approaches?**
A: ReAct costs more (multiple LLM calls per task). Worth it for complex reasoning, not for simple queries. Use simpler approaches for straightforward tasks.

**Q: How do you detect when agent is stuck?**
A: Repeated actions (same action twice in a row), same observation twice (action had no effect), max step limit reached. When stuck, escalate or try fallback approach.
