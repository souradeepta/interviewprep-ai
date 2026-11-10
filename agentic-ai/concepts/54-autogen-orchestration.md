# AutoGen Orchestration
**TL;DR:** AutoGen coordinates conversational agents and tools; useful orchestration requires explicit turn limits, termination rules, and least-privilege tool boundaries.

## Prerequisites
Agent loops, tool authorization, and asynchronous programming.

## Learning Outcomes
- Represent group conversation as a bounded state machine.
- Control speaker selection, turn budget, and termination explicitly.
- Measure whether delegation improves success enough to justify extra calls.

## Core Intuition
A panel of specialists helps only when each person has a distinct task and the chair can end the meeting. An orchestrator that lets agents keep asking each other for opinions creates cost and delay without new evidence. Assign roles based on capabilities and evidence access, and let a deterministic controller stop the process.

## How It Works
Define participants with narrow responsibilities, available tools, and output contracts. A manager selects the next speaker or follows a fixed sequence, while each message records the task, evidence, and handoff reason. The termination policy checks a concrete completion condition and a maximum turn or token budget; “the agents agree” is not enough for a task that requires an external fact or verified action. Tool calls pass through a permission layer, and state changes are persisted outside the transient conversation so retries can resume safely.
Delegation is useful when work can be separated, such as one agent collecting evidence and another checking citations. It is less useful when agents share the same model, context, and blind spots. Independent verification should use a different evidence source or a deterministic check where possible. Avoid treating majority vote as proof: correlated agents can repeat the same error. A human approval gate belongs before irreversible external effects, and the approval request should show the proposed action and supporting evidence rather than a long unstructured transcript.
Conversation topology is a design choice with measurable consequences. A fixed researcher-then-reviewer sequence is easier to test than unrestricted peer chat, while a manager loop can adapt when evidence is missing. Use role separation only when agents have distinct instructions, tools, or evidence sources; renaming identical prompts does not create independent judgment. Persist artifacts and decisions between turns so a worker restart does not require replaying every message or repeating an external action.
Compare a single-agent baseline with the orchestrated design on the same tasks and model configuration. Measure end-to-end success, number of model turns, tool calls, duplicated evidence, wall-clock latency, token cost, and termination failures. An additional agent is worthwhile only if it improves an agreed outcome after accounting for its resource cost. Include adversarial tasks that cause disagreement, tool errors, empty outputs, and repeated messages. Trace speaker transitions to locate waste.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Single agent | Low latency and simpler state | May exceed one context or skill boundary |
| Fixed sequence | Predictable roles and cost | Cannot adapt order to evidence |
| Manager-selected turns | Flexible delegation | Manager errors and loop risk |
| Peer discussion | Can expose disagreement | High token cost and correlated errors |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def next_speaker(turns, limit, participants):
    if turns >= limit: return 'stop'
    return participants[turns % len(participants)]

assert next_speaker(4, 4, ['researcher', 'reviewer']) == 'stop'
```

## Failure Modes
- Conversation loops between two agents: enforce maximum turns and detect repeated state/message signatures.
- All agents repeat one unsupported claim: require citations or independent tool evidence and test source independence.
- Retry repeats an external action: persist action status and use idempotency keys behind the tool gateway.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| When do multiple agents justify their cost? | When the task decomposes into distinct evidence or validation work and measured success improves over a single-agent baseline. Design follow-up: what is your stopping condition? |
| How do you prevent infinite conversation? | Bound turns, tokens, wall time, and tool actions, then check an explicit completion predicate. Debug follow-up: which trace reveals the cycle? |
| How should agent disagreement be resolved? | Compare claims against evidence and policy; escalate unresolved high-impact cases rather than relying on majority vote. |

## Related Concepts
See [Multi-agent systems](14-multi-agent-systems.md), [agent communication](13-agent-communication.md), [error recovery](26-error-recovery.md).

## References
- [microsoft.github.io](https://microsoft.github.io/autogen/stable/)
- [arxiv.org](https://arxiv.org/abs/2308.08155)

## Further Topic Detail
## Core Intuition
A panel of specialists helps only when each person has a distinct task and the chair can end the meeting. An orchestrator that lets agents keep asking each other for opinions creates cost and delay without new evidence. Assign roles based on capabilities and evidence access, and let a deterministic controller stop the process.


## How It Works
Define participants with narrow responsibilities, available tools, and output contracts. A manager selects the next speaker or follows a fixed sequence, while each message records the task, evidence, and handoff reason. The termination policy checks a concrete completion condition and a maximum turn or token budget; “the agents agree” is not enough for a task that requires an external fact or verified action. Tool calls pass through a permission layer, and state changes are persisted outside the transient conversation so retries can resume safely.
Delegation is useful when work can be separated, such as one agent collecting evidence and another checking citations. It is less useful when agents share the same model, context, and blind spots. Independent verification should use a different evidence source or a deterministic check where possible. Avoid treating majority vote as proof: correlated agents can repeat the same error. A human approval gate belongs before irreversible external effects, and the approval request should show the proposed action and supporting evidence rather than a long unstructured transcript.
Conversation topology is a design choice with measurable consequences. A fixed researcher-then-reviewer sequence is easier to test than unrestricted peer chat, while a manager loop can adapt when evidence is missing. Use role separation only when agents have distinct instructions, tools, or evidence sources; renaming identical prompts does not create independent judgment. Persist artifacts and decisions between turns so a worker restart does not require replaying every message or repeating an external action.
Compare a single-agent baseline with the orchestrated design on the same tasks and model configuration. Measure end-to-end success, number of model turns, tool calls, duplicated evidence, wall-clock latency, token cost, and termination failures. An additional agent is worthwhile only if it improves an agreed outcome after accounting for its resource cost. Include adversarial tasks that cause disagreement, tool errors, empty outputs, and repeated messages. Trace speaker transitions to locate waste.


## Detailed Explanation

Coordinate multiple agents with different roles and capabilities to solve complex tasks through conversation



## Core Intuition

Coordinate multiple agents with different roles and capabilities to solve complex tasks through conversation



## How It Works

1. Define agents with specific roles (researcher, critic, executor)
2. Assign capabilities: tool access, knowledge base, reasoning
3. Agent interaction: send messages back-and-forth
   - Agent A → Agent B: question or result
   - Agent B → Agent A: response or next action
4. Termination: conversation ends when task solved or max rounds reached
5. Nested agents: agent calls another agent as a tool
6. Example: user → researcher (gathers info) → critic (evaluates) → executor (takes action)





## Interview Q&A


**Q: How do you define agent roles in AutoGen?**
A: System message defines role: 'You are a researcher. Your job is to...'. Role shapes how agent interprets requests and what tools it has. Clear roles reduce hallucination and wasted steps.

**Q: What's better: one smart agent or many specialized agents?**
A: Many agents: better division of labor, easier to debug (separate concerns), can verify work. One agent: simpler, fewer tokens (fewer messages), faster. Use many for complex tasks, one for simple.

**Q: How do you handle agent disagreement?**
A: Add a moderator agent that arbitrates. Or voting: each agent proposes solution, majority wins. Or escalation: complex cases go to human. Context matters: some tasks need consensus, others don't.

**Q: What is the cost of multi-agent systems vs single agent?**
A: Multi-agent: more API calls (message overhead), more tokens total. Single agent: fewer calls but may fail on complex tasks. Measure: token count × cost/token + latency tradeoff. Multi-agent often cheaper for complex tasks.

**Q: How do you prevent infinite loops in agent conversations?**
A: Set max_rounds (e.g., 20 turns max). Monitor: if same question repeated, stop. Termination condition: check if task complete (heuristic). Human-in-loop: user can stop anytime.
