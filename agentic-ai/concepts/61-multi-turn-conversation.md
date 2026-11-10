# Multi-Turn Conversation Management
**TL;DR:** Conversation management preserves useful state across turns while controlling context growth, contradictions, privacy, and topic changes.

## Prerequisites
Agent memory, context windows, and structured data validation.

## Learning Outcomes
- Separate transcript history from durable user-approved state.
- Choose truncation, summarization, and retrieval by information risk.
- Test cross-turn consistency and topic-switch behavior.

## Core Intuition
A meeting recorder captures everything, while meeting minutes preserve decisions and open questions. A conversation agent needs both concepts: the transcript can be searched when needed, while a compact state record carries only verified facts and commitments. A summary is a lossy cache, so it needs provenance and correction.

## How It Works
The operating flow identifies the task state, selects a bounded strategy, gathers only the evidence needed for that decision, validates the result, and records the outcome. State should be updated transactionally with the response or action that established it. If a tool call fails after the conversation summary has recorded success, future turns inherit a false premise. Store a status such as proposed, confirmed, or failed and update it only after the authoritative event arrives. Keep provenance to the turn or tool result, expire transient facts, and make user corrections override earlier inferred state. These rules reduce contradictions more effectively than simply expanding the context window.
Maintain a structured conversation state with fields such as active task, confirmed preferences, unresolved questions, and commitments, each linked to the turn or source that established it. Keep recent turns verbatim for local reference; retrieve older turns only when relevant. Summaries should distinguish user claims from verified system facts and preserve uncertainty. On a topic switch, checkpoint unfinished work and ask whether it should be resumed when ambiguity matters. Allow users to inspect, correct, and delete durable memory.
State should be updated transactionally with the response or action that established it. If a tool call fails after the conversation summary has recorded success, future turns inherit a false premise. Store a status such as proposed, confirmed, or failed and update it only after the authoritative event arrives. Keep provenance to the turn or tool result, expire transient facts, and make user corrections override earlier inferred state. These rules reduce contradictions more effectively than simply expanding the context window.
Evaluate with multi-turn scenarios that include corrections, pronoun references, interruptions, topic switches, and long gaps. Measure factual carryover, stale-memory use, contradiction rate, token use, and the rate at which the agent asks for needed clarification. Include adversarial checks where a later turn contradicts an earlier preference. Compare the resulting state with an annotated state record rather than judging only response fluency. Track state changes with versions to debug unexpected carryover.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Full transcript | High recall and audit trail | Context cost and privacy exposure |
| Sliding recent window | Predictable cost | Forgets distant commitments |
| Rolling summary | Compact continuity | Can lose caveats or invert facts |
| Indexed retrieval | Selective recall | Retrieval misses and stale entries |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def update_topic(state, new_topic):
    return {'active_topic': new_topic, 'pending': state.get('pending', [])}

assert update_topic({'pending': ['draft']}, 'billing')['pending'] == ['draft']
```

## Failure Modes
- Summary converts a tentative claim into fact: preserve source and confidence fields and test corrections.
- Old topic leaks into a new task: checkpoint state and use relevance-gated retrieval.
- Sensitive details persist unexpectedly: define retention, deletion, and user inspection controls.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| What should persist between turns? | Only task-relevant, properly sourced state with clear retention rules; the transcript can remain separately searchable. Design follow-up: how are corrections propagated? |
| How do you manage context limits? | Keep a recent window, retrieve relevant older turns, and summarize with provenance while measuring loss. Debug follow-up: which test exposes omission? |
| How do you detect contradiction? | Compare new claims against sourced state and distinguish changed user intent from inconsistency; ask when unresolved. |

## Related Concepts
See [Memory types](11-memory-types.md), [agent memory management](12-agent-memory-management.md), [context management](19-context-window-management.md).

## References
- [arxiv.org](https://arxiv.org/abs/2304.03442)
- [platform.openai.com](https://platform.openai.com/docs/guides/conversation-state)

## Further Topic Detail
## Detailed Explanation

Multi-turn conversations involve back-and-forth exchanges where context from previous turns influences interpretation of current turns. This is natural for humans but challenging for AI systems: maintaining conversation coherence, tracking what's been established, handling contradictions, and managing growing context. Multi-turn interactions reveal weaknesses invisible in single-turn systems: inconsistency (contradicting earlier statements), context loss (forgetting discussed facts), and repetition (answering the same question multiple times).

Challenges include: (1) Context window limitations (can't keep entire conversation history as it grows), (2) Context relevance (not all previous turns matter equally), (3) Inconsistency (model might give conflicting responses in different turns), (4) User expectations (users expect agents to remember everything discussed). Solutions involve: context summarization (condensing older turns), selective retrieval (finding relevant past turns), coherence monitoring (detecting inconsistencies), and explicit state tracking (maintaining facts established during conversation).

Multi-turn conversation is central to practical agent applications—almost no real interaction is single-turn. Understanding it requires recognizing that conversation is collaborative meaning-making, not isolated exchanges. It's harder than single-turn because context compounds the complexity.


## Core Intuition

A phone customer support representative in their first call can look up everything. But a customer service agent working with a returning customer must remember they discussed this issue last week, their account details, their preferences. Multi-turn conversation is managing this growing, interconnected context.


## How It Works

1. Turns: user input → agent response → user input → ...
2. Context: keep relevant history (last N turns, summarized context)
3. State: track conversation state (topic, subtasks, context)
4. Coherence: ensure responses consistent with context and previous statements
5. Turn-taking: manage who speaks (user or agent, avoid deadlock)
6. Interruption: handle user interruptions, task switching
7. Cleanup: end conversation gracefully, summarize outcome




## Interview Q&A


**Q: How much conversation history should you keep?**
A: Full history: accurate but uses tokens. Last N turns: balance accuracy and efficiency. Summarization: compress old turns to facts. Hybrid: keep last 5 full turns, summarize older context. Adjust based on context window size and task complexity.

**Q: How do you handle context window overflows?**
A: Overflow: conversation exceeds model's context window. Solutions: (1) drop oldest messages, (2) summarize old context, (3) retrieval (fetch relevant messages from history). Try summarization first (preserve info), then drop if needed.

**Q: What is conversation state and how do you track it?**
A: State: current topic, subtasks completed, user intent, decisions made. Track: explicitly (state variable) or implicitly (inferred from messages). Use explicit for critical applications (fewer errors). Update: after each agent response.

**Q: How do you prevent agents from contradicting themselves?**
A: Check: before responding, review past statements. Verify: ensure new statement consistent with context. Fallback: if inconsistency detected, acknowledge and clarify. Log: track contradictions for debugging.

**Q: How do you handle topic switching in conversation?**
A: Detect: user changes topic (detect intent shift). Handle: (1) acknowledge old topic, (2) switch context, (3) reset sub-state if needed. Challenge: intent detection not always clear (genuine switch vs. tangent).
