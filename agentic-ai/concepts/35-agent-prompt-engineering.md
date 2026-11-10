# Agent Prompt Engineering
**TL;DR:** Agent prompt engineering specifies tool choice, input/output contracts, stopping behavior, and uncertainty handling for a reusable agent policy.

## Prerequisites
Agent loops, tool boundaries, and evaluation.

## Learning Outcomes
- Explain the main prompt contract workflow and its state boundary.
- Compare an automated path with review or fallback using measured outcomes.
- Diagnose a representative failure using trace evidence.

## Core Intuition
Agent Prompt Engineering is best understood as a controlled workflow with explicit evidence and a bounded decision. The system should expose what it knows, what it is allowed to do, and when a person or deterministic checker must take over. This framing prevents a fluent answer from being confused with a verified outcome.

## How It Works
Treat prompts as versioned application configuration. Keep task policy separate from retrieved untrusted data, constrain tools with schemas and authorization, and put non-negotiable safety checks in code. Examples should cover both a successful path and a refusal or tool failure. Prompt wording alone cannot enforce permissions.
## Detailed Explanation

Prompt engineering for agents designs prompts that guide agent behavior. Techniques: role definition ("helpful assistant"), context (background info), instruction (explicit steps), examples (few-shot learning), constraints ("only use these tools"), feedback (guide toward correct behavior). Difference from user prompt engineering: agent prompts are long-lived (reused across many queries), stable (changes impact all uses), and behavior-optimized. Challenges: brittleness (small changes cause big behavioral shifts), testing (hard to measure prompt quality), latency (longer prompts = slower). Best for: steering agent toward specific behaviors, constraining action space, teaching via examples.


## Core Intuition

Job description + playbook for your team. Clear instructions = better work. Vague instructions = disappointing results.


## How It Works

System prompt + Task instructions + Context + Examples + Constraints:

1. **System Prompt** — Define role and baseline behavior
2. **Task Instruction** — What to do
3. **Context** — Relevant information
4. **Examples** — Few-shot good examples
5. **Constraints** — What agent cannot do
6. **Evaluation** — How to measure success


## Interview Q&A

**Q: How do agent system prompts differ from standard LLM system prompts?**
A: Agent system prompts must define tool use (what tools exist, when to use them, what output format is expected), multi-step planning behavior (how to break down tasks), error handling (what to do when a tool fails or returns unexpected output), stopping criteria (how to know when the task is complete), and handoff conditions (when to ask for clarification vs. proceed). Standard system prompts focus on persona and style; agent system prompts are more like a behavioral specification.

**Q: How do you prevent prompt injection attacks in multi-agent systems?**
A: Prompt injection: malicious content in tool outputs or external data attempting to override agent instructions. Defense in depth: (1) explicitly instruct the agent to treat all tool outputs as data, not instructions; (2) use structured schemas for tool outputs (JSON) rather than free text; (3) implement a "safety check" agent that scans tool outputs for instruction-like patterns; (4) log all tool outputs and flag suspicious patterns; (5) sandbox the agent's tool execution environment so even successful injection has limited blast radius.

**Q: How do you design agent prompts for tool selection when there are many tools available?**
A: With many tools (10+), the agent struggles to select the right one and may over-use a few familiar tools. Design prompts with: tool categorization (group tools by function), usage guidelines (when to use each), examples of multi-tool workflows, and explicit instructions on tool selection criteria. Consider retrieval-augmented tool selection: embed tool descriptions and retrieve the most relevant tools for the current task, rather than including all tools in every prompt. Test tool selection accuracy: given a task description, does the agent select the appropriate tools?

**Q: What is the role of few-shot examples in agent prompts and how many do you need?**
A: Few-shot examples in agent prompts demonstrate: the correct format for tool calls, multi-step reasoning patterns, how to handle tool errors, and how to combine tool outputs. Unlike standard few-shot prompting, agent examples must show the full interaction trace (reasoning + tool calls + results + next step). 1-3 examples covering the most common workflow patterns are typically sufficient—more examples consume context window and may confuse the agent with rare patterns it applies inappropriately.

**Q: How do you iteratively improve an agent prompt based on failure analysis?**
A: Log all agent trajectories (full sequence of reasoning + tool calls). Categorize failures: wrong tool selected, incorrect parameter format, wrong stopping condition, reasoning error. For each category, create a fix: add explicit instruction, add a few-shot example of the failure mode handled correctly, or restructure the task decomposition. Test each fix on a held-out set of the failure type before adding to production. Track failure rates by category over versions—a fix that reduces one failure may increase another.

**Q: How do you prompt an agent to know when it has completed a task?**
A: Define explicit completion criteria in the prompt: what constitutes a successful result (specific format, specific content requirements), and what conditions should trigger giving up (max steps reached, tool returns unavailable, ambiguous requirements). Add completion self-check: before concluding, the agent should verify its output against the task requirements. Implement a supervisor pattern: a separate prompt reviews the agent's conclusion and confirms it meets the task spec or requests additional work. Incomplete tasks that report as complete are a critical failure mode.
Use a held-out task set to measure tool selection, schema validity, completion, refusal behavior, token use, and regressions after each edit. Change one policy element at a time so failures can be attributed to the prompt or runtime.

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
def build_prompt(policy, user_task, examples):
    return {'policy_version': policy, 'task': user_task, 'examples': list(examples)}

prompt = build_prompt('v3', 'summarize report', [])
assert prompt['policy_version'] == 'v3' and prompt['task']
```

## Failure Modes
- Agent Prompt Engineering uses stale or incomplete input: log source and timestamp, validate freshness, and request a refresh before acting.
- Agent Prompt Engineering produces an unsupported result: require evidence and a domain-specific validator; route unresolved cases to review.
- Agent Prompt Engineering exceeds its resource or authority boundary: enforce explicit budgets and permissions in the controller, then report the stopped state.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| How do agent system prompts differ from standard LLM system prompts? | Agent system prompts must define tool use (what tools exist, when to use them, what output format is expected), multi-step planning behavior (how to break down tasks), error handling (what to do when a tool fails or returns unexpected output), stopping criteria (how to know when the task is complete), and handoff conditions (when to ask for clarification vs. proceed). Standard system prompts focus on persona and style; agent system prompts are more like a behavioral specification. |
| How do you prevent prompt injection attacks in multi-agent systems? | Prompt injection: malicious content in tool outputs or external data attempting to override agent instructions. Defense in depth: (1) explicitly instruct the agent to treat all tool outputs as data, not instructions; (2) use structured schemas for tool outputs (JSON) rather than free text; (3) implement a "safety check" agent that scans tool outputs for instruction-like patterns; (4) log all tool outputs and flag suspicious patterns; (5) sandbox the agent's tool execution environment so even successful injection has limited blast radius. |
| How do you design agent prompts for tool selection when there are many tools available? | With many tools (10+), the agent struggles to select the right one and may over-use a few familiar tools. Design prompts with: tool categorization (group tools by function), usage guidelines (when to use each), examples of multi-tool workflows, and explicit instructions on tool selection criteria. Consider retrieval-augmented tool selection: embed tool descriptions and retrieve the most relevant tools for the current task, rather than including all tools in every prompt. Test tool selection accuracy: given a task description, does the agent select the appropriate tools? |

## Related Concepts
See [agent testing](28-agent-testing.md), [agent evaluation](29-agent-evals.md), and [agent monitoring](30-agent-monitoring.md).

## References
- [www.nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)
- [opentelemetry.io](https://opentelemetry.io/docs/)

## Further Topic Detail
## Detailed Explanation

Prompt engineering for agents designs prompts that guide agent behavior. Techniques: role definition ("helpful assistant"), context (background info), instruction (explicit steps), examples (few-shot learning), constraints ("only use these tools"), feedback (guide toward correct behavior). Difference from user prompt engineering: agent prompts are long-lived (reused across many queries), stable (changes impact all uses), and behavior-optimized. Challenges: brittleness (small changes cause big behavioral shifts), testing (hard to measure prompt quality), latency (longer prompts = slower). Best for: steering agent toward specific behaviors, constraining action space, teaching via examples.


## Core Intuition

Job description + playbook for your team. Clear instructions = better work. Vague instructions = disappointing results.


## How It Works

System prompt + Task instructions + Context + Examples + Constraints:

1. **System Prompt** — Define role and baseline behavior
2. **Task Instruction** — What to do
3. **Context** — Relevant information
4. **Examples** — Few-shot good examples
5. **Constraints** — What agent cannot do
6. **Evaluation** — How to measure success


## Interview Q&A

**Q: How do agent system prompts differ from standard LLM system prompts?**
A: Agent system prompts must define tool use (what tools exist, when to use them, what output format is expected), multi-step planning behavior (how to break down tasks), error handling (what to do when a tool fails or returns unexpected output), stopping criteria (how to know when the task is complete), and handoff conditions (when to ask for clarification vs. proceed). Standard system prompts focus on persona and style; agent system prompts are more like a behavioral specification.

**Q: How do you prevent prompt injection attacks in multi-agent systems?**
A: Prompt injection: malicious content in tool outputs or external data attempting to override agent instructions. Defense in depth: (1) explicitly instruct the agent to treat all tool outputs as data, not instructions; (2) use structured schemas for tool outputs (JSON) rather than free text; (3) implement a "safety check" agent that scans tool outputs for instruction-like patterns; (4) log all tool outputs and flag suspicious patterns; (5) sandbox the agent's tool execution environment so even successful injection has limited blast radius.

**Q: How do you design agent prompts for tool selection when there are many tools available?**
A: With many tools (10+), the agent struggles to select the right one and may over-use a few familiar tools. Design prompts with: tool categorization (group tools by function), usage guidelines (when to use each), examples of multi-tool workflows, and explicit instructions on tool selection criteria. Consider retrieval-augmented tool selection: embed tool descriptions and retrieve the most relevant tools for the current task, rather than including all tools in every prompt. Test tool selection accuracy: given a task description, does the agent select the appropriate tools?

**Q: What is the role of few-shot examples in agent prompts and how many do you need?**
A: Few-shot examples in agent prompts demonstrate: the correct format for tool calls, multi-step reasoning patterns, how to handle tool errors, and how to combine tool outputs. Unlike standard few-shot prompting, agent examples must show the full interaction trace (reasoning + tool calls + results + next step). 1-3 examples covering the most common workflow patterns are typically sufficient—more examples consume context window and may confuse the agent with rare patterns it applies inappropriately.

**Q: How do you iteratively improve an agent prompt based on failure analysis?**
A: Log all agent trajectories (full sequence of reasoning + tool calls). Categorize failures: wrong tool selected, incorrect parameter format, wrong stopping condition, reasoning error. For each category, create a fix: add explicit instruction, add a few-shot example of the failure mode handled correctly, or restructure the task decomposition. Test each fix on a held-out set of the failure type before adding to production. Track failure rates by category over versions—a fix that reduces one failure may increase another.

**Q: How do you prompt an agent to know when it has completed a task?**
A: Define explicit completion criteria in the prompt: what constitutes a successful result (specific format, specific content requirements), and what conditions should trigger giving up (max steps reached, tool returns unavailable, ambiguous requirements). Add completion self-check: before concluding, the agent should verify its output against the task requirements. Implement a supervisor pattern: a separate prompt reviews the agent's conclusion and confirms it meets the task spec or requests additional work. Incomplete tasks that report as complete are a critical failure mode.
