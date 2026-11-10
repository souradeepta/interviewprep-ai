# LangChain Frameworks
**TL;DR:** LangChain supplies composable model, prompt, parser, retriever, and tool interfaces; application code still owns policy, state, reliability, and evaluation.

## Prerequisites
Python, model APIs, structured outputs, and tool-use basics.

## Learning Outcomes
- Choose a deterministic chain or an agent loop for a workflow.
- Validate model output at typed boundaries before side effects.
- Measure framework overhead and pin versions for reproducibility.

## Core Intuition
A framework is a set of adapters and orchestration parts, similar to a workshop bench with standard clamps. It speeds assembly when components fit the task, but the bench does not guarantee that the finished machine is safe. Framework abstractions can hide retries, prompts, and tool boundaries, so engineers must know what actually executes.

## How It Works
A reliable application makes each boundary explicit: input validation, prompt construction, model invocation, parsing, tool authorization, and final response. A fixed chain is appropriate when order is known; an agent executor is justified when choosing among tools is part of the task. Parsers validate shape but do not prove factual correctness. Retrievers return candidate evidence but do not ensure the model used it faithfully. Wrap each external call with timeout, bounded retry, and trace metadata; propagate request identifiers across nested components.
Framework APIs change more quickly than the underlying design concepts. Pin the dependency version, record the model and prompt versions, and keep business logic behind small interfaces so a library upgrade can be tested independently. Inspect callback and tracing behavior for accidental capture of prompts, user data, or secrets. A declarative graph can improve visibility for branching workflows, but a plain function sequence is easier to debug when there are only a few steps. Choose abstraction based on actual branching and integration needs.
The framework should be an implementation detail around stable application contracts. Define typed request and response schemas, a narrow tool interface, and a trace event model that survives library upgrades. Test prompts as versioned assets and test parsers with malformed and adversarial outputs. If the application uses a framework memory abstraction, verify exactly when state is written and whether concurrent turns can overwrite each other. Treat callback handlers as production code because they can expose secrets or create unbounded telemetry.
Benchmark a representative task set with and without the framework around the same model provider. Compare successful structured responses, tool-selection errors, p50/p95 latency, token use, dependency startup cost, and trace completeness. A framework should earn its maintenance cost through less integration code or better observability. Do not compare incomparable provider settings or use framework popularity as a production readiness metric. Include failure injection for timeout, invalid JSON, tool denial, and provider rate limiting.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Plain functions | Low abstraction and easy debugging | More adapter code to maintain |
| Composable chains | Reusable fixed workflows | Can obscure data flow when over-nested |
| Agent executor | Dynamic tool selection | Unbounded loops and variable cost |
| Graph orchestration | Explicit branches and checkpoints | More concepts and framework coupling |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def validate_payload(payload, required):
    missing = set(required) - payload.keys()
    if missing: raise ValueError(f'missing fields: {sorted(missing)}')
    return payload

assert validate_payload({'question': 'why?'}, ['question'])['question'] == 'why?'
```

## Failure Modes
- Parser accepts syntactically valid but invalid business data: enforce schema plus semantic invariants before using output.
- Implicit retry repeats a side effect: distinguish safe reads from writes and require idempotency keys.
- Upgrade changes runtime behavior: pin versions and replay a golden task suite before deployment.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| When choose a chain over an agent? | Use a chain for a known sequence; choose an agent only when dynamic tool choice has measurable value. Design follow-up: what is the maximum step budget? |
| What does structured parsing guarantee? | Only output shape and configured validation rules, not truth or authorization. Debug follow-up: how are semantic constraints checked? |
| How do you evaluate framework adoption? | Compare task success, failure visibility, latency, token cost, and upgrade burden against a minimal implementation. |

## Related Concepts
See [Agent frameworks comparison](63-agent-frameworks-comparison.md), [tool calling](04-tool-calling.md), [agent testing](28-agent-testing.md).

## References
- [python.langchain.com](https://python.langchain.com/docs/)
- [python.langchain.com](https://python.langchain.com/docs/concepts/)

## Further Topic Detail
## Core Intuition
A framework is a set of adapters and orchestration parts, similar to a workshop bench with standard clamps. It speeds assembly when components fit the task, but the bench does not guarantee that the finished machine is safe. Framework abstractions can hide retries, prompts, and tool boundaries, so engineers must know what actually executes.


## How It Works
A reliable application makes each boundary explicit: input validation, prompt construction, model invocation, parsing, tool authorization, and final response. A fixed chain is appropriate when order is known; an agent executor is justified when choosing among tools is part of the task. Parsers validate shape but do not prove factual correctness. Retrievers return candidate evidence but do not ensure the model used it faithfully. Wrap each external call with timeout, bounded retry, and trace metadata; propagate request identifiers across nested components.
Framework APIs change more quickly than the underlying design concepts. Pin the dependency version, record the model and prompt versions, and keep business logic behind small interfaces so a library upgrade can be tested independently. Inspect callback and tracing behavior for accidental capture of prompts, user data, or secrets. A declarative graph can improve visibility for branching workflows, but a plain function sequence is easier to debug when there are only a few steps. Choose abstraction based on actual branching and integration needs.
The framework should be an implementation detail around stable application contracts. Define typed request and response schemas, a narrow tool interface, and a trace event model that survives library upgrades. Test prompts as versioned assets and test parsers with malformed and adversarial outputs. If the application uses a framework memory abstraction, verify exactly when state is written and whether concurrent turns can overwrite each other. Treat callback handlers as production code because they can expose secrets or create unbounded telemetry.
Benchmark a representative task set with and without the framework around the same model provider. Compare successful structured responses, tool-selection errors, p50/p95 latency, token use, dependency startup cost, and trace completeness. A framework should earn its maintenance cost through less integration code or better observability. Do not compare incomparable provider settings or use framework popularity as a production readiness metric. Include failure injection for timeout, invalid JSON, tool denial, and provider rate limiting.


## Detailed Explanation

Build complex language model applications using chains, agents, memory, and tool integrations



## Core Intuition

Build complex language model applications using chains, agents, memory, and tool integrations



## How It Works

1. LLMs (language models): wrappers for OpenAI, HuggingFace, etc.
2. Chains: sequence of calls to LLMs and other tools
   - LLMChain: template → format input → LLM → parse output
   - SequentialChain: run chains in sequence, pass outputs
3. Agents: LLM decides which tools to use iteratively
   - Think: LLM decides next action
   - Act: execute tool
   - Observe: see result
   - Repeat
4. Memory: store conversation history, retrieved context
5. Tools: calculators, search, databases, APIs





## Interview Q&A


**Q: What's the difference between chains and agents?**
A: Chains: predetermined sequence of steps. Agents: LLM decides steps dynamically. Chains are deterministic, agents flexible. Use chains for fixed workflows, agents for reasoning-based decisions.

**Q: How does LangChain memory work?**
A: Stores messages (user, assistant) in conversation. Types: ConversationBufferMemory (all), ConversationSummaryMemory (summarize old), ConversationKGMemory (knowledge graph). Tradeoff: length vs richness.

**Q: What are prompts and prompt templates in LangChain?**
A: Templates: reusable prompt structures with variables. Example: 'Analyze {text} for {sentiment}'. At runtime, variables are filled. Enables prompt reuse, versioning, testing.

**Q: How do you handle errors in chains?**
A: Try-catch in code, but also: ValidationError for bad format, LLMException for API fails. Add fallbacks: if tool fails, try alternative. Retry with backoff for transient errors.

**Q: What is output parsing and why is it needed?**
A: LLMs output free text; need to extract structured data. Parsers: JSON (force format), PydanticOutputParser (validate schema), CommaSeparatedListOutputParser. Critical for using outputs downstream.
