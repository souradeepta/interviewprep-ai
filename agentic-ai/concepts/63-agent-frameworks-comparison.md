# Agent Frameworks Comparison
**TL;DR:** Choose an agent framework by required control, state model, tool boundaries, deployment fit, and measured maintenance cost.

## Prerequisites
Agent loops, tool calling, deployment basics, and evaluation.

## Learning Outcomes
- Compare frameworks against concrete workflow requirements.
- Build a small proof of concept that exposes state and tool execution.
- Assess migration and observability costs before adopting abstractions.

## Core Intuition
Framework choice resembles choosing a database library: the most popular option is not automatically suitable, and changing later can be expensive if application logic is coupled to its objects. Compare the boundary it owns with the boundary your team needs to control.

## How It Works
The operating flow identifies the task state, selects a bounded strategy, gathers only the evidence needed for that decision, validates the result, and records the outcome. The highest-value comparison is a vertical slice that exercises state, tools, failure recovery, evaluation, and deployment. A polished hello-world hides the difficult questions: can the framework resume after a worker crash, correlate nested tool calls, redact sensitive traces, and expose cancellation? Add a migration test for stored state and prompts before adoption. Teams should also compare license, maintenance activity, support model, security response, and provider portability using current project evidence.
Create a shortlist from requirements: deterministic workflow or dynamic loop, durable checkpoints, human approval, async jobs, streaming, provider portability, tracing, and local testing. Implement one representative path, including a failure and a resume, in each candidate. Inspect the generated prompt, retry policy, tool arguments, state persistence, and logs. Prefer standard interfaces at application boundaries and keep domain policy outside framework-specific callbacks. Verify current project health and support commitments before relying on a hosted service.
The highest-value comparison is a vertical slice that exercises state, tools, failure recovery, evaluation, and deployment. A polished hello-world hides the difficult questions: can the framework resume after a worker crash, correlate nested tool calls, redact sensitive traces, and expose cancellation? Add a migration test for stored state and prompts before adoption. Teams should also compare license, maintenance activity, support model, security response, and provider portability using current project evidence.
Score proof-of-concepts on task success, auditability, p95 latency, cost, dependency size, developer time, and recovery after worker restart. Include a “no framework” baseline to expose whether the abstraction adds value. Pin exact versions and run the same golden evaluation suite. Avoid framework feature-count comparisons; a feature matters only if it meets a stated requirement without weakening observability or control.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Custom loop | Maximum control and low coupling | More reliability code to own |
| Composable library | Reusable integrations | API churn and hidden defaults |
| Graph framework | Explicit state and checkpoints | Learning and vendor coupling |
| Managed agent service | Fast initial delivery | Portability and control limits |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def required_capabilities(requirements, framework_features):
    return set(requirements) <= set(framework_features)

assert required_capabilities({'checkpoint', 'trace'}, {'checkpoint', 'trace', 'tools'})
```

## Failure Modes
- Framework hides tool retry semantics: inspect execution traces and test duplicate writes.
- Abstraction couples business rules to library types: add application-owned interfaces and migration tests.
- Adoption chosen from a demo only: run representative failure, resume, and load scenarios.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| How do you choose among frameworks? | Write requirements and compare representative prototypes including failure recovery, traceability, and cost. Design follow-up: which capabilities are non-negotiable? |
| When should you write a custom loop? | When workflow is small and framework overhead or coupling exceeds integration value. Debug follow-up: how will you test retry behavior? |
| How do you manage upgrades? | Pin versions and replay contract and regression tests before changing runtime dependencies. |

## Related Concepts
See [LangChain frameworks](53-langchain-frameworks.md), [AutoGen orchestration](54-autogen-orchestration.md), [agent deployment](56-agent-deployment-patterns.md).

## References
- [python.langchain.com](https://python.langchain.com/docs/)
- [microsoft.github.io](https://microsoft.github.io/autogen/stable/)

## Further Topic Detail
## Detailed Explanation

Multiple agent frameworks exist with different philosophies, capabilities, and trade-offs: LangChain (flexible, composable), AutoGen (multi-agent orchestration), OpenAI Assistants API (managed infrastructure), Anthropic Claude API (direct model access), and domain-specific frameworks (AgentGPT, Taskweaver). Each represents different points in the spectrum: flexibility (control everything yourself) vs. abstraction (platform handles complexity).

Key comparison dimensions: (1) Level of abstraction (does platform handle agent loop or do you?), (2) Multi-agent capabilities (can agents coordinate?), (3) Integration breadth (what tools/APIs supported?), (4) Cost structure (per-API-call or managed service?), (5) Flexibility (can you customize reasoning?), (6) Learning curve (easy to start or requires system design understanding?). No framework is universally best—selection depends on use case: building a simple chatbot (Assistants API), complex multi-step systems (LangChain), multi-agent coordination (AutoGen), research/prototyping (direct API access).

Understanding framework trade-offs is crucial because choosing wrong creates unnecessary complexity (over-engineered simple systems) or insufficient capability (frameworks too limited for your needs). Evaluating frameworks requires clarity about your requirements: Do you need multi-agent coordination? How much customization? What integrations matter?


## Core Intuition

Frameworks are like restaurants: some offer full service (staff handles everything, you just order), some are buffet (you pick what you want from available options), some are ingredient suppliers (you cook yourself). Different situations call for different approaches—not about good or bad restaurants.


## How It Works

1. LangChain: chains (templates), agents (tool use), memory, integrations
2. AutoGen: multi-agent conversations, automatic role assignment, flexible interactions
3. ReAct: reasoning + acting loop, simple structure, clear reasoning trace
4. Comparison: ease of use, flexibility, production readiness, community
5. LangChain: most popular, best integrations, middle learning curve
6. AutoGen: best for multi-agent, experimental, fewer integrations
7. ReAct: simplest, good for learning, limited extensibility




## Interview Q&A


**Q: Which framework should you choose for a new project?**
A: Start with LangChain: most mature, best docs, most integrations. Use AutoGen: if multi-agent is core (AutoGen excels here). Use ReAct: if learning or prototyping (simple, clear). In production: depends on team expertise and requirements.

**Q: How do frameworks differ in handling state?**
A: LangChain: explicit memory management (you control state). AutoGen: implicit (framework manages conversation). ReAct: minimal state (just reasoning trace). Tradeoff: control vs convenience. ReAct simplest, LangChain most flexible.

**Q: What's the learning curve for each framework?**
A: ReAct: lowest (simple loop, 1-2 hours to understand). LangChain: medium (many concepts, 1-2 days to productive). AutoGen: medium (new paradigm, 1-2 days to productive). All have good docs/tutorials.

**Q: Can you mix frameworks (e.g., LangChain agent with AutoGen?)**
A: Possible but rare: different abstractions, not designed to interop. Better: choose one and stick with it. If need features from both: evaluate if really needed, or wait for frameworks to converge.

**Q: Which framework is best for production?**
A: LangChain: if you want full control and maturity. AutoGen: if you want multi-agent as core. ReAct: not designed for production (too simple). For production: add monitoring, error handling, deployment logic on top of any framework.
