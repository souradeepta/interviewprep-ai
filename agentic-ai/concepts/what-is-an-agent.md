# What Is an Agent?

## TL;DR
An agent is a system that perceives its environment and takes actions to achieve goals. AI agents use an LLM as their "brain," reasoning steps (ReAct, chain-of-thought) as their "planning," and tools (APIs, calculators, search) as their "hands." Core loop: think → act → observe → repeat.

## Core Intuition
A human solving a hard problem: thinks through steps, uses tools (calculator, search), observes results, adjusts strategy. AI agents do the same — the LLM "thinks," tools let it "act," the environment gives feedback.

## How It Works

**Basic agent loop:**
1. **Perception:** agent receives observation (user query, environment state)
2. **Reasoning:** LLM thinks about next action (via prompting)
3. **Action:** agent executes tool (API call, function, etc.)
4. **Observation:** tool returns result
5. **Repeat** until goal reached or max steps

**ReAct (Reasoning + Acting):** alternate between thought (reasoning step) and action (tool call).

**Tool use:** agent declares which tool to use + arguments in a structured format (XML, JSON). System executes and returns result.

**Memory:** agents maintain context (previous steps, results) to inform future decisions.

## Key Properties / Trade-offs
- **Agency cost:** more steps = more LLM calls = higher latency and cost. Optimize tool choice.
- **Determinism:** agents are non-deterministic (LLM sampling). Same query may give different answers. Address with temperatures, retries, voting.
- **Hallucination risk:** LLM may invent tool outputs or misunderstand results. Validate all tool calls and outputs.

## Common Mistakes / Gotchas
- **Unbounded loops:** agent gets stuck retrying same tool. Add max_steps + fallbacks.
- **Poor tool design:** vague tool names/descriptions → agent calls wrong tool. Be explicit in tool specs.
- **Ignoring latency:** real-time applications need fast tools. Batch or cache when possible.

## Code Example
```python
# Pseudocode for a basic agent
from anthropic import Anthropic

client = Anthropic()
tools = [
    {"name": "calculator", "description": "Compute math", 
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}}},
    {"name": "search", "description": "Search the web",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
]

messages = [{"role": "user", "content": "What's 42 * 17?"}]
for step in range(10):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022", max_tokens=1024,
        tools=tools, messages=messages)
    if response.stop_reason == "tool_use":
        tool_use = next(b for b in response.content if b.type == "tool_use")
        result = execute_tool(tool_use.name, tool_use.input)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": f"Tool result: {result}"})
    else:
        print(next(b.text for b in response.content if hasattr(b, 'text')))
        break
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is an agent?" | An LLM + tools + reasoning loop. Thinks, acts, observes, repeats until goal reached. |
| "ReAct vs simple tool calling?" | ReAct adds explicit reasoning steps (thought, action, observation). Better for complex tasks. |
| "How do you prevent infinite loops?" | Max steps limit + early stopping + fallback strategies. |

## Related Topics
- [Tool Use](tool-use.md) — [Memory Types](memory-types.md) — [Planning & Reasoning](planning-reasoning.md)

## Resources
- [ReAct: Synergizing Reasoning and Acting in LLMs](https://arxiv.org/abs/2210.03629)
- [Agentic AI (Anthropic blog)](https://www.anthropic.com/research/agents)
