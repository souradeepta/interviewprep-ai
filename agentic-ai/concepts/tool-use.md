# Tool Use

## TL;DR
Give LLMs access to external functions/APIs to take actions beyond text generation. Agent tells LLM "you can use calculator, web search, database," LLM decides when to call them, system executes and returns results. Foundation of agentic AI; enables real-world impact.

## Core Intuition
LLMs are stateless and can't perform actions—they only generate text. Tool use is the bridge: define functions the agent can call, let LLM decide when and with what arguments. Tool output becomes context for next reasoning step. Iterate until task complete.

## How It Works

**Tool Definition (Schema):**
```json
{
  "name": "calculator",
  "description": "Performs arithmetic calculations",
  "input_schema": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "Mathematical expression (e.g., '5 + 3 * 2')"
      }
    },
    "required": ["expression"]
  }
}
```

**Agent Loop with Tool Use:**
```
1. User: "What's 42 * 17?"
2. LLM (with tools available): "I'll use the calculator tool to compute this."
   → Tool use: calculator(expression="42 * 17")
3. System executes: 42 * 17 = 714
4. LLM receives result: "The answer is 714."
5. LLM generates: "The result is 714."
```

**Structured Tool Calling (XML Format):**
```
<tool_use>
  <invoke name="calculator">
    <parameter name="expression">42 * 17</parameter>
  </invoke>
</tool_use>

System returns:
Result: 714
```

**Multiple Tools:**
```
Agent can choose between:
- calculator: math
- web_search: current info
- database_query: fetch user data
- code_executor: run scripts

LLM learns to call appropriate tool for task.
```

**Tool Result Integration:**
```
After tool executes:
  Assistant: "I used calculator and got 714."
  User: "Got it, now multiply by 2"
  Assistant: "I'll call calculator again."
    → calculator(expression="714 * 2")
  Result: 1428
  Assistant: "The result is 1428."
```

## Key Properties / Trade-offs

| Aspect | No Tools | Tool Use |
|--------|----------|----------|
| Accuracy | Hallucinate answers | Real data from tools |
| Real-world impact | Limited (text only) | High (can perform actions) |
| Latency | Fast (one LLM call) | Slower (LLM + tool calls) |
| Cost | Low (1 LLM call) | Higher (LLM + tools) |
| Determinism | Repeatable | Non-deterministic (LLM sampling) |
| Error handling | Simple | Complex (tool failures) |

**Tool Categories:**

| Type | Examples | Use Case |
|------|----------|----------|
| Computation | calculator, code execution | Math, data processing |
| Retrieval | web search, database query | Information lookup |
| Action | email send, API calls | Triggering workflows |
| Perception | image recognition, OCR | Processing inputs |
| Reasoning | symbolic solver | Structured reasoning |

## Common Mistakes / Gotchas

- **Vague tool descriptions:** "compute" is too vague. "Perform arithmetic on mathematical expressions (e.g., '5+3', '2**3')" is clear.
- **Missing input validation:** LLM might call calculator("hello"). Add input validation and error handling.
- **Not handling tool failures:** Network timeout, API error, invalid input. Always provide fallback and inform LLM of failure.
- **Tool confusion:** Too many similar tools confuse LLM. Organize by domain or function. Use clear, distinct names.
- **Infinite loops:** LLM gets stuck calling same tool. Add max_calls limit per task + tracking of already-called tools.
- **Not returning context:** If tool fails, return informative error to LLM (not just "error"). LLM needs to understand what went wrong.
- **Slow tools:** If tool takes >5s, LLM timeout or latency issues. Async execution or caching needed.
- **Security issues:** Don't expose dangerous tools (system commands, database deletes) to unvetted agents. Validate and sandbox.

## Code Example

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "calculator",
        "description": "Perform arithmetic calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression (e.g., '5 + 3 * 2')"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

def execute_tool(name, input_dict):
    """Execute a tool and return result."""
    if name == "calculator":
        try:
            result = eval(input_dict["expression"])  # NOTE: unsafe for untrusted input
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    elif name == "web_search":
        # Simulate web search
        return {"success": True, "results": f"Search results for '{input_dict['query']}'"}
    
    return {"success": False, "error": f"Unknown tool: {name}"}

def run_agent(user_message):
    """Run agent loop with tool use."""
    messages = [{"role": "user", "content": user_message}]
    
    for step in range(10):  # Max 10 steps
        # Get LLM response
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Check if LLM wants to use a tool
        if response.stop_reason == "tool_use":
            # Find tool use block
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})
            
            # Process each tool use
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Using tool: {block.name} with input: {block.input}")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
            
            messages.append({"role": "user", "content": tool_results})
        
        else:
            # LLM finished (no more tool use)
            final_response = next(
                (block.text for block in response.content if hasattr(block, 'text')),
                None
            )
            print(f"Agent response: {final_response}")
            return final_response
    
    return "Max steps reached"

# Run agent
run_agent("What's 42 * 17? Then search for the significance of that number.")
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "What is tool use?" | Enable LLMs to call external functions/APIs. Agent decides when and with what arguments. System executes and returns results. |
| "Why important?" | LLMs can only generate text. Tools let them take real actions (calculate, search, update databases). |
| "How to define tools?" | Provide name, description, and input schema (JSON format). Clear descriptions help LLM choose correctly. |
| "Handle tool failures?" | Always return informative error to LLM. Set max_calls limit. Add timeout and fallback strategies. |
| "Security?" | Sandbox tools, validate inputs, don't expose dangerous operations (file deletes, system commands). |
| "Latency?" | Tools add latency. Use async execution, caching, or batch processing for slow operations. |

## Related Topics
- [Function Calling](function-calling.md) — more formal structure for tool invocation
- [Tool Calling](tool-calling.md) — patterns and libraries for structured tool use
- [What Is an Agent](what-is-an-agent.md) — tool use is core to agent architecture
- [Error Recovery](error-recovery.md) — handling tool failures gracefully
- [Agent Loops](agent-loops.md) — tool use within the agent loop

## Resources
- [Anthropic: Tool Use and Function Calling](https://docs.anthropic.com/claude/reference/tool-use)
- [LangChain: Tool Integrations](https://python.langchain.com/docs/integrations/tools/)
- [OpenAI: Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Building Reliable Agents with Tool Use (OpenAI Blog)](https://openai.com/blog/function-calling-and-other-api-updates-for-chat-models/)
