# Function Calling

TL;DR: Function Calling is a design pattern for making an agent more reliable by controlling how it function calling under explicit constraints.

## Prerequisites

This lesson assumes familiarity with agentic-ai/tool-calling. You should be able to describe an LLM request/response cycle, inspect inputs and outputs, and reason about the application boundary where this technique runs. The lesson focuses on function calling as an engineering choice: identify the problem it solves, the resources it consumes, and the evidence needed to know whether it works.

## Learning Outcomes

By the end, you can:
- Explain the mechanism of function calling and name its key assumptions.
- Implement a minimal version and trace one successful and one failing execution.
- Compare it with a simpler alternative using quality, latency, cost, and safety measures.
- Diagnose a failure from observable evidence and propose a bounded recovery.

## Core Intuition

Treat function calling as a controlled decision in a larger system rather than a magic capability. State the goal, allowed inputs, permitted actions, and stopping condition first. Then trace how information changes at each step. This makes the design reviewable: a teammate can inspect what evidence led to a decision, what the system did, and whether the result met the task contract. The central engineering question is whether the extra mechanism improves outcomes enough to justify its additional complexity and operational cost.

## How It Works

A practical design starts with a task boundary and an observable success criterion. The orchestrator collects the relevant state, selects the next operation, validates its inputs, and records the result. It then checks whether the goal is complete or whether another bounded step is justified. Keep the state passed between steps explicit and small; include only data that affects the decision. Separate policy from execution so a decision can be inspected before it causes an external effect. When possible, make operations idempotent and attach a stable identifier so retries do not duplicate work.

For function calling, the important implementation detail is to make the decision point testable. Define the expected input shape, valid output shape, allowed resource budget, timeout, and fallback behavior. Record a trace containing the request identifier, chosen operation, outcome category, and duration, while excluding secrets and unnecessary personal data. A good trace supports diagnosis without requiring access to hidden model reasoning. Use representative examples and counterexamples to measure success, then inspect disagreements and boundary cases before increasing autonomy.

## Trade-offs

| Approach | Strength | Cost or risk | Choose when |
|---|---|---|---|
| Direct, single-step response | Lowest latency and simplest control path | Cannot adapt after new evidence | The task is bounded and context already contains needed facts |
| Function Calling | Can improve task completion by making the relevant process explicit | More state, integration points, and failure paths to operate | Evaluation shows a repeatable gain for the target workload |
| Human review or deterministic workflow | Clear accountability and predictable policy enforcement | Slower and may require manual capacity | Errors have high impact or policy requires approval |

Compare alternatives on the same task set. Measure end-to-end task success and constraint violations alongside p50/p95 latency and cost per completed task. Do not count a call as successful merely because it returned valid syntax; verify that the result satisfies the user-facing task and all policy constraints.

## Implementation

The following small Python scaffold makes a bounded decision visible and rejects invalid results before they can influence later steps:

```python
def run_step(decide, state, allowed_actions, max_steps=3):
    trace = []
    for step in range(max_steps):
        choice = decide(state)
        if choice not in allowed_actions:
            return {"ok": False, "reason": "invalid_action", "trace": trace}
        result = allowed_actions.get(choice)(state)
        trace.append({"step": step, "action": choice, "result": result})
        state = {**state, "last_result": result}
        if result.get("done") is True:
            return {"ok": True, "state": state, "trace": trace}
    return {"ok": False, "reason": "step_limit", "trace": trace}
```

In production, validate schemas at the boundary, set per-operation timeouts, redact sensitive trace fields, and make the step limit configurable by task risk. Keep deterministic checks outside the model call so malformed choices and unsafe actions fail closed.

## Failure Modes

- **Unclear task boundary:** the system performs extra work or returns an untestable result. Diagnose by comparing traces with the task contract; mitigate with explicit completion criteria and a deterministic stop check.
- **Invalid or stale state:** later decisions rely on missing, malformed, or outdated evidence. Diagnose with schema and freshness checks; mitigate by validating state at each boundary and refreshing authoritative data.
- **Unbounded retries or work:** cost and latency grow without improving completion. Diagnose by step-count and timeout distributions; mitigate with budgets, backoff for transient errors, and a terminal failure state.
- **Misleading evaluation:** aggregate success hides harmful edge cases. Diagnose by slicing results by task type and impact; mitigate with held-out adversarial cases and human review for high-impact outcomes.

## Interview Quick Reference

| Question | Strong answer | Follow-up probe |
|---|---|---|
| What problem does function calling solve? | It addresses a stated task limitation and should be justified by measured end-to-end improvement. | What baseline would you compare against? |
| How would you keep it reliable? | Validate boundaries, bound resources, log useful traces, and define recovery behavior. | How would you debug a regression? |
| When would you avoid it? | When a simpler method meets the quality target or added complexity exceeds the value. | Which metric changes that decision? |

## Related Concepts

This lesson connects to agentic-ai/tool-calling. It also informs the downstream topics that consume its outputs: schema validation, observability, evaluation, and safe recovery. When designing a system, map these dependencies explicitly so a failure can be localized to the decision, execution, or verification stage.

## References

- [OpenAI API documentation](https://platform.openai.com/docs/guides/function-calling)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

## Detailed Explanation

Function calling is OpenAI's formalized approach to structured tool invocation. While tool calling is the general pattern (LLM requests execution of a function with structured parameters), function calling specifically refers to OpenAI's native implementation: define functions in JSON Schema format, pass to API, LLM returns structured function calls, system executes. Key distinction from generic tool calling: (1) native API support—built into model responses, (2) JSON Schema—OpenAI's preferred format, (3) parallel functions—multiple function calls in single response, (4) type safety—strict parameter validation. Mechanism: (1) define functions as JSON objects with name, description, parameters schema, (2) pass to chat completion with `tools` parameter, (3) LLM returns `tool_calls` in response, (4) parse and execute, (5) return results as tool messages, (6) loop until done. Advantages: seamless integration with OpenAI API, strong type safety, parallel execution built-in. Challenges: OpenAI-specific (not portable), JSON Schema complexity, learning curve. Best for: production systems using OpenAI models, teams familiar with OpenAI ecosystem. Complementary to: Claude's tool_use format (similar concept, different syntax).

## Core Intuition

Function calling is like having a contract with a function provider. You define the contract upfront: "I have these functions available, each takes these parameters with these types." LLM knows the contract and requests function calls following it exactly. The provider executes and returns results. This is more formal than generic tool calling—tighter integration, stricter contracts. Same concept as tool calling, but with OpenAI's specific structure and native API support.

## How It Works

Function calling operates through definition, schema specification, invocation, parsing, and execution:

1. **Function Definition** — Define in JSON Schema: name, description, parameters with types
2. **API Integration** — Pass functions to OpenAI chat completion API
3. **LLM Processing** — LLM reads function definitions, decides which to call
4. **Function Call Response** — LLM returns structured function calls in response
5. **Parsing** — Extract function name and parameters from response
6. **Execution** — Call actual functions with validated parameters
7. **Result Integration** — Return results back to LLM as tool messages
8. **Loop Control** — Continue if more calls needed, or finish

```mermaid
graph TD
    A["Define Functions<br/>JSON Schema"] --> B["Pass to<br/>OpenAI API"]
    B --> C["LLM Receives<br/>Function Defs"]
    C --> D["LLM Decides<br/>to Call Function"]
    D --> E["Return Function<br/>Call in Response"]
    E --> F["Parse<br/>Function Call"]
    F --> G["Validate<br/>Parameters"]
    G --> H{"Valid?"}
    H -->|No| I["Return Error"]
    H -->|Yes| J["Execute<br/>Function"]
    J --> K["Get Result"]
    K --> L["Add as<br/>Tool Message"]
    L --> M{"Done?"}
    M -->|No| C
    M -->|Yes| N["Final<br/>Response"]
```

## Architecture / Trade-offs

**Integration Level:**
- Native API support — Built-in, seamless, easiest
- Custom parsing — More control, more work
- Hybrid — Use both for different scenarios

**Parameter Definition:**
- JSON Schema — OpenAI standard, verbose but precise
- Simplified schemas — Easier but less type safety
- Code-first (from Python types) — Generates schema from code

**Execution Model:**
- Sequential — One function call at a time
- Parallel — Multiple functions in one response (supported)
- Streaming — Stream function calls as LLM generates them

## Interview Q&A

**Q: What's the difference between function calling and tool calling?**
A: Tool calling is the general pattern (LLM requests function execution with structured parameters). Function calling is OpenAI's specific implementation. Same concept, different syntax. Tool calling (general) works across all LLM providers. Function calling (specific) is OpenAI's native feature.

**Q: When should you use OpenAI's function calling vs Claude's tool_use?**
A: If using OpenAI (GPT-4, GPT-3.5), use function calling—native API support, seamless integration. If using Claude, use tool_use (different syntax but same concept). For multi-model support, use adapter layer that translates between formats.

**Q: How do you define functions for the OpenAI API?**
A: JSON Schema format. Define: (1) name—unique function name, (2) description—what function does, include examples, (3) parameters—JSON schema with type, required fields, descriptions. Example:
```json
{
  "name": "calculate",
  "description": "Perform arithmetic. Example: calculate(expression='5+3')",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {"type": "string", "description": "Math expression"}
    },
    "required": ["expression"]
  }
}
```

**Q: Can you call multiple functions in one response?**
A: Yes. OpenAI supports parallel function calls—multiple functions in single response. Return all as tool messages, LLM processes results and continues. Faster than sequential (don't wait for first to finish before second).

**Q: How do you handle function parameter validation?**
A: (1) Schema validation—OpenAI validates against schema before LLM sees it (LLM shouldn't generate invalid), (2) Runtime validation—check parameters before executing, (3) Type coercion—convert types if safe, (4) Error return—if invalid, return error message to LLM (don't crash).

**Q: What's the difference between tool_choice and function_choice?**
A: `tool_choice` (Claude) specifies which tool agent must use. `function_choice` (OpenAI) similar—force function call or specify which function. Use when you know which function is needed upfront. Use `auto` (default) to let LLM decide.

## Best Practices

1. **JSON Schema is Critical** — Spend time getting schema right. Bad schema = LLM confusion, invalid calls.

2. **Detailed Descriptions** — Include examples in description. "Calculate arithmetic. Example: expression='5+3' returns 8."

3. **Type Precision** — Use correct JSON types. `"type": "number"` not `"type": "float"` (JSON doesn't distinguish).

4. **Required vs Optional** — Only mark as required if always present. Too many required fields = confusion.

5. **Enums for Constraints** — Use enum for restricted choices. `"enum": ["approve", "reject"]` is clearer than string.

6. **Support Parallel Calls** — Design system to handle multiple function calls in one response (faster).

7. **Validate Parameters** — Even though OpenAI validates, validate again before executing (defense in depth).

8. **Provide Feedback on Errors** — If function fails, return detailed error to LLM. LLM learns from feedback.

9. **Test with Real API** — Don't just test schema in isolation. Test actual OpenAI responses; real models deviate.

10. **Document Function Contract** — What does function guarantee? "Always succeeds", "May timeout", "May return empty"?

## Common Pitfalls

**Pitfall 1: Overly Complex Schema**
Issue: Schema has nested objects, complex constraints. JSON schema becomes hard to read. LLM confused.
Fix: Flatten if possible. Simple flat structure > nested objects. Use additional_properties: false to block extras.

**Pitfall 2: Missing Examples in Description**
Issue: LLM told "Calculate arithmetic" but no example. Guesses format.
Fix: Include examples. "Calculate arithmetic (e.g., expression='5+3') returns 8."

**Pitfall 3: No Parameter Validation**
Issue: LLM generates parameters. System executes without checking. Crashes on invalid data.
Fix: Validate parameters before executing. Check types, ranges, required fields.

**Pitfall 4: Not Supporting Parallel Calls**
Issue: System handles one function call at a time. LLM could call multiple simultaneously, but system doesn't support.
Fix: Design for parallel. Parse all function calls from response, execute in parallel (asyncio).

**Pitfall 5: Unclear Required Fields**
Issue: Mark many fields as required. LLM required to provide all, even when unsure. Hallucinations.
Fix: Only required if always present. Make optional if LLM might not know value.

**Pitfall 6: No Type Coercion**
Issue: Schema says integer, LLM sends "5" (string). Validation fails. Error.
Fix: Coerce types if safe (string "5" → int 5). Or handle gracefully.

**Pitfall 7: Functions Not Actually Called**
Issue: Define functions, pass to API, LLM returns calls. System parses but never actually executes.
Fix: Implement execution layer. Parse → validate → execute → return results.

**Pitfall 8: JSON Schema Misunderstanding**
Issue: Use wrong JSON schema format. LLM doesn't understand parameters. Invalid calls.
Fix: Learn JSON Schema spec. Test schema with validator. Start simple, add complexity gradually.

## Code Examples

### Example 1: OpenAI Function Definition

```python
from openai import OpenAI

client = OpenAI()

# Define functions
functions = [
    {
        "name": "calculate",
        "description": "Perform arithmetic calculations. Examples: '5+3', '10*4', '100/2'",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results (default: 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
]

def handle_function_calls(response):
    """Process function calls from OpenAI response."""
    if response.choices[0].finish_reason == "tool_calls":
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "calculate":
                result = eval(arguments["expression"])
                print(f"Calculated: {result}")
            
            elif function_name == "web_search":
                query = arguments["query"]
                limit = arguments.get("limit", 10)
                print(f"Searching: {query} (limit: {limit})")

# Make API call with functions
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's 42 * 17? Then search for its significance."}],
    tools=[{"type": "function", "function": f} for f in functions]
)

handle_function_calls(response)
```

### Example 2: Parallel Function Calls

```python
def handle_parallel_calls(response):
    """Handle multiple function calls in parallel."""
    if response.choices[0].finish_reason == "tool_calls":
        calls = response.choices[0].message.tool_calls
        print(f"Processing {len(calls)} function calls in parallel")
        
        results = []
        for tool_call in calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # Execute (in real system, parallelize with asyncio)
            if function_name == "calculate":
                result = eval(arguments["expression"])
            elif function_name == "web_search":
                result = f"Search results for {arguments['query']}"
            
            results.append({
                "tool_call_id": tool_call.id,
                "function": function_name,
                "result": result
            })
            print(f"  ✓ {function_name}: {result}")
        
        return results
```

### Example 3: Function Validation and Error Handling

```python
from jsonschema import validate, ValidationError

class FunctionCallValidator:
    def __init__(self, functions):
        self.functions = {f["name"]: f for f in functions}
    
    def validate_call(self, function_name: str, arguments: dict) -> tuple[bool, str]:
        """Validate function call against schema."""
        if function_name not in self.functions:
            return False, f"Function not found: {function_name}"
        
        schema = self.functions[function_name]["parameters"]
        try:
            validate(instance=arguments, schema=schema)
            return True, ""
        except ValidationError as e:
            return False, f"Validation error: {e.message}"
    
    def execute_with_validation(self, function_name: str, arguments: dict):
        """Execute function with validation."""
        valid, error = self.validate_call(function_name, arguments)
        
        if not valid:
            return {"success": False, "error": error}
        
        try:
            if function_name == "calculate":
                result = eval(arguments["expression"])
            else:
                result = f"Result for {function_name}"
            
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": f"Execution failed: {str(e)}"}

# Usage
validator = FunctionCallValidator(functions)
result = validator.execute_with_validation("calculate", {"expression": "5 + 3"})
print(result)  # {"success": True, "result": 8}
```

## Related Concepts

- **Tool Calling** — General pattern (function calling is OpenAI's implementation)
- **Tool Use** — Broader concept of using external tools
- **Structured Output** — Constraining LLM outputs to schema
- **Agent Loops** — Function calling within agent loop
- **Error Recovery** — Handling function call failures

