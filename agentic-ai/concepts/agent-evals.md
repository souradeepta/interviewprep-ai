# Agent Evals

## TL;DR
Test agents systematically on tasks: does the agent reach the goal? Choose right tools? Avoid hallucinations? Metrics: task success rate, tool accuracy, latency, cost. Use benchmarks (public datasets) and custom evals (production tasks). Enables confidence in deployment.

## Core Intuition
Agents are probabilistic (non-deterministic LLM sampling). Same task may succeed sometimes, fail others. Need systematic testing on many examples to measure reliability. Can't just run once and declare "it works."

## How It Works

**Evaluation Framework:**

**1. Define Success Criteria:**
```
Task: "Book a flight from NYC to LA on Jan 15, budget $500"

Success metrics:
  - Task completion: booked a flight (yes/no)
  - Constraint satisfaction: date = Jan 15, price ≤ $500
  - Conversation quality: ≤ 5 tool calls (efficiency)
  - Tool usage: no hallucinated tool results
```

**2. Create Test Cases:**
```
Test case 1:
  Input: "Book NYC to LA Jan 15, $500 budget"
  Expected: Flight booked, correct date/price
  
Test case 2:
  Input: "Book NYC to LA Jan 15, $100 budget" (impossible)
  Expected: Agent rejects (too expensive), fallback recommendation
  
Test case 3:
  Input: "Book NYC to LA" (incomplete)
  Expected: Agent asks clarifying questions
```

**3. Run Evaluation:**
```
For each test case:
  - Run agent
  - Check success (task completed?)
  - Check constraints (all met?)
  - Log tool calls, latency, cost
  - Human review (was output sensible?)
```

**4. Aggregate Metrics:**
```
Success rate: 45/50 = 90%
Tool accuracy: 48/50 correct calls (96%)
Avg latency: 2.3 seconds
Avg cost: $0.15 per task
Human satisfaction: 4.2/5
```

**Evaluation Methods:**

**Automated Metrics:**
```
- Task completion: did agent reach goal?
- Constraint adherence: all constraints satisfied?
- Tool call validity: tools exist, params correct?
- Hallucination: agent claims thing that didn't happen?
- Latency: how long did task take?
- Cost: $ spent on LLM calls + tool usage
```

**Human Evaluation:**
```
- Quality: was output sensible?
- Helpfulness: did it actually help user?
- Safety: any harmful actions?
- Efficiency: was path reasonable?
- Consistency: does it behave predictably?

Typical: 5-100 samples, 1-2 raters, Kappa ≥ 0.7
```

**Benchmarks:**
```
Public:
  - GAIA: generalist agent tasks (complex, multi-step)
  - WebArena: web automation (booking, searching)
  - ToolBench: tool-using capability
  - InteractiveCodeSearch: code retrieval and execution

Custom:
  - Production queries (log real user tasks)
  - Edge cases (failure modes you care about)
  - Regression tests (ensure new changes don't break)
```

## Key Properties / Trade-offs

| Evaluation Type | Cost | Reliability | Coverage |
|---|---|---|---|
| Automated metrics | Free | Medium (success ≠ quality) | Limited (what you code) |
| Human evals | High | High | Limited (100s examples) |
| Automated + Human | Medium | High | Good |
| Public benchmarks | Free | Medium | Broad (general) |
| Custom evals | Low-Medium | High | Specific (your tasks) |

**Evaluation Cadence:**
```
Development:
  - Continuous: after each change, run full eval suite
  
Pre-deployment:
  - Comprehensive: 100+ test cases, human review top failures
  
Production:
  - Monitoring: track success rate, catch regressions
  - Periodic: monthly deep-dive on failure patterns
```

## Common Mistakes / Gotchas

- **Evaluating once:** Agents are stochastic. Evaluate 10+ times per task to get distribution.
- **Shallow success criteria:** "Reached goal" is vague. Define specific, measurable constraints.
- **Not tracking intermediate steps:** Agent may reach goal but took absurd path. Track tool calls, reasoning.
- **Biased test set:** Easy tasks → high scores → false confidence. Include edge cases, adversarial examples.
- **No baseline:** "90% success" is only meaningful compared to baseline (human? random? simpler agent?).
- **Ignoring cost:** Agent A: 95% accuracy, $10/task. Agent B: 92% accuracy, $0.50/task. B is often better. Track cost.
- **Human evals without standards:** Raters need rubric, examples. Without standards, agreement low (Kappa < 0.5).
- **Not handling ties:** Sometimes task has multiple correct answers (e.g., multiple flights). Accept any valid solution.

## Code Example

```python
import json
from anthropic import Anthropic

client = Anthropic()

# Define tools
tools = [
    {
        "name": "search_flights",
        "description": "Search for flights",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_city": {"type": "string"},
                "to_city": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "max_price": {"type": "number"}
            },
            "required": ["from_city", "to_city", "date"]
        }
    },
    {
        "name": "book_flight",
        "description": "Book a flight",
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string"},
                "passenger_name": {"type": "string"}
            },
            "required": ["flight_id", "passenger_name"]
        }
    }
]

def evaluate_agent(test_cases):
    """Evaluate agent on test cases."""
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== Test Case {i+1}: {test_case['name']} ===")
        
        # Run agent
        messages = [{"role": "user", "content": test_case["task"]}]
        
        success = False
        for step in range(10):  # Max 10 steps
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=tools,
                messages=messages
            )
            
            if response.stop_reason == "tool_use":
                # Process tool calls
                messages.append({"role": "assistant", "content": response.content})
                
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        # Simulate tool execution
                        result = simulate_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })
                
                messages.append({"role": "user", "content": tool_results})
            else:
                # Agent finished
                final_response = next(
                    (block.text for block in response.content if hasattr(block, 'text')),
                    None
                )
                print(f"Agent response: {final_response}")
                
                # Check success
                success = test_case["check"](final_response)
                break
        
        # Record result
        status = "PASS" if success else "FAIL"
        print(f"Result: {status}")
        results["details"].append({
            "test": test_case["name"],
            "passed": success
        })
        
        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Report
    print(f"\n=== Summary ===")
    print(f"Passed: {results['passed']}/{results['total']} ({100*results['passed']/results['total']:.1f}%)")
    return results

def simulate_tool(tool_name, input_dict):
    """Simulate tool execution."""
    if tool_name == "search_flights":
        return {
            "flights": [
                {"id": "FL001", "price": 350, "date": "2024-01-15"},
                {"id": "FL002", "price": 450, "date": "2024-01-15"},
            ]
        }
    elif tool_name == "book_flight":
        return {"status": "booked", "confirmation": "CONF123"}
    return {"error": f"Unknown tool: {tool_name}"}

# Define test cases
test_cases = [
    {
        "name": "Simple booking",
        "task": "Book a flight from NYC to LA on Jan 15, with $500 budget",
        "check": lambda response: "booked" in response.lower() or "confirmation" in response.lower()
    },
    {
        "name": "Budget constraint",
        "task": "Book a flight from NYC to LA on Jan 15, with $300 budget",
        "check": lambda response: "expensive" in response.lower() or "no flights" in response.lower()
    },
    {
        "name": "Missing info",
        "task": "Book a flight from NYC",
        "check": lambda response: "date" in response.lower() or "where" in response.lower()
    }
]

# Run evaluation
results = evaluate_agent(test_cases)
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "How to eval agents?" | Task completion (yes/no), constraint satisfaction, tool accuracy, latency, cost. Use both auto + human. |
| "Success rate ≠ quality?" | Right. Task may complete but inefficiently (many tool calls) or unsafely. Track intermediate steps too. |
| "How many tests?" | At least 30-50 per category (simple, complex, edge cases). 1000+ for production confidence. |
| "Human eval importance?" | Automated can't catch all issues (unsafe actions, poor reasoning). Human review critical. |
| "Benchmarks?" | Use public (GAIA, WebArena) for general capability. Build custom for production tasks. |
| "Track regression?" | Yes. Keep test suite from old version, ensure new version doesn't fail them. |

## Related Topics
- [Agent Debugging](agent-debugging.md) — when evals fail, how to debug
- [Agent Testing](agent-testing.md) — unit tests for components
- [Agent Monitoring](agent-monitoring.md) — continuous eval in production
- [Safety & Alignment](safety-alignment.md) — evals for safe behavior

## Resources
- [GAIA: A Benchmark for General AI Assistants](https://huggingface.co/spaces/gaia-benchmark/leaderboard)
- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://webarena.dev/)
- [ToolBench: Benchmark for Tool-Using Language Models](https://github.com/OpenBMB/ToolBench)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
