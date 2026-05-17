# Planning & Reasoning

## TL;DR
Structured approaches for agents to break down goals into steps and reason through solutions. Methods: decomposition (break task into subtasks), chain-of-thought (show reasoning steps), tree-of-thought (explore multiple paths), MCTS (Monte Carlo tree search). Better planning → fewer tool calls, lower cost, higher reliability.

## Core Intuition
Humans solve complex problems by breaking them down ("I need to: 1) gather info, 2) analyze, 3) decide"). Agents should do the same. Instead of jumping to action, agents reason through a plan first. More thinking upfront → better actions → fewer mistakes.

## How It Works

**No Planning (Naive):**
```
Task: "Book a flight"
LLM: "I'll call book_flight_api"
Result: Failed (didn't check dates, no budget, etc.)
```

**With Planning:**
```
Task: "Book a flight"
LLM thinks: "I need to: 
  1) Ask user for preferences (dates, budget, destination)
  2) Search flights matching criteria
  3) Check user budget against options
  4) Confirm before booking
  5) Book the selected flight"
Then executes step-by-step with feedback.
```

**Chain-of-Thought (CoT):**
```
Prompt: "What's the answer to: If John has 3 apples and buys 2 more, how many does he have?"

Without CoT:
LLM: "5"

With CoT:
LLM: "John starts with 3 apples. He buys 2 more. 3 + 2 = 5. So John has 5 apples."
(Shows reasoning steps explicitly)
```

**Task Decomposition:**
```
Goal: "Analyze sales data and write report"

Decomposed:
  1. Load sales data from database
  2. Clean and preprocess
  3. Compute key metrics (revenue, growth, churn)
  4. Identify trends and anomalies
  5. Create visualizations
  6. Write executive summary
  7. Attach visualizations

Agent handles each subtask, passes results to next step.
```

**Tree-of-Thought (ToT):**
```
Instead of one linear path, explore multiple reasoning branches:

                    Start
                      |
          ____________|____________
         /            |            \
      Path 1       Path 2        Path 3
       / \          / \          / \
    ... ...      ... ...      ... ...
     |             |             |
   Bad          Good          Bad
   
Agent evaluates each path, follows most promising, backtracks if stuck.
```

**Multi-Step Reasoning with Tool Use:**
```
1. Agent receives task
2. Breaks into steps (planning)
3. For each step:
   a. Reasons about what tool/action needed
   b. Executes tool
   c. Observes result
   d. Adjusts plan based on new info
4. Repeats until goal achieved or max steps
```

## Key Properties / Trade-offs

| Approach | Reasoning Cost | Action Cost | Accuracy | Latency |
|----------|---|---|---|---|
| No planning | None | High (wrong tools) | Low | Fast |
| CoT | Low (1-2 steps) | Medium | Medium | Fast |
| Decomposition | Medium (break down) | Low (clear steps) | High | Slow |
| Tree-of-Thought | High (multiple paths) | Low | High | Slow |
| MCTS | Very High | Low | Very High | Slow |

**When to use each:**
- Simple tasks: no planning needed
- Domain-specific: decomposition works well
- Ambiguous tasks: CoT helps clarity
- High-stakes: tree-of-thought or MCTS for thorough exploration

## Common Mistakes / Gotchas

- **Over-planning:** Planning for 10 minutes on a 1-minute task. Set reasonable reasoning budget.
- **Rigid plans:** Plan changes as new info arrives. Agent should adapt, not blindly follow initial plan.
- **Assuming LLM reasoning is correct:** CoT helps, but LLM can still reason incorrectly. Validate results.
- **Tree explosion:** Exploring all paths leads to exponential growth. Use heuristics to prune unlikely branches.
- **No fallback:** If plan fails, agent needs alternative. Add re-planning on failure.
- **Missing context in decomposition:** Breaking down without understanding dependencies fails. Keep context across steps.
- **Latency explosion:** Each planning step adds LLM call. Budget calls carefully; batch where possible.

## Code Example

```python
from anthropic import Anthropic

client = Anthropic()

def agent_with_planning(task):
    """Agent that plans before acting."""
    messages = []
    
    # Step 1: Ask LLM to plan
    planning_prompt = f"""
    Task: {task}
    
    Before taking action, break this task into clear steps. What will you do?
    Respond with numbered steps (1. ... 2. ... etc.)
    """
    messages.append({"role": "user", "content": planning_prompt})
    
    # Get plan
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=messages
    )
    plan = response.content[0].text
    print(f"Plan: {plan}")
    
    # Step 2: Execute plan with feedback
    messages.append({"role": "assistant", "content": plan})
    execution_prompt = f"""
    Now execute this plan. After each step, report what you found.
    Tools available: search_engine, calculator, database_query
    """
    messages.append({"role": "user", "content": execution_prompt})
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=messages
    )
    result = response.content[0].text
    print(f"Result: {result}")
    
    return result

# Example: Complex task requiring planning
task = "Find the total revenue from sales in Q1 2024, compare to Q1 2023, and identify top-performing products"
agent_with_planning(task)

# -----------

def agent_chain_of_thought(question):
    """Agent using chain-of-thought reasoning."""
    cot_prompt = f"""
    Answer the following question. Show your reasoning step-by-step:
    
    {question}
    
    Think through it:
    1. What do I know?
    2. What steps do I need?
    3. Compute / reason through
    4. Final answer
    """
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": cot_prompt}]
    )
    print(response.content[0].text)

# Example
agent_chain_of_thought("If a store has 100 shirts, sells 23, then receives 45 new shipment, how many shirts do they have?")
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Why planning?" | Complex tasks require breaking down. Planning upfront → fewer mistakes, better tool use, lower cost. |
| "CoT effectiveness?" | Chain-of-Thought improves reasoning on complex tasks. Explicitly showing steps helps LLM and humans verify logic. |
| "When decomposition?" | Structured, multi-step tasks (data pipelines, workflows). Each subtask can be solved independently. |
| "Tree-of-Thought?" | For high-stakes, ambiguous tasks. Explore multiple paths, backtrack if stuck. More expensive but thorough. |
| "Planning overhead?" | Each planning step is an LLM call. Budget reasonably; don't plan for trivial tasks. |
| "Handle plan changes?" | Collect feedback after each step. Re-plan if results differ from expectations. |

## Related Topics
- [ReAct (Reasoning + Acting)](react-reasoning-acting.md) — formalized reasoning + action framework
- [Tree of Thought](tree-of-thought.md) — explore multiple reasoning paths
- [MCTS for Agents](mcts-for-agents.md) — principled tree search for planning
- [Agent Loops](agent-loops.md) — looping structure for multi-step tasks

## Resources
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)
- [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601)
- [ReAct: Synergizing Reasoning and Acting in LLMs](https://arxiv.org/abs/2210.03629)
- [Reasoning in Large Language Models](https://openai.com/research/)
