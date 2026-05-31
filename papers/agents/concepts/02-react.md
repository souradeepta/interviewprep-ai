---
title: "ReAct: Synergizing Reasoning and Acting in Language Models"
authors: "Yao, Zhao, Pitis, et al."
year: 2022
venue: "ICLR"
arxiv: "https://arxiv.org/abs/2210.03629"
domain: "agents"
difficulty: "advanced"
interview_frequency: "high"
related_concepts:
  - agentic-ai/concepts/XX-tool-use
  - agentic-ai/concepts/XX-agent-loops
---

# ReAct: Synergizing Reasoning and Acting in Language Models

## Paper Overview

**Title:** ReAct: Synergizing Reasoning and Acting in Language Models

**Authors:** Shunyu Yao, Jeffrey Zhao, Gregory Pitis, et al. (Princeton University / DeepMind)

**Published:** ICLR 2023 | [arXiv](https://arxiv.org/abs/2210.03629)

**Citation:** 5,000+ (foundational for modern AI agents)

Chain-of-Thought (CoT) showed that reasoning helps models solve problems. But reasoning alone isn't enough—many tasks require *action*: retrieving information from the internet, using a calculator, checking a database. ReAct introduces a unified framework that interleaves **reasoning** (thinking) with **acting** (using tools). A ReAct agent generates a thought ("I need to find information about X"), then performs an action ("Search for X"), observes the result, and repeats until reaching the answer. This simple loop is the foundation of modern AI agents and has become standard practice in production systems.

**Why this matters for interviews:** ReAct defines how modern agents work. Nearly every agent interview question involves ReAct concepts: "How would you design an agent to...?", "What happens when tool use fails?", "How do you handle observations?". Understanding ReAct is essential for agent/systems engineer roles.

---

## Core Contribution

### The Problem: Reasoning Without Action is Limited

Chain-of-Thought improved reasoning, but only for problems that could be solved *in the model's head*. For knowledge-intensive tasks, reasoning alone fails:

- **Question:** "What is the birthplace of the inventor of the mechanical clock?"
- **CoT alone:** Model might reason "The mechanical clock was invented by... someone named John?" and get it wrong
- **With action:** Model reasons "I need to find who invented the mechanical clock", searches the web, observes results, then reasons further

CoT struggles because:
1. Model's knowledge is frozen at training time (outdated)
2. Some information is too specific to memorize
3. Calculations might be wrong (reasoning only, no verification)
4. Complex problems need real data interaction

### The Solution: Interleave Reasoning With Action

Instead of reasoning to a final answer, the model reasons → acts → observes → repeats:

```
Thought: I need to know the birthplace of the mechanical clock's inventor.
Action: Search[inventor of the mechanical clock]
Observation: John Harrison invented the mechanical chronometer in 1735.
Thought: Now I know John Harrison. Let me search for his birthplace.
Action: Search[John Harrison birthplace]
Observation: John Harrison was born in Foulby, Yorkshire, England.
Thought: So the answer is Foulby, Yorkshire, England.
```

### Key Innovation: The ReAct Loop

**Principle:** Agents don't just think OR act—they constantly cycle between thinking and doing.

```
1. State: Current information and history
   ↓
2. Thought: Model generates what to think about next
   ↓
3. Action: Model chooses an action (search, calculate, retrieve, etc.)
   ↓
4. Observation: System executes action, returns result
   ↓
5. Loop back to Step 2 (or terminate if done)
```

This loop enables:
- **Grounding in reality:** Actions tie reasoning to facts
- **Error recovery:** If an observation contradicts reasoning, model can adjust
- **Continuous learning:** Each action provides new information for next thought
- **Task decomposition:** Complex tasks break into thought-action-observation cycles

---

## Key Ideas & Algorithm

### ReAct Loop Structure

**Thought Generation:**
- Model considers: "What do I know? What do I need? What should I do?"
- Outputs a natural language thought explaining the next step

**Action Selection:**
- Model chooses from available actions: Search, Calculate, Lookup, Retrieve, etc.
- Action format: `Action: SearchType[query]` or `Action: Calculate[expression]`
- Actions must be *grounded* (real tools the system actually has)

**Observation Processing:**
- System executes the action
- Returns observation (search results, calculation result, retrieved data)
- Observation becomes part of context for next thought

**Termination Condition:**
- Model decides when to stop iterating
- Outputs final answer with `Final Answer: ...`
- Agent terminates and returns result

### Detailed ReAct Algorithm

| Step | Component | Example |
|------|-----------|---------|
| 1. **Thought** | Model thinks about what to do | "I need to find information about X" |
| 2. **Action** | Model chooses action type + parameters | `Search[X]` or `Calculate[2+3]` |
| 3. **Observation** | System executes, returns result | "Found: X is a..." or "Result: 5" |
| 4. **Context Update** | Add thought+action+observation to history | Becomes input for next thought |
| 5. **Loop/Terminate** | If done, output answer; else go to step 1 | `Final Answer: ...` |

### ReAct vs. CoT vs. Acting Alone

| Aspect | CoT Only | Acting Only | ReAct |
|--------|----------|------------|-------|
| **Reasoning** | Yes | No | Yes |
| **Tool Use** | No | Yes | Yes |
| **Grounded** | No (hallucination risk) | Yes | Yes |
| **Interpretable** | High (see reasoning) | Low (black box) | High |
| **Task Complexity** | Simple reasoning | Simple retrieval | Complex multi-step |
| **Knowledge Currency** | Frozen at training | Always up-to-date | Up-to-date |
| **When to use** | Math, logic puzzles | Lookup tasks | General agents |

### Action Space Design

ReAct agents need well-defined actions:

**Common Actions:**
- `Search[query]` — Web search or knowledge base search
- `Lookup[entity]` — Look up specific entity in database
- `Calculate[expression]` — Simple arithmetic
- `GetInfo[type, id]` — Retrieve structured data
- `ReadDocument[path]` — Read from knowledge base
- `Finish[answer]` — Output final answer

**Action Format:**
- Syntax: `Action: ActionType[parameter]`
- Must be parseable by system
- Clear semantics (system must know what each action does)

**Trade-offs:**
- Too many actions: Model confused, slower
- Too few actions: Can't express all needs
- Sweet spot: 5-10 well-designed actions

---

## Architecture & Trade-offs

### Open-Ended Loops vs. Bounded Loops

**Open-Ended (Unbounded Loop):**
```
while not done:
    thought = model.generate_thought()
    action = model.generate_action()
    observation = execute(action)
    context.append((thought, action, observation))
```

**Pros:**
- Model decides when done
- Can handle unexpected complexity
- Flexible problem solving

**Cons:**
- Risk of infinite loops (model repeats same action forever)
- Unbounded latency (user waiting)
- Potential cost explosion (each step uses tokens)

**Solution:** Set maximum iterations (e.g., 10, 15, 20) and terminate if reached.

### Single Action vs. Multiple Actions Per Thought

**Single Action (Simpler):**
- One thought → one action → one observation per step
- Clear causality
- Easier to debug

**Multiple Actions (More Efficient):**
- One thought → multiple parallel actions → multiple observations
- Faster for independent tasks
- More complex to manage

**Recommendation:** Start with single action; optimize to multiple if needed.

### Tool Design for ReAct

**Good Tool Design:**
- Clear name: `Search[query]` not `DoSomething[input]`
- Informative return: Returns actual data, not "success/fail"
- Handles errors gracefully: Returns "No results found" not raises exception
- Fast: <1 second per call (don't slow down agent)

**Bad Tool Design:**
- Ambiguous: Multiple tools doing similar things
- Silent failures: Returns nothing when error occurs
- Slow: Each call takes 10+ seconds
- Wrong output type: Returns structured data when text expected

### Observation Quality

**High-Quality Observations:**
- Relevant: Contains info needed to answer question
- Concise: Focused on task, not verbose
- Structured: Easy for model to parse
- Error-tolerant: Handles partial/missing info

**Low-Quality Observations:**
- Irrelevant: Doesn't help answer question
- Verbose: 10,000 words when 100 needed
- Unstructured: Requires parsing
- Missing: No error message, just silence

**Insight:** Observation quality directly impacts agent success. Good tool design matters.

### Error Recovery

**Common Failure Modes:**
1. **Invalid action:** Model outputs action that doesn't exist
2. **Malformed input:** Model gives bad parameters to action
3. **Failed action:** Tool returns error
4. **Wrong observation:** Tool returns unexpected format
5. **Hallucination:** Model believes false observation

**Recovery Strategies:**
- **Graceful degradation:** Return `Action failed: reason` and let model retry
- **Fallback actions:** If primary action fails, offer alternatives
- **Validation:** Check action format before executing
- **Repeating:** Allow model to retry with different parameters

---

## Interview Q&A

**Q: Why combine reasoning and acting? Why not just act (tool use) without reasoning?**

A: Pure tool use (acting without reasoning) struggles with complex problems because the model doesn't think about *what* to do before doing it. Without reasoning, a model might search for the wrong thing, misinterpret results, or miss connections. ReAct's interleaving lets the model think "I need X, and X requires finding Y first", then act accordingly. Reasoning guides what actions to take; actions ground reasoning in reality.

**Q: How do you prevent infinite loops in ReAct agents?**

A: Set a maximum number of iterations (e.g., 10 or 15) and force termination. Additionally, monitor for repeated actions—if the agent takes the same action twice in a row, it's likely looping. Some systems use "thought diversity" checks: if the model generates the same thought twice, break the loop and force a decision. In production, measure average iterations per query and alert if it exceeds baseline (e.g., if average is 3 steps and a query takes 20, something is wrong).

**Q: What's the difference between ReAct and Chain-of-Thought? When would you use each?**

A: CoT is for reasoning-only tasks (math, logic, deduction) where the answer is computable from the model's training data. ReAct is for knowledge-intensive or tool-requiring tasks (answering current events, calculations, retrievals) where the model needs external information. CoT runs once; ReAct loops. In practice, modern agents use both: ReAct's loop structure, with CoT *inside* each thought step (model reasons deeply within a thought).

**Q: How do you design the action space for a ReAct agent?**

A: Start with the minimal set of actions needed to solve your task. For a question-answering agent: Search, Lookup, Calculate, Finish. For a planning agent: Create, Update, Delete, Execute, Finish. Avoid too many similar actions (confuses the model). Test on 10-20 example tasks and see if the model tries actions you didn't define—if so, add them. Iterate until the model can express its intent clearly through existing actions.

**Q: What happens if the model hallucinates in a ReAct observation?**

A: If observations are from real tools, the model sees real data and can't hallucinate there. But if the model *misinterprets* the observation or forgets what it learned, it can go wrong. Mitigate by: (1) keeping observations concise and clear, (2) repeating key facts from observations in the next thought, (3) validating observations against model's internal state (if observation contradicts prior learning, flag it).

**Q: How do you optimize ReAct for latency? It seems like the loop could be slow.**

A: Several strategies: (1) Limit iterations (10 max usually sufficient), (2) Use faster models for simple tool calls, (3) Parallelize independent actions, (4) Cache observations (if same action called twice, return cached result), (5) Use "thought shortcuts" (skip reasoning for obvious next steps). In production, measure latency per iteration and optimize slow tools first.

**Q: ReAct seems like it could be expensive (multiple iterations × tokens per iteration). How do you manage cost?**

A: Cost optimization strategies: (1) Use smaller models for agents (once tool outputs are available, a 7B model often works), (2) Cache tool outputs (avoid redundant searches), (3) Limit iterations (max 10-15), (4) Use streaming output (show results to user while agent is still thinking), (5) Batch similar queries (one agent call answers multiple sub-questions). In practice, a ReAct agent costs ~1-2 API calls for simple queries, ~5-10 for complex ones.

---

## Best Practices

- **Design actions first, before building agent:** List what the agent needs to do (search, lookup, calculate, etc.). That's your action space. Don't build agents first and retrofit actions.

- **Test each action independently:** Before putting actions in an agent, test them. Can the agent call the action? Does it get the right output? Does error handling work?

- **Keep observations concise:** Long observations confuse models and waste tokens. Summarize search results to top 3, truncate long text. Agents work better with *focused* information.

- **Include reasoning in prompts:** "Think step by step before using tools" helps. Example: "You have access to: Search, Lookup, Calculate. Think about what you need, then use tools to get it."

- **Monitor for infinite loops:** Log iterations per query. If a query takes 10+ iterations when average is 2, something's wrong. Add automatic loop detection.

- **Use explicit termination:** `Final Answer: ...` or `Finish[answer]` makes it clear when the agent is done. Don't rely on implicit stopping.

- **Validate action output:** Even if tools are reliable, validate that action outputs match expected format. If validation fails, return helpful error message to model.

- **Start simple, add complexity:** Begin with 5 actions. If the agent struggles to express needs, add more. But don't start with 20 actions.

---

## Common Pitfalls

- **Mistake: Unbounded loops without iteration limit.** Agent repeats same search forever because it doesn't understand the observation. Always set `max_iterations = 10-15` and force termination.

- **Mistake: Noisy or verbose observations.** Returning all 100 search results to the model wastes tokens and confuses it. Filter to top 5-10 most relevant results per tool call.

- **Mistake: Tool returning success/fail instead of data.** `SearchResult: True` is useless. Tools should return `SearchResult: "Found: ... article about ..."` with actual information.

- **Mistake: No error handling in tool design.** If a tool fails, it should return `Error: reason` not raise an exception. Let the agent see the error and decide what to do.

- **Mistake: Model hallucinates actions that don't exist.** Model outputs `Action: TranslateToLatin[text]` but that action doesn't exist. Solution: validate actions before executing; if invalid, return "Action not recognized. Available: Search, Lookup, Calculate."

- **Mistake: Observations don't match query context.** If the agent searched for "2024 Olympics" and gets results about "1984 Olympics", the mismatch will confuse it. Validate that observations are relevant to the query.

- **Mistake: Using ReAct for simple tasks.** Not every task needs an agent. For simple questions that the model can answer directly, use CoT. ReAct adds latency and cost. Use it only when tools are actually needed.

---

## Code Examples

### Example 1: Simple ReAct Loop

```python
import re

class ReActAgent:
    """Basic ReAct agent with search and calculate actions."""
    
    def __init__(self, model_api, tools, max_iterations=10):
        self.model = model_api
        self.tools = tools  # {"search": func, "calculate": func, ...}
        self.max_iterations = max_iterations
    
    def run(self, question: str) -> str:
        """Run ReAct loop until answer found."""
        context = f"Question: {question}\n\n"
        
        for iteration in range(self.max_iterations):
            # Step 1: Generate thought
            prompt = context + "Thought:"
            thought = self.model.generate(prompt)
            context += f"Thought: {thought}\n"
            
            # Step 2: Parse and execute action
            action_str = self.model.generate(context + "Action:")
            action, param = self._parse_action(action_str)
            
            if action == "finish":
                return param  # Return final answer
            
            context += f"Action: {action_str}\n"
            
            # Step 3: Get observation from tool
            if action in self.tools:
                observation = self.tools[action](param)
            else:
                observation = f"Error: Action '{action}' not found"
            
            context += f"Observation: {observation}\n\n"
        
        return "Max iterations reached"
    
    def _parse_action(self, action_str: str) -> tuple:
        """Parse 'Action: SearchType[query]' into (action_type, param)."""
        match = re.match(r'(\w+)\[(.+?)\]', action_str.strip())
        if match:
            return match.group(1).lower(), match.group(2)
        return "error", "Invalid action format"

# Example tools (simulated)
def search(query):
    return f"Found: Articles about {query}"

def calculate(expr):
    try:
        return f"Result: {eval(expr)}"
    except:
        return "Error: Invalid expression"

tools = {"search": search, "calculate": calculate}

# Usage
agent = ReActAgent(model_api=None, tools=tools)
# answer = agent.run("What is the capital of France?")
```

### Example 2: ReAct with Multiple Actions

```python
class MultiActionReAct:
    """ReAct agent that can take multiple actions per thought."""
    
    def __init__(self, model_api, tools, max_iterations=10):
        self.model = model_api
        self.tools = tools
        self.max_iterations = max_iterations
    
    def run(self, question: str) -> str:
        """Run ReAct loop with multi-action support."""
        context = f"Question: {question}\n"
        history = []
        
        for iteration in range(self.max_iterations):
            # Generate thought
            thought = self.model.generate(context + "Thought: ")
            context += f"Thought: {thought}\n"
            
            # Generate multiple actions (until model says "Done")
            actions = []
            while True:
                action_str = self.model.generate(context + "Action: ")
                if "done" in action_str.lower() or "finish" in action_str.lower():
                    final_answer = action_str.replace("Done:", "").replace("Finish:", "").strip()
                    return final_answer
                
                action_type, param = self._parse_action(action_str)
                actions.append((action_type, param))
                
                if len(actions) >= 3:  # Max 3 actions per thought
                    break
            
            # Execute all actions in parallel (or sequentially)
            observations = []
            for action_type, param in actions:
                if action_type in self.tools:
                    obs = self.tools[action_type](param)
                else:
                    obs = f"Error: Unknown action {action_type}"
                observations.append(obs)
                context += f"Action: {action_type}[{param}]\nObservation: {obs}\n"
            
            context += "\n"
            history.append((thought, actions, observations))
        
        return "Max iterations reached"
    
    def _parse_action(self, action_str: str) -> tuple:
        import re
        match = re.match(r'(\w+)\[(.+?)\]', action_str.strip())
        if match:
            return match.group(1).lower(), match.group(2)
        return "error", "Invalid action"
```

### Example 3: Error Handling in ReAct

```python
class RobustReAct:
    """ReAct with error recovery."""
    
    def __init__(self, model_api, tools, max_iterations=10):
        self.model = model_api
        self.tools = tools
        self.max_iterations = max_iterations
    
    def run(self, question: str) -> str:
        context = f"Question: {question}\n"
        failed_actions = set()
        
        for iteration in range(self.max_iterations):
            thought = self.model.generate(context + "Thought: ")
            context += f"Thought: {thought}\n"
            
            action_str = self.model.generate(context + "Action: ")
            action_type, param = self._parse_action(action_str)
            
            # Validate action
            if action_type not in self.tools and action_type != "finish":
                context += f"Action: {action_str}\nObservation: Error: Unknown action. Available: {', '.join(self.tools.keys())}\n\n"
                continue
            
            if action_type == "finish":
                return param
            
            # Execute action with error handling
            try:
                if (action_type, param) in failed_actions:
                    observation = f"Note: {action_type}[{param}] already failed. Try a different approach."
                else:
                    observation = self.tools[action_type](param)
                    if "error" in observation.lower():
                        failed_actions.add((action_type, param))
            except Exception as e:
                observation = f"Error executing {action_type}: {str(e)}"
                failed_actions.add((action_type, param))
            
            context += f"Action: {action_str}\nObservation: {observation}\n\n"
        
        return "Max iterations reached"
    
    def _parse_action(self, action_str: str) -> tuple:
        import re
        if "finish" in action_str.lower():
            return "finish", action_str
        match = re.match(r'(\w+)\[(.+?)\]', action_str.strip())
        if match:
            return match.group(1).lower(), match.group(2)
        return "error", action_str
```

---

## Related Concepts

- [Chain-of-Thought Prompting](./01-chain-of-thought.md) — Reasoning component of ReAct
- [Tree of Thoughts](./03-tree-of-thoughts.md) — Explore multiple action sequences, not just greedy
- [agentic-ai/concepts/XX-tool-use](../../agentic-ai/concepts/XX-tool-use.md) — Action/tool design in agents
- [agentic-ai/concepts/XX-agent-loops](../../agentic-ai/concepts/XX-agent-loops.md) — General agent architecture

---

**Last Updated:** 2026-05-31
