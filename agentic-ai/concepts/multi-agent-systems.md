# Multi-Agent Systems

## TL;DR
Multiple specialized agents working together to solve complex problems. Agents communicate, delegate tasks, and coordinate to achieve shared goals. Patterns: hierarchical (manager-worker), cooperative (peers), competitive (racing), debate (arguing sides). Enables scaling complex reasoning and division of labor.

## Core Intuition
One agent may not be expert in everything. Instead, hire specialists: analyst agent (good at data), lawyer agent (understands contracts), engineer agent (codes). They talk to each other, delegate, and together solve the problem better than one generalist.

## How It Works

**Agent Roles and Specialization:**
```
Task: "Design a mobile app"

Single agent struggles (not expert in design, backend, frontend, UX)

Multi-agent:
  - Product Manager: clarifies requirements, owns roadmap
  - Designer: creates wireframes, UI/UX
  - Backend Engineer: designs API, database
  - Frontend Engineer: implements interface
  
Each handles expertise, communicates progress.
```

**Coordination Patterns:**

**1. Hierarchical (Manager-Worker):**
```
      Manager
       /    \
    Worker1  Worker2  Worker3

Manager:
  - Receives task
  - Breaks into subtasks
  - Delegates to workers
  - Aggregates results
  - Reports to user

Workers:
  - Execute assigned subtask
  - Report results to manager
```

**2. Cooperative (Peer-to-Peer):**
```
Agent1 ←→ Agent2
  ↓  ↘ ↙  ↓
Agent3 ←→ Agent4

All agents know each other, communicate directly.
Can request help, share observations.
No central authority.
```

**3. Competitive (Multi-attempt):**
```
Agent1: Try approach A → Result A
Agent2: Try approach B → Result B
Agent3: Try approach C → Result C

Evaluator: Pick best result or combine
(Useful for brainstorming, problem-solving)
```

**4. Debate (Adversarial):**
```
Thesis Agent: Argues "X is true" with reasoning
Antithesis Agent: Argues "X is false" with counter-reasoning
Judge Agent: Evaluates both sides, determines winner

Iterates: each side responds to counterarguments
Final: Judge declares winner or consensus
```

**Communication and Coordination:**
```
Shared state / blackboard:
  - All agents can read/write task info
  - Agents post observations, results, requests
  - Others read and respond asynchronously

Message passing:
  - Agent A sends message to Agent B
  - B responds when ready
  - Formal protocol, explicit communication

Synchronous vs async:
  - Sync: all agents wait for slowest (reliable, slow)
  - Async: agents proceed independently (fast, coordination harder)
```

## Key Properties / Trade-offs

| Aspect | Single Agent | Multi-Agent |
|--------|---|---|
| Specialization | Generalist (lower quality) | Expert specialists (higher quality) |
| Cost | 1 agent call | N agent calls (higher cost) |
| Latency | Fast | Slower (coordination overhead) |
| Complexity | Simple | Complex (coordination needed) |
| Robustness | Single point of failure | Redundancy, can reassign |
| Reasoning | Limited to one view | Multiple perspectives, debate |

**Pattern Selection:**
- **Hierarchical:** Clear task decomposition, worker independent (data analysis, report writing)
- **Cooperative:** Agents need to consult each other (design team, brainstorming)
- **Competitive:** Explore multiple approaches, pick best (voting, ensemble)
- **Debate:** Controversial, complex decisions (pros/cons analysis, strategic planning)

## Common Mistakes / Gotchas

- **Too many agents:** N agents = O(N²) communication overhead. Keep to 3-10 for typical tasks.
- **Unclear role division:** Overlapping specialization → confusion, duplicate work. Define roles clearly.
- **Poor communication protocol:** Agents talk past each other, miss updates. Use explicit messaging format.
- **Synchronization bottleneck:** Waiting for all agents before proceeding → latency. Use async communication.
- **Deadlock:** Agent A waits for B, B waits for C, C waits for A. Add timeouts, priority resolution.
- **Hallucinating collaboration:** One agent claims it asked another; it didn't. Use message logs for proof.
- **No fallback:** If one agent fails, whole system blocks. Add retry logic, alternative agents.
- **Evaluation ambiguity:** Debate agents argue forever. Set max rounds, use clear judgment criteria.

## Code Example

```python
from anthropic import Anthropic

client = Anthropic()

def multi_agent_system(task):
    """Multi-agent system with manager and specialist workers."""
    
    # Define agents with role/expertise
    agents = {
        "manager": {
            "role": "Project manager, coordinates subtasks",
            "system": "You are a project manager. Break the task into clear subtasks and delegate to specialists."
        },
        "analyst": {
            "role": "Data analyst, interprets information",
            "system": "You are a data analyst. Focus on data, metrics, and factual analysis."
        },
        "strategist": {
            "role": "Strategist, recommends decisions",
            "system": "You are a strategist. Focus on decisions, trade-offs, and long-term impact."
        }
    }
    
    # Shared blackboard (all agents can read/write)
    blackboard = {
        "task": task,
        "subtasks": [],
        "analysis": {},
        "recommendations": {},
        "status": "started"
    }
    
    # Manager breaks down task
    print("=== Manager Planning ===")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=agents["manager"]["system"],
        messages=[{"role": "user", "content": f"Task: {task}\n\nBreak this into 3-4 subtasks for specialists."}]
    )
    plan = response.content[0].text
    print(plan)
    blackboard["subtasks"] = plan
    
    # Analyst handles data subtask
    print("\n=== Analyst Analysis ===")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=agents["analyst"]["system"],
        messages=[{"role": "user", "content": f"Task: {task}\n\nProvide factual analysis and key metrics."}]
    )
    analysis = response.content[0].text
    print(analysis)
    blackboard["analysis"] = analysis
    
    # Strategist provides recommendations
    print("\n=== Strategist Recommendations ===")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=agents["strategist"]["system"],
        messages=[{
            "role": "user",
            "content": f"Task: {task}\n\nGiven the analysis above, provide strategic recommendations.\n\nAnalysis:\n{analysis}"
        }]
    )
    recommendations = response.content[0].text
    print(recommendations)
    blackboard["recommendations"] = recommendations
    
    # Manager synthesizes results
    print("\n=== Manager Summary ===")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=agents["manager"]["system"],
        messages=[{
            "role": "user",
            "content": f"""Task: {task}

Specialists provided:
Analysis: {analysis}
Recommendations: {recommendations}

Synthesize into final plan."""
        }]
    )
    final_plan = response.content[0].text
    print(final_plan)
    blackboard["status"] = "completed"
    
    return blackboard

# Example
task = "Design a marketing strategy for a new AI product targeting enterprises"
result = multi_agent_system(task)
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "What is multi-agent system?" | Multiple specialized agents working together. Each expert in domain, communicate to solve complex problems. |
| "When use multi-agent?" | Complex tasks benefiting from specialization (design team, analysis + strategy). Simple tasks: overkill. |
| "Hierarchical vs cooperative?" | Hierarchical: clear manager (good for task decomposition). Cooperative: peer (good for brainstorming, collaboration). |
| "Coordination cost?" | N agents = O(N²) communication overhead. Usually 3-10 agents optimal; more than that, diminishing returns. |
| "Failure handling?" | One agent fails → reassign or retry. Use timeouts, fallbacks. Log all communications for debugging. |
| "Evaluation?" | Compare multi-agent result to single agent. Multi-agent should be better (specialization) and/or faster (parallelism). |

## Related Topics
- [Hierarchical Agents](hierarchical-agents.md) — manager-worker pattern
- [Cooperative Agents](cooperative-agents.md) — peer-to-peer agents
- [Competitive Agents](competitive-agents.md) — racing/voting approaches
- [Agent Communication](agent-communication.md) — protocols and message passing
- [Agent Loops](agent-loops.md) — individual agent's decision loop

## Resources
- [Multi-Agent Systems: A Modern Approach to Distributed Artificial Intelligence](https://mitpress.mit.edu/9780262133616/)
- [Agents as Reasoning Engines (OpenAI)](https://openai.com/research/agents-as-reasoning-engines)
- [LangGraph: Multi-Agent Orchestration](https://python.langchain.com/docs/langgraph/)
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
