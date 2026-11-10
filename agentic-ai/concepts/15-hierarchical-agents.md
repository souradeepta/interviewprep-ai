# Hierarchical Agents

TL;DR: Hierarchical Agents is a design pattern for making an agent more reliable by controlling how it hierarchical agents under explicit constraints.

## Prerequisites

This lesson assumes familiarity with agentic-ai/multi-agent-systems. You should be able to describe an LLM request/response cycle, inspect inputs and outputs, and reason about the application boundary where this technique runs. The lesson focuses on hierarchical agents as an engineering choice: identify the problem it solves, the resources it consumes, and the evidence needed to know whether it works.

## Learning Outcomes

By the end, you can:
- Explain the mechanism of hierarchical agents and name its key assumptions.
- Implement a minimal version and trace one successful and one failing execution.
- Compare it with a simpler alternative using quality, latency, cost, and safety measures.
- Diagnose a failure from observable evidence and propose a bounded recovery.

## Core Intuition

Treat hierarchical agents as a controlled decision in a larger system rather than a magic capability. State the goal, allowed inputs, permitted actions, and stopping condition first. Then trace how information changes at each step. This makes the design reviewable: a teammate can inspect what evidence led to a decision, what the system did, and whether the result met the task contract. The central engineering question is whether the extra mechanism improves outcomes enough to justify its additional complexity and operational cost.

## How It Works

A practical design starts with a task boundary and an observable success criterion. The orchestrator collects the relevant state, selects the next operation, validates its inputs, and records the result. It then checks whether the goal is complete or whether another bounded step is justified. Keep the state passed between steps explicit and small; include only data that affects the decision. Separate policy from execution so a decision can be inspected before it causes an external effect. When possible, make operations idempotent and attach a stable identifier so retries do not duplicate work.

For hierarchical agents, the important implementation detail is to make the decision point testable. Define the expected input shape, valid output shape, allowed resource budget, timeout, and fallback behavior. Record a trace containing the request identifier, chosen operation, outcome category, and duration, while excluding secrets and unnecessary personal data. A good trace supports diagnosis without requiring access to hidden model reasoning. Use representative examples and counterexamples to measure success, then inspect disagreements and boundary cases before increasing autonomy.

## Trade-offs

| Approach | Strength | Cost or risk | Choose when |
|---|---|---|---|
| Direct, single-step response | Lowest latency and simplest control path | Cannot adapt after new evidence | The task is bounded and context already contains needed facts |
| Hierarchical Agents | Can improve task completion by making the relevant process explicit | More state, integration points, and failure paths to operate | Evaluation shows a repeatable gain for the target workload |
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
| What problem does hierarchical agents solve? | It addresses a stated task limitation and should be justified by measured end-to-end improvement. | What baseline would you compare against? |
| How would you keep it reliable? | Validate boundaries, bound resources, log useful traces, and define recovery behavior. | How would you debug a regression? |
| When would you avoid it? | When a simpler method meets the quality target or added complexity exceeds the value. | Which metric changes that decision? |

## Related Concepts

This lesson connects to agentic-ai/multi-agent-systems. It also informs the downstream topics that consume its outputs: schema validation, observability, evaluation, and safe recovery. When designing a system, map these dependencies explicitly so a failure can be localized to the decision, execution, or verification stage.

## References

- [OpenAI API documentation](https://platform.openai.com/docs/guides/function-calling)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

## Detailed Explanation

Hierarchical agents organize multi-agent systems into layers where higher-level agents delegate tasks to lower-level specialists. A manager agent receives a user goal and decomposes it into subtasks, assigning each to domain-expert workers. Workers solve their specialized subtask independently, then return results to the manager, who aggregates and synthesizes them into a cohesive response. This mirrors organizational structures: CEO sets strategy, directors run departments, managers supervise teams. Hierarchical organization excels at complex problems requiring different expertise (data analyst, engineer, product manager) but introduces coordination overhead and potential bottlenecks at manager levels. The key advantage: specialization—each worker becomes expert in a narrow domain, producing higher quality than generalists. The key disadvantage: communication latency and manager becoming bottleneck if poorly designed. When to use: complex tasks with clear decomposition, multiple specialized knowledge domains, need for quality per subtask. When not to: simple tasks (overhead not worth it), tasks requiring tight coordination, unpredictable task dependencies.

## Core Intuition

Imagine building a product: CEO plans direction, engineering manager assigns features to engineers, each engineer builds their module independently, manager integrates. Better than one person doing everything (specialist knowledge) but slower than flat team (coordination overhead). Hierarchical agents work the same way: manager breaks problem into specialized subtasks, each worker solves their part expertly, manager coordinates the solution. Trade-off: quality vs speed.

## How It Works

Hierarchical agents operate through task decomposition, delegation, and aggregation:

1. **Manager Receives Goal** — User submits complex task
2. **Task Decomposition** — Manager analyzes goal, identifies subtasks and required expertise
3. **Delegation** — Manager assigns subtasks to appropriate workers
4. **Parallel Execution** — Workers solve their subtasks independently and in parallel
5. **Result Collection** — Manager gathers results from all workers
6. **Aggregation** — Manager synthesizes results, resolves conflicts, produces final response
7. **Feedback Loop** — If aggregation reveals new subtasks or conflicts, manager re-delegates

```mermaid
graph TD
    A["User Goal"] --> B["Manager: Decompose"]
    B --> C["Identify Subtasks"]
    C --> D["Identify Required Expertise"]
    D --> E["Match to Workers"]
    E --> F["Delegate"]
    F --> G["Worker 1<br/>Specialist A"]
    F --> H["Worker 2<br/>Specialist B"]
    F --> I["Worker 3<br/>Specialist C"]
    G --> J["Task A Result"]
    H --> K["Task B Result"]
    I --> L["Task C Result"]
    J --> M["Manager: Aggregate"]
    K --> M
    L --> M
    M --> N{"Conflicts?"}
    N -->|Yes| B
    N -->|No| O["Final Response"]
```

## Architecture / Trade-offs

**Hierarchy Depth:**
- Shallow (1-2 levels) — Fast, low communication overhead, limited specialization
- Deep (3+ levels) — More specialization, higher latency, more complex coordination

**Delegation Strategy:**
- Function-based — Each worker handles type of task (analyst, engineer, manager)
- Domain-based — Each worker knows specific domain (databases, caching, ML)
- Skill-based — Each worker has specific capability (prompt optimization, tool calling)

**Manager Role:**
- Heavy coordination — Manager breaks down task, assigns, monitors, aggregates (tight control, slower)
- Light coordination — Manager only routes; workers self-organize (faster, potential inconsistency)

**Scaling:**
- Vertical — Add more hierarchy levels for bigger problems (increases latency)
- Horizontal — Add workers at same level (flat scalability, manager becomes bottleneck)

## Interview Q&A

**Q: When should you use hierarchical agents instead of a single agent?**
A: Use hierarchy when: (1) task needs multiple expert domains, (2) specialization improves quality, (3) subtasks can be solved independently. Don't use if: simple task (overhead not worth it), tight coupling between subtasks, or unpredictable dependencies. Measure: does specialization + decomposition save tokens/time vs single agent's generalist approach?

**Q: What's the bottleneck in hierarchical systems?**
A: The manager becomes bottleneck if it: (1) takes too long to decompose, (2) has limited bandwidth (can only manage N workers), (3) becomes sequential (waits for each worker instead of parallel dispatch). Solutions: (1) pre-compute decompositions, (2) limit manager span (max 5-7 direct workers), (3) parallelize worker execution with asyncio, (4) use peer-to-peer workers for some tasks.

**Q: How do you handle conflicts in aggregation?**
A: Conflicts occur when workers disagree or produce incompatible results. Solutions: (1) Define aggregation rules upfront (majority vote, trust highest-confidence worker), (2) Have manager re-query conflicting workers for explanations, (3) Escalate to human, (4) Use voting/weighted average if tasks allow. Example: two analysts disagree on recommendation. Manager asks each to defend their view, chooses based on evidence strength.

**Q: How do you choose which worker gets which subtask?**
A: Match subtask type to worker specialization. Use explicit routing: if subtask is "data analysis" → send to analyst worker; if "code generation" → send to engineer. Can use: (1) keyword matching (look for "SQL", "query" in task), (2) ML classifier (trained on task-to-worker mappings), (3) Explicit function routing (Python function maps task type to worker), (4) Worker self-nomination (workers bid on tasks). Rule: make routing deterministic and fast; don't spend 10s routing a 1s task.

**Q: What if a worker fails or times out?**
A: Strategies: (1) Retry the worker (temporary failure), (2) Route to backup worker with different specialization, (3) Have manager solve directly as fallback, (4) Escalate to user. Example: engineer worker timeout (too complex task). Manager can: escalate to senior engineer, break task smaller for junior engineer, or handle directly. Track failure rates per worker to detect systematic issues.

**Q: How many hierarchy levels should you have?**
A: Rule of thumb: 2-3 levels max. Each level adds latency. Examples: (1) Single manager + workers = 1 level (good for most problems), (2) CEO + directors + managers = 2 levels (for very complex systems), (3) 3+ levels = rare, only for massive organizational structures. If you have 4+ levels, reconsider: are subtasks really that complex or are you over-engineering?

**Q: How to prevent a worker from becoming a bottleneck?**
A: Workers become bottlenecks if: (1) sequential processing (handle one task at a time), (2) complex internal computation (slow solver), (3) resource-constrained (GPU worker, limited quota). Solutions: (1) Parallelize internally (batch requests), (2) Pre-compute results (caching), (3) Simplify task routing (don't route complex tasks to slow worker), (4) Add worker replicas (multiple workers per specialty).

## Best Practices

1. **Design Clear Specialization** — Each worker expert in narrow domain (analyst, engineer, strategist). Don't make workers "do everything"; specialization is the point.

2. **Explicit Routing** — Use deterministic routing rules. "If task contains keyword 'database', route to DB specialist." Avoid guessing or expensive ML classifiers for simple routing.

3. **Parallel Dispatch** — Don't wait for one worker to finish before sending next task. Use asyncio.gather() to parallelize. Manager sends all subtasks simultaneously.

4. **Set Timeouts** — Each worker should have timeout (e.g., max 30s per subtask). If timeout, manager routes to fallback or escalates.

5. **Define Aggregation Rules Upfront** — How does manager combine results? Majority vote? Weighted by confidence? Concatenation? Define before workers run; adapting mid-aggregation is error-prone.

6. **Monitor Manager Load** — Track time spent in manager (decomposing, aggregating). If >50% of total latency is manager, you need better parallelization or faster decomposition.

7. **Limit Manager Span** — Max 5-7 direct workers per manager. Above that, add intermediate managers (go deeper). Wide shallow trees create communication overhead.

8. **Cache Decompositions** — If same goal asked repeatedly, reuse decomposition. Don't re-solve "how do I decompose X?" every time.

9. **Fallback Patterns** — Always have fallback: if workers fail, manager can (1) re-try, (2) route to different worker, (3) solve directly, (4) escalate. Design this upfront.

10. **Log Full Hierarchy** — For debugging, log which manager asked which worker to do what, when, with what result. Trace the whole tree.

## Common Pitfalls

**Pitfall 1: Unclear Subtask Definitions**
Issue: Manager breaks task into vague subtasks. Workers confused about scope, overlap, missing pieces.
Fix: Be explicit. "Analyze user retention data for cohort aged 18-25 from last 90 days" not "analyze retention data." Include constraints.

**Pitfall 2: No Parallelization**
Issue: Manager sends task to Worker 1, waits, gets result, then sends to Worker 2. Sequential defeats the purpose.
Fix: Dispatch all subtasks in parallel using asyncio.gather(). Manager should wait for all workers at once.

**Pitfall 3: Manager Becomes Bottleneck**
Issue: Manager is slow (spends 20s decomposing, 10s aggregating). Total latency dominated by manager overhead.
Fix: Pre-compute decompositions. Simplify aggregation. If manager is bottleneck, you're not scaling; reconsider if hierarchy is needed.

**Pitfall 4: Workers Don't Specialize**
Issue: Workers are generalists, each could do any task. Defeats the specialization benefit.
Fix: Design workers to be domain experts. Analyst knows SQL, data visualization. Engineer knows Python, system design. Strategist knows trade-offs, cost analysis.

**Pitfall 5: No Handling of Worker Failure**
Issue: Worker times out or fails. No fallback. Manager hangs.
Fix: Always have timeouts. Always have fallback (retry, different worker, escalate). Design fault tolerance upfront.

**Pitfall 6: Over-Complex Hierarchy**
Issue: 4+ levels of hierarchy. Each level adds latency. Total latency becomes unacceptable.
Fix: Limit to 2-3 levels. If task is complex, simplify decomposition or add workers at same level instead of going deeper.

**Pitfall 7: Aggregation Conflicts Unresolved**
Issue: Workers disagree. No clear rule for which result to trust. Manager unsure how to combine.
Fix: Define aggregation rules upfront. "Majority vote", "Trust highest-confidence", "Merge with human review". Don't decide during aggregation.

## Code Examples

### Example 1: Basic Hierarchical Agent

```python
import asyncio
from typing import List, Dict

class SpecialistWorker:
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
    
    async def solve(self, task: str) -> Dict[str, str]:
        """Solve task in specialist domain."""
        await asyncio.sleep(0.5)  # Simulate work
        return {
            "worker": self.name,
            "specialty": self.specialty,
            "result": f"Solved '{task}' using {self.specialty}"
        }

class ManagerAgent:
    def __init__(self, workers: List[SpecialistWorker]):
        self.workers = {w.specialty: w for w in workers}
        self.task_assignments = {
            "data": "data",
            "analysis": "data",
            "code": "engineer",
            "implement": "engineer",
            "strategy": "strategist",
            "trade-off": "strategist"
        }
    
    def decompose_task(self, goal: str) -> List[Dict]:
        """Break goal into subtasks and assign to workers."""
        subtasks = []
        keywords = goal.lower().split()
        
        # Simple keyword-based routing
        specialties_needed = set()
        for kw in keywords:
            if kw in self.task_assignments:
                specialties_needed.add(self.task_assignments[kw])
        
        # If no clear specialties, use all
        if not specialties_needed:
            specialties_needed = set(self.workers.keys())
        
        for i, specialty in enumerate(specialties_needed):
            subtasks.append({
                "id": i,
                "specialty": specialty,
                "task": f"{goal} (aspect: {specialty})"
            })
        
        return subtasks
    
    async def delegate_and_aggregate(self, goal: str) -> str:
        """Decompose, delegate to workers, aggregate results."""
        # Decompose
        subtasks = self.decompose_task(goal)
        print(f"Manager decomposed into {len(subtasks)} subtasks")
        
        # Delegate and execute in parallel
        results = []
        tasks = []
        for subtask in subtasks:
            worker = self.workers[subtask["specialty"]]
            tasks.append(worker.solve(subtask["task"]))
        
        results = await asyncio.gather(*tasks)
        
        # Aggregate
        final = f"Goal: {goal}\n"
        for r in results:
            final += f"  {r['worker']}: {r['result']}\n"
        
        return final

# Usage
async def main():
    workers = [
        SpecialistWorker("Analyst", "data"),
        SpecialistWorker("Engineer", "engineer"),
        SpecialistWorker("Strategist", "strategist")
    ]
    manager = ManagerAgent(workers)
    
    result = await manager.delegate_and_aggregate(
        "Implement data analysis and strategy for recommendation system"
    )
    print(result)

asyncio.run(main())
```

### Example 2: Manager Agent with Fault Tolerance

```python
import asyncio
from typing import Optional

class FaultTolerantManager:
    def __init__(self, workers: Dict, max_retries: int = 2, timeout: int = 5):
        self.workers = workers
        self.max_retries = max_retries
        self.timeout = timeout
    
    async def call_worker_with_retry(self, specialty: str, task: str) -> Optional[Dict]:
        """Call worker with retry and timeout handling."""
        for attempt in range(self.max_retries):
            try:
                worker = self.workers[specialty]
                result = await asyncio.wait_for(
                    worker.solve(task),
                    timeout=self.timeout
                )
                return result
            except asyncio.TimeoutError:
                print(f"⏱️  {specialty} worker timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
            except Exception as e:
                print(f"❌ {specialty} worker error: {e}")
                return {"worker": specialty, "result": "FAILED", "error": str(e)}
        
        return None  # All retries exhausted
    
    async def delegate_with_fallback(self, goal: str) -> str:
        """Delegate with fallback handling."""
        subtasks = self.decompose_task(goal)
        
        results = []
        tasks = []
        for subtask in subtasks:
            tasks.append(
                self.call_worker_with_retry(subtask["specialty"], subtask["task"])
            )
        
        results = await asyncio.gather(*tasks)
        
        # Check for failures
        failed = [r for r in results if r is None or r.get("result") == "FAILED"]
        if failed:
            print(f"⚠️  {len(failed)} workers failed, using fallback")
            # Fallback: manager solves directly or escalates
            return "Escalated to human review"
        
        return "\n".join([f"{r['worker']}: {r['result']}" for r in results])
    
    def decompose_task(self, goal: str) -> List[Dict]:
        # Same as before
        return [{"specialty": "data", "task": goal}]
```

### Example 3: Hierarchical Cascade (Multi-Level)

```python
class HierarchicalCascade:
    def __init__(self):
        self.top_manager = ManagerAgent(name="CEO")
        self.departments = {
            "engineering": ManagerAgent(name="Engineering Manager"),
            "data": ManagerAgent(name="Data Manager"),
            "product": ManagerAgent(name="Product Manager")
        }
    
    async def decompose_and_delegate_recursive(self, goal: str, level: int = 0) -> str:
        """Multi-level hierarchy: CEO delegates to department managers."""
        indent = "  " * level
        print(f"{indent}→ Level {level} Manager: decomposing '{goal}'")
        
        # Top-level decomposition
        if level == 0:
            # CEO breaks into department tasks
            department_tasks = {
                "engineering": f"{goal} (engineering perspective)",
                "data": f"{goal} (data perspective)",
                "product": f"{goal} (product perspective)"
            }
        else:
            # Department manager breaks into specialist tasks
            department_tasks = {k: goal for k in self.departments.keys()}
        
        # Parallel delegation to next level
        results = []
        tasks = []
        for dept, task in department_tasks.items():
            if level == 0:
                # Delegate to department managers
                mgr = self.departments.get(dept)
                if mgr:
                    tasks.append(mgr.delegate_and_aggregate(task))
        
        if tasks:
            results = await asyncio.gather(*tasks)
        
        return "\n".join(results)

# Usage
async def main():
    cascade = HierarchicalCascade()
    result = await cascade.decompose_and_delegate_recursive(
        "Build recommendation system"
    )
    print(result)

asyncio.run(main())
```

## Related Concepts

- **Multi-Agent Systems** — Broader patterns beyond hierarchy
- **Cooperative Agents** — Peer-to-peer alternative to hierarchy
- **Skill Composition** — Combining skills hierarchically
- **Agent Loops** — Iteration patterns within hierarchies
- **Error Recovery** — Handling failures in delegated tasks

