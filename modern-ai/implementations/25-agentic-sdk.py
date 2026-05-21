"""
Auto-generated from 25-agentic-sdk.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agentic SDK / Agent Framework
# ## Learning Objectives
# 1. Build minimal agent loop (observe-think-act) with tool registry
# 2. Implement state management and error recovery
# 3. Support task decomposition and goal tracking
# 4. Test multi-step reasoning and tool usage patterns
# ======================================================================

# Prerequisites & Imports
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict

print("Agentic SDK Implementation")
print(f"Building minimal agent framework with tool use")



# ======================================================================
# ## Level 1: Basic Agent Loop (Observe-Think-Act)
# ======================================================================

# Level 1: Minimal Agent Loop

class AgentState(Enum):
    IDLE = 'idle'
    THINKING = 'thinking'
    ACTING = 'acting'
    DONE = 'done'

@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    input_schema: Dict = field(default_factory=dict)

class BasicAgent:
    """Minimal agent with observe-think-act loop."""
    
    def __init__(self, name: str = 'Agent'):
        self.name = name
        self.state = AgentState.IDLE
        self.tools: Dict[str, Tool] = {}
        self.memory = []  # Action history
        self.goal = None
    
    def register_tool(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
        print(f"✓ Registered tool: {tool.name}")
    
    def observe(self, context: Dict) -> Dict:
        """Observe current state."""
        self.state = AgentState.THINKING
        print(f"[OBSERVE] Goal: {self.goal}")
        print(f"[OBSERVE] Context: {list(context.keys())}")
        return context
    
    def think(self, context: Dict) -> str:
        """Think about next action."""
        # Simple: check what tools are available
        tools_str = ", ".join(self.tools.keys())
        decision = f"I have tools: {tools_str}. I should use {list(self.tools.keys())[0] if self.tools else 'none'}"
        print(f"[THINK] {decision}")
        return decision
    
    async def act(self, tool_name: str, args: Dict) -> Dict:
        """Execute chosen tool."""
        self.state = AgentState.ACTING
        
        if tool_name not in self.tools:
            return {'error': f'Tool {tool_name} not found'}
        
        tool = self.tools[tool_name]
        try:
            result = await tool.func(**args) if asyncio.iscoroutinefunction(tool.func) else tool.func(**args)
            self.memory.append({'tool': tool_name, 'args': args, 'result': result})
            print(f"[ACT] {tool_name}({list(args.keys())}) → {result}")
            return result
        except Exception as e:
            print(f"[ERROR] {tool_name}: {str(e)[:50]}")
            return {'error': str(e)}
    
    async def run_once(self, goal: str, context: Dict):
        """Run one iteration of OTA loop."""
        self.goal = goal
        
        # Observe
        observed = self.observe(context)
        
        # Think
        decision = self.think(observed)
        
        # Act
        if self.tools:
            first_tool = list(self.tools.keys())[0]
            await self.act(first_tool, context)
        
        self.state = AgentState.DONE

# Test Level 1
print("Testing basic agent loop...\n")

# Define simple tools
async def calculate(a: int, b: int, op: str = '+') -> Dict:
    if op == '+':
        return {'result': a + b}
    elif op == '*':
        return {'result': a * b}
    return {'error': f'Unknown op: {op}'}

async def search(query: str) -> Dict:
    return {'results': [f'Result 1 for {query}', f'Result 2 for {query}']}

agent = BasicAgent('MyAgent')
agent.register_tool(Tool('calculate', 'Do math', calculate, {'a': int, 'b': int, 'op': str}))
agent.register_tool(Tool('search', 'Search info', search, {'query': str}))

# Run
await agent.run_once('Solve math problem', {'a': 5, 'b': 3, 'op': '+'})

print(f"\nAgent state: {agent.state.value}")
print(f"Memory: {len(agent.memory)} actions")


# ======================================================================
# ## Level 2: Advanced with State Management, Error Recovery, Task Decomposition
# ======================================================================

# Level 2: Advanced Agent with State, Recovery, Decomposition

class TaskType(Enum):
    ATOMIC = 'atomic'      # Single tool call
    SEQUENTIAL = 'sequential'  # Multiple tools in order
    PARALLEL = 'parallel'  # Multiple tools concurrently

@dataclass
class Task:
    id: str
    goal: str
    task_type: TaskType = TaskType.ATOMIC
    subtasks: List['Task'] = field(default_factory=list)
    status: str = 'pending'
    result: Optional[Dict] = None
    attempts: int = 0

class AdvancedAgent(BasicAgent):
    """Agent with task management and recovery."""
    
    def __init__(self, name: str = 'Agent', max_retries: int = 3):
        super().__init__(name)
        self.max_retries = max_retries
        self.tasks: Dict[str, Task] = {}
        self.current_task: Optional[Task] = None
        self.error_count = 0
    
    def decompose_task(self, goal: str, available_tools: List[str]) -> Task:
        """Decompose goal into subtasks."""
        # Simple heuristic: multiple keywords = multiple subtasks
        keywords = goal.lower().split()
        
        if len(keywords) > 3 and len(available_tools) > 1:
            # Multi-step task
            subtasks = [
                Task(f'subtask_{i}', f'{goal} (step {i+1})', TaskType.ATOMIC)
                for i in range(min(2, len(available_tools)))
            ]
            task = Task(f'task_{int(time.time()*1000)}', goal, TaskType.SEQUENTIAL, subtasks)
        else:
            # Single-step task
            task = Task(f'task_{int(time.time()*1000)}', goal, TaskType.ATOMIC)
        
        print(f"[DECOMPOSE] {goal} → {len(task.subtasks)} subtasks")
        return task
    
    async def execute_with_recovery(self, task: Task, context: Dict) -> Dict:
        """Execute task with error recovery."""
        self.current_task = task
        task.attempts += 1
        
        try:
            # Choose a tool based on goal
            tool_name = list(self.tools.keys())[0] if self.tools else None
            if not tool_name:
                raise Exception('No tools available')
            
            # Execute
            result = await self.act(tool_name, context)
            
            if 'error' not in result:
                task.status = 'completed'
                task.result = result
                return result
            else:
                raise Exception(result.get('error', 'Tool failed'))
        
        except Exception as e:
            self.error_count += 1
            
            if task.attempts < self.max_retries:
                print(f"[RETRY] Attempt {task.attempts}/{self.max_retries}: {str(e)[:40]}")
                await asyncio.sleep(0.5 * (2 ** (task.attempts - 1)))  # Backoff
                return await self.execute_with_recovery(task, context)
            else:
                task.status = 'failed'
                print(f"[FAILED] Max retries exceeded")
                return {'error': f'Task failed after {self.max_retries} attempts'}
    
    async def run_complex(self, goal: str, context: Dict) -> Dict:
        """Run multi-step task."""
        # Decompose
        task = self.decompose_task(goal, list(self.tools.keys()))
        self.tasks[task.id] = task
        
        # Execute
        if task.task_type == TaskType.ATOMIC:
            return await self.execute_with_recovery(task, context)
        
        elif task.task_type == TaskType.SEQUENTIAL:
            results = []
            for subtask in task.subtasks:
                result = await self.execute_with_recovery(subtask, context)
                results.append(result)
            return {'subtask_results': results}
        
        else:  # PARALLEL
            tasks = [self.execute_with_recovery(st, context) for st in task.subtasks]
            results = await asyncio.gather(*tasks)
            return {'parallel_results': results}
    
    def get_agent_stats(self) -> Dict:
        return {
            'total_tasks': len(self.tasks),
            'completed': sum(1 for t in self.tasks.values() if t.status == 'completed'),
            'failed': sum(1 for t in self.tasks.values() if t.status == 'failed'),
            'total_errors': self.error_count,
            'memory_size': len(self.memory)
        }

# Test Level 2
print("\nTesting advanced agent with recovery...\n")

agent = AdvancedAgent('SmartAgent', max_retries=2)
agent.register_tool(Tool('analyze', 'Analyze data', calculate))
agent.register_tool(Tool('search', 'Search info', search))

# Run complex task
result = await agent.run_complex('Analyze data and search for results', {'a': 10, 'b': 5, 'op': '*', 'query': 'optimization'})
print(f"\nResult: {result}")
print(f"Agent stats: {agent.get_agent_stats()}")



# ======================================================================
# ## Real-World Example 1: Multi-Step Research Agent
# ======================================================================

# Example 1: Multi-step research agent

class ResearchAgent(AdvancedAgent):
    """Agent that performs multi-step research."""
    
    def __init__(self):
        super().__init__('ResearchAgent')
        self.findings = []
    
    async def research_pipeline(self, topic: str) -> Dict:
        """Multi-step research: search → analyze → synthesize."""
        print(f"\n[RESEARCH] Starting research on '{topic}'...")
        
        # Step 1: Search
        print("[STEP 1] Searching for relevant information...")
        search_result = await self.act('search', {'query': topic})
        sources = search_result.get('results', [])
        
        # Step 2: Analyze (simulate)
        print("[STEP 2] Analyzing findings...")
        analysis = {
            'source_count': len(sources),
            'topics_identified': ['core_concept', 'applications', 'challenges'],
            'confidence': 0.85
        }
        self.findings.append(analysis)
        
        # Step 3: Synthesize
        print("[STEP 3] Synthesizing results...")
        synthesis = {
            'topic': topic,
            'summary': f'Found {len(sources)} sources covering main aspects',
            'key_insights': ['Insight 1', 'Insight 2'],
            'confidence': analysis['confidence']
        }
        
        return synthesis

# Test
print("Example 1: Multi-Step Research Agent")

agent = ResearchAgent()
agent.register_tool(Tool('search', 'Search', search))
agent.register_tool(Tool('calculate', 'Calculate', calculate))

result = await agent.research_pipeline('machine learning optimization')
print(f"\nResearch result:")
for key, value in result.items():
    print(f"  {key}: {value}")

print(f"\nFindings collected: {len(agent.findings)}")



# ======================================================================
# ## Real-World Example 2: Error Recovery in Multi-Tool Orchestration
# ======================================================================

# Example 2: Fault-tolerant tool orchestration

class FaultTolerantAgent(AdvancedAgent):
    """Agent that handles tool failures gracefully."""
    
    def __init__(self):
        super().__init__('FaultTolerantAgent', max_retries=3)
        self.tool_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
    
    async def call_tool_with_fallback(self, primary_tool: str, fallback_tool: str, args: Dict) -> Dict:
        """Try primary tool, fallback to alternative."""
        # Try primary
        print(f"[TRY] {primary_tool}")
        result = await self.act(primary_tool, args)
        
        if 'error' not in result:
            self.tool_stats[primary_tool]['success'] += 1
            return result
        
        # Try fallback
        print(f"[FALLBACK] {fallback_tool}")
        result = await self.act(fallback_tool, args)
        
        if 'error' not in result:
            self.tool_stats[fallback_tool]['success'] += 1
        else:
            self.tool_stats[fallback_tool]['failure'] += 1
        
        return result
    
    def get_tool_health(self) -> Dict:
        """Get tool reliability metrics."""
        health = {}
        for tool_name, stats in self.tool_stats.items():
            total = stats['success'] + stats['failure']
            if total > 0:
                health[tool_name] = stats['success'] / total
        return health

# Test
print("\nExample 2: Fault-Tolerant Orchestration\n")

agent = FaultTolerantAgent()
agent.register_tool(Tool('fast_compute', 'Quick computation', calculate))
agent.register_tool(Tool('slow_compute', 'Slow but reliable', calculate))

print("Attempting fast_compute with slow_compute fallback:")
result = await agent.call_tool_with_fallback('fast_compute', 'slow_compute', {'a': 20, 'b': 4, 'op': '+'})
print(f"Result: {result}")

print(f"\nTool health: {agent.get_tool_health()}")



# ======================================================================
# ## Real-World Example 3: Complex Multi-Hop Reasoning
# ======================================================================

# Example 3: Multi-hop reasoning

class ReasoningAgent(AdvancedAgent):
    """Agent that chains reasoning steps."""
    
    async def multi_hop_reasoning(self, problem: str, steps: int = 3) -> Dict:
        """Multi-hop problem solving."""
        print(f"\n[REASONING] Problem: {problem} ({steps} steps)")
        
        reasoning_trace = []
        current_state = {'input': problem}
        
        for step_num in range(steps):
            print(f"\n[HOP {step_num + 1}]")
            
            # Think: What's the next step?
            decision = self.think(current_state)
            reasoning_trace.append(decision)
            
            # Act: Use a tool
            if self.tools:
                tool_name = list(self.tools.keys())[step_num % len(self.tools)]
                result = await self.act(tool_name, {'a': step_num + 1, 'b': step_num + 2, 'op': '+'})
                current_state = {'step': step_num + 1, 'result': result}
        
        return {
            'problem': problem,
            'steps_taken': steps,
            'final_state': current_state,
            'reasoning_trace': reasoning_trace
        }

# Test
print("Example 3: Multi-Hop Reasoning")

agent = ReasoningAgent('ReasoningAgent')
agent.register_tool(Tool('calculate', 'Calculate', calculate))
agent.register_tool(Tool('search', 'Search', search))

result = await agent.multi_hop_reasoning('Solve complex optimization problem', steps=3)
print(f"\nFinal result:")
for key in ['problem', 'steps_taken']:
    print(f"  {key}: {result[key]}")

print(f"Reasoning trace length: {len(result['reasoning_trace'])}")



# ======================================================================
# ## Comparison & Metrics
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np

# Simulate agent performance metrics
agent_types = ['Basic', 'Advanced', 'Research', 'Reasoning']
success_rates = [0.75, 0.92, 0.88, 0.85]
error_recovery = [0.0, 0.75, 0.80, 0.70]  # % of errors recovered
avg_steps = [1, 2.5, 3.2, 4.5]
exec_time = [0.050, 0.120, 0.250, 0.350]  # seconds

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Success rate
colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71']
ax1.bar(agent_types, success_rates, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Success Rate', fontsize=11)
ax1.set_title('Agent Success Rate', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.0)
for i, v in enumerate(success_rates):
    ax1.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=10)

# Error recovery
ax2.bar(agent_types, error_recovery, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Recovery Rate', fontsize=11)
ax2.set_title('Error Recovery Capability', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.0)
for i, v in enumerate(error_recovery):
    if v > 0:
        ax2.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=10)

# Average steps
ax3.plot(range(len(agent_types)), avg_steps, marker='o', linewidth=2, markersize=10, color='#3498db')
ax3.set_xticks(range(len(agent_types)))
ax3.set_xticklabels(agent_types)
ax3.set_ylabel('Average Steps', fontsize=11)
ax3.set_title('Task Complexity', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
for i, v in enumerate(avg_steps):
    ax3.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=10)

# Execution time
ax4.bar(agent_types, exec_time, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Execution Time (seconds)', fontsize=11)
ax4.set_title('Execution Speed', fontsize=12, fontweight='bold')
for i, v in enumerate(exec_time):
    ax4.text(i, v + 0.01, f'{v*1000:.0f}ms', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

print("Agent Performance Comparison:")
print(f"\n{'Agent':<15} {'Success':<12} {'Recovery':<12} {'Steps':<10} {'Time (ms)':<12}")
print("-" * 65)
for agent, success, recovery, steps, time_ms in zip(agent_types, success_rates, error_recovery, avg_steps, exec_time):
    print(f"{agent:<15} {success:>10.0%} {recovery:>10.0%} {steps:>8.1f} {time_ms*1000:>10.0f}")



# ======================================================================
# ## Key Takeaways
# **Agent Architecture:**
# 1. Observe-Think-Act loop is core pattern
# 2. Tool registry enables dynamic dispatch
# 3. Task decomposition handles complexity
# 4. Error recovery improves reliability
# **Decision Making:**
# - Heuristic-based routing (simple)
# - Plan-based (sequence of steps)
# - Reasoning-based (multi-hop inference)
# **Resilience Patterns:**
# - Retry with backoff
# - Primary + fallback tools
# - Health metrics guide routing
# - Graceful degradation
# **Production Considerations:**
# - Agent trace logging (audit trail)
# - Performance monitoring (latency, success rate)
# - Token usage tracking (cost control)
# - Tool health dashboards
# **When to use agents:**
# - Complex multi-step problems
# - Need error recovery and retries
# - Tool composition required
# - Interactive task solving
# **Related Concepts:** [[tool-use]], [[planning]], [[state-machines]], [[error-recovery]]
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Extend Example 1:** Add parallel step execution for independent subtasks
# 2. **Modify Example 2:** Implement tool ranking based on historical success rates
# 3. **Enhance Example 3:** Add goal-aware tool selection (route based on problem type)
# ======================================================================
