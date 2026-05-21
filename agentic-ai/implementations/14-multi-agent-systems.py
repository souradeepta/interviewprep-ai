"""
Auto-generated from 14-multi-agent-systems.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Multi-Agent Systems
# Learning objectives:
# - Understand hierarchical, cooperative, competitive, and debate coordination patterns
# - Implement manager-worker and peer agent systems
# ======================================================================

import os
import json
import asyncio
from typing import Dict, List, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for multi-agent systems!")


# ======================================================================
# ## Level 1: Basic Hierarchical System
# Manager delegates tasks to specialist workers.
# ======================================================================

class HierarchicalTeam:
    """Manager assigns tasks to specialists"""
    def __init__(self):
        self.client = Anthropic()
        self.agents = {
            "writer": "Writes clear, concise content",
            "critic": "Reviews and provides feedback",
            "summarizer": "Creates summaries"
        }
    
    def assign_task(self, task: str, agent_type: str) -> str:
        """Manager delegates to specialist"""
        role = self.agents.get(agent_type, "general assistant")
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"[You are a {agent_type}. {role}]\nTask: {task}"
            }]
        )
        
        return response.content[0].text
    
    def execute(self, request: str) -> Dict:
        """Manager orchestrates: plan → execute → review"""
        # Step 1: Writer creates
        print("\n[Manager] → Writer: Create content")
        content = self.assign_task(f"Write about: {request}", "writer")
        
        # Step 2: Critic reviews
        print("[Manager] → Critic: Review content")
        feedback = self.assign_task(f"Review this: {content[:100]}", "critic")
        
        # Step 3: Summarizer summarizes
        print("[Manager] → Summarizer: Summarize")
        summary = self.assign_task(f"Summarize: {content[:100]}", "summarizer")
        
        return {
            "content": content[:80] + "...",
            "feedback": feedback[:80] + "...",
            "summary": summary[:80] + "..."
        }

# Test
team = HierarchicalTeam()
result = team.execute("machine learning basics")
print(f"\nResult: {json.dumps(result, indent=2)}")


# ======================================================================
# ## Level 2: Advanced Multi-Agent with Error Handling
# Production system with delegation, timeout handling, and fallbacks.
# ======================================================================

import time

class ProductionMultiAgentSystem:
    """Production-grade multi-agent with error handling"""
    def __init__(self, timeout_seconds=10):
        self.client = Anthropic()
        self.timeout = timeout_seconds
        self.agents = {
            "analyst": {"role": "Analyzes problems", "specialty": "analysis"},
            "implementer": {"role": "Implements solutions", "specialty": "code"},
            "validator": {"role": "Validates results", "specialty": "testing"}
        }
        self.metrics = {"total_calls": 0, "errors": 0, "timeouts": 0}
    
    def call_agent(self, agent_type: str, prompt: str, retries=2) -> str:
        """Call agent with retry logic and error handling"""
        for attempt in range(retries):
            try:
                self.metrics["total_calls"] += 1
                start = time.time()
                
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=256,
                    messages=[{
                        "role": "user",
                        "content": f"[{agent_type}: {self.agents[agent_type]['role']}]\n{prompt}"
                    }]
                )
                
                elapsed = time.time() - start
                if elapsed > self.timeout:
                    print(f"⏱️  {agent_type} timeout ({elapsed:.1f}s)")
                    self.metrics["timeouts"] += 1
                    if attempt < retries - 1:
                        continue
                    return f"[Timeout after {self.timeout}s]"
                
                return response.content[0].text
            
            except Exception as e:
                self.metrics["errors"] += 1
                if attempt < retries - 1:
                    continue
                return f"[Error: {str(e)[:50]}]"
        
        return "[Failed after retries]"
    
    def solve_problem(self, problem: str) -> Dict:
        """Multi-step problem solving with delegation"""
        # Stage 1: Analysis
        print("\nStage 1: Analysis")
        analysis = self.call_agent("analyst", f"Analyze this problem: {problem}")
        
        # Stage 2: Implementation
        print("Stage 2: Implementation")
        solution = self.call_agent("implementer", f"Implement based on: {analysis[:100]}")
        
        # Stage 3: Validation
        print("Stage 3: Validation")
        validation = self.call_agent("validator", f"Validate this solution: {solution[:100]}")
        
        return {
            "problem": problem,
            "analysis": analysis[:60] + "...",
            "solution": solution[:60] + "...",
            "validation": validation[:60] + "...",
            "metrics": self.metrics
        }

# Test
system = ProductionMultiAgentSystem(timeout_seconds=15)
result = system.solve_problem("How to optimize a slow API?")
print(f"\nMetrics: {result['metrics']}")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Hierarchical Task Decomposition
# ======================================================================

class HierarchicalTaskDecomposition:
    """Manager breaks task into subtasks, workers handle parallel execution"""
    def __init__(self):
        self.client = Anthropic()
        self.task_log = []
    
    def decompose(self, main_task: str) -> List[str]:
        """Break large task into subtasks"""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""Break this into 3 independent subtasks:
{main_task}

Format: Return only the 3 tasks, one per line."""
            }]
        )
        
        tasks = [t.strip() for t in response.content[0].text.split("\n") if t.strip()]
        return tasks[:3]
    
    def execute_subtask(self, subtask: str) -> str:
        """Worker executes one subtask"""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"Execute this task: {subtask}"
            }]
        )
        return response.content[0].text
    
    def execute(self, main_task: str) -> Dict:
        """Manager decomposes, workers execute in parallel (simulated)"""
        print(f"\n[Manager] Decomposing: {main_task}")
        subtasks = self.decompose(main_task)
        
        results = {}
        for i, subtask in enumerate(subtasks, 1):
            print(f"[Worker {i}] Executing: {subtask[:40]}...")
            result = self.execute_subtask(subtask)
            results[f"subtask_{i}"] = result[:50] + "..."
            self.task_log.append({"task": subtask, "result": result[:50]})
        
        return {
            "main_task": main_task,
            "subtasks": subtasks,
            "results": results,
            "tasks_completed": len(subtasks)
        }

# Test
system = HierarchicalTaskDecomposition()
result = system.execute("Build a content recommendation system")
print(f"\nCompleted {result['tasks_completed']} subtasks")


# ======================================================================
# ### Example 2: Cooperative Peer Negotiation
# ======================================================================

class CooperativePeers:
    """Agents consult each other to reach consensus"""
    def __init__(self):
        self.client = Anthropic()
        self.conversation_history = []
    
    def agent_perspective(self, agent_name: str, topic: str, round_num: int) -> str:
        """Get one agent's perspective"""
        context = ""
        if self.conversation_history:
            context = "\n".join([f"- {h['agent']}: {h['view'][:50]}" for h in self.conversation_history[-2:]])
            context = f"\nPrior perspectives:\n{context}"
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""You are {agent_name}. Give your perspective on: {topic}
{context}
Round {round_num}. One sentence, then build on others' points."""
            }]
        )
        return response.content[0].text
    
    def negotiate(self, topic: str, num_rounds=2) -> Dict:
        """Multiple agents discuss and build consensus"""
        agents = ["Alice", "Bob", "Charlie"]
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n--- Round {round_num} ---")
            for agent in agents:
                view = self.agent_perspective(agent, topic, round_num)
                self.conversation_history.append({
                    "agent": agent,
                    "round": round_num,
                    "view": view
                })
                print(f"[{agent}]: {view[:60]}...")
        
        return {
            "topic": topic,
            "rounds": num_rounds,
            "agents": agents,
            "final_perspectives": [
                {"agent": h["agent"], "view": h["view"][:80]} 
                for h in self.conversation_history[-3:]
            ]
        }

# Test
peers = CooperativePeers()
result = peers.negotiate("Should AI be regulated?", num_rounds=2)
print(f"\nConsensus reached after {result['rounds']} rounds with {len(result['agents'])} agents")


# ======================================================================
# ### Example 3: Competitive Racing with Evaluation
# ======================================================================

class CompetitiveAgents:
    """Multiple agents compete to solve problem, evaluator picks best"""
    def __init__(self):
        self.client = Anthropic()
    
    def generate_solution(self, agent_name: str, problem: str) -> str:
        """One agent attempts solution"""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"[Agent {agent_name}] Solve: {problem}. Be specific and practical."
            }]
        )
        return response.content[0].text
    
    def evaluate(self, problem: str, solutions: Dict[str, str]) -> str:
        """Evaluator ranks solutions"""
        solutions_text = "\n".join([
            f"Agent {name}: {sol[:80]}" for name, sol in solutions.items()
        ])
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""Problem: {problem}

Solutions:
{solutions_text}

Which is best? Explain in one sentence."""
            }]
        )
        return response.content[0].text
    
    def race(self, problem: str) -> Dict:
        """Agents compete, evaluator judges"""
        agents = ["Alice", "Bob", "Charlie"]
        solutions = {}
        
        print(f"\n[Competitors racing on: {problem[:40]}...]")
        for agent in agents:
            sol = self.generate_solution(agent, problem)
            solutions[agent] = sol
            print(f"[{agent}]: {sol[:50]}...")
        
        print(f"\n[Evaluator] Judging...")
        winner_assessment = self.evaluate(problem, solutions)
        print(f"[Evaluation]: {winner_assessment[:80]}...")
        
        return {
            "problem": problem,
            "competitors": agents,
            "solutions": {k: v[:60] + "..." for k, v in solutions.items()},
            "evaluation": winner_assessment[:100]
        }

# Test
race = CompetitiveAgents()
result = race.race("How to reduce carbon emissions?")
print(f"\nRace complete with {len(result['competitors'])} competitors")


# ======================================================================
# ## Key Takeaways
# 1. **Hierarchical scales best.** Manager assigns tasks, workers execute. Adds one coordination layer, not N-way communication.
# 2. **Start with 3-5 agents.** More agents = more communication overhead. Test with small team first.
# ======================================================================
