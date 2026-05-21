"""
Auto-generated from 01-what-is-an-agent.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # What Is an Agent?
# Learning objectives:
# - Understand agent definition: perception, reasoning, action
# - Implement basic agent loop (ReAct pattern)
# ======================================================================

import os
import json
import time
from anthropic import Anthropic
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready to build agents!")


# ======================================================================
# ## Level 1: Basic Agent Loop
# Minimal agent implementing perception → reasoning → action.
# ======================================================================

class BasicAgent:
    """Minimal agent: ask question, get answer, done."""
    def __init__(self):
        self.client = Anthropic()
    
    def perceive(self, question: str) -> dict:
        """Perception: receive user input."""
        return {"input": question, "type": "question"}
    
    def reason(self, perception: dict) -> str:
        """Reasoning: use LLM to think about problem."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": perception["input"]}]
        )
        return response.content[0].text
    
    def act(self, reasoning: str) -> dict:
        """Action: return result."""
        return {"response": reasoning, "status": "complete"}
    
    def run(self, question: str) -> dict:
        """Agent loop: perceive → reason → act."""
        print(f"\n[Agent] Perceiving: {question[:40]}...")
        perception = self.perceive(question)
        
        print(f"[Agent] Reasoning...")
        reasoning = self.reason(perception)
        
        print(f"[Agent] Acting...")
        result = self.act(reasoning)
        
        return result

# Test
agent = BasicAgent()
result = agent.run("What is 2+2?")
print(f"\nResult: {result['response'][:100]}...")


# ======================================================================
# ## Level 2: Advanced Agent with Tool Use
# Agent can select and use tools (ReAct pattern).
# ======================================================================

import json

class ToolUsingAgent:
    """Agent that can select and use tools."""
    def __init__(self):
        self.client = Anthropic()
        self.tools = [
            {
                "name": "calculate",
                "description": "Perform mathematical calculation",
                "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}}
            },
            {
                "name": "search",
                "description": "Search for information",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        self.step = 0
    
    def perceive(self, task: str) -> str:
        """Perceive task."""
        return task
    
    def reason_and_plan(self, task: str) -> dict:
        """Use LLM to reason about task."""
        self.step += 1
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            tools=self.tools,
            messages=[{"role": "user", "content": task}]
        )
        
        return {
            "step": self.step,
            "response": response.content,
            "stop_reason": response.stop_reason
        }
    
    def use_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute tool."""
        if tool_name == "calculate":
            expr = tool_input.get("expression", "")
            try:
                result = eval(expr)
                return {"result": str(result)}
            except:
                return {"error": "Invalid expression"}
        elif tool_name == "search":
            return {"results": ["Search result 1", "Search result 2"]}
        return {"error": f"Unknown tool: {tool_name}"}
    
    def run(self, task: str) -> dict:
        """Agent loop with tool use."""
        print(f"\n[Agent] Task: {task[:40]}...")
        
        for iteration in range(3):
            perception = self.perceive(task)
            reasoning = self.reason_and_plan(perception)
            
            print(f"  Step {reasoning['step']}: {reasoning['stop_reason']}")
            
            if reasoning["stop_reason"] == "end_turn":
                break
            elif reasoning["stop_reason"] == "tool_use":
                for block in reasoning["response"]:
                    if hasattr(block, "type") and block.type == "tool_use":
                        print(f"    Using tool: {block.name}")
        
        return {"completed": True, "iterations": reasoning["step"]}

# Test
agent = ToolUsingAgent()
result = agent.run("What is 15 * 25?")
print(f"\nAgent result: {result}")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Agent with Error Recovery
# ======================================================================

class RobustAgent:
    """Agent with error handling and recovery."""
    def __init__(self, max_retries: int = 3):
        self.client = Anthropic()
        self.max_retries = max_retries
        self.metrics = {"success": 0, "failures": 0, "retries": 0}
    
    def run_with_retry(self, task: str) -> dict:
        """Run agent with automatic retry on failure."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=256,
                    messages=[{"role": "user", "content": task}]
                )
                
                self.metrics["success"] += 1
                return {
                    "success": True,
                    "response": response.content[0].text,
                    "attempts": attempt + 1
                }
            
            except Exception as e:
                self.metrics["retries"] += 1
                if attempt == self.max_retries - 1:
                    self.metrics["failures"] += 1
                    return {"success": False, "error": str(e), "attempts": attempt + 1}
        
        return {"success": False, "error": "Max retries exceeded"}

# Test
agent = RobustAgent(max_retries=2)
result = agent.run_with_retry("Tell me a fun fact about agents.")
print(f"Result: success={result['success']}, attempts={result.get('attempts', 0)}")
if result["success"]:
    print(f"Response: {result['response'][:80]}...")


# ======================================================================
# ### Example 2: Agent with Caching
# ======================================================================

class CachingAgent:
    """Agent that caches results to reduce API calls."""
    def __init__(self):
        self.client = Anthropic()
        self.cache = {}
    
    def run(self, task: str) -> dict:
        """Run with caching."""
        # Check cache
        if task in self.cache:
            print(f"[Cache HIT] {task[:30]}...")
            return {"response": self.cache[task], "from_cache": True}
        
        print(f"[Cache MISS] {task[:30]}... (calling API)")
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": task}]
        )
        
        result = response.content[0].text
        self.cache[task] = result  # Store in cache
        
        return {"response": result, "from_cache": False}

# Test
agent = CachingAgent()
print("First call:")
result1 = agent.run("What is 2+2?")
print("Second call (same query):")
result2 = agent.run("What is 2+2?")
print(f"Cache size: {len(agent.cache)} items")


# ======================================================================
# ### Example 3: Agent with Cost Tracking
# ======================================================================

class CostAwareAgent:
    """Agent that tracks costs and optimizes spending."""
    def __init__(self):
        self.client = Anthropic()
        self.costs = {"input_tokens": 0, "output_tokens": 0, "total_usd": 0}
    
    def run(self, task: str) -> dict:
        """Run with cost tracking."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": task}]
        )
        
        # Track costs
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        # Approximate pricing (Sonnet)
        input_cost = input_tokens * 0.003 / 1000
        output_cost = output_tokens * 0.006 / 1000
        total_cost = input_cost + output_cost
        
        self.costs["input_tokens"] += input_tokens
        self.costs["output_tokens"] += output_tokens
        self.costs["total_usd"] += total_cost
        
        return {
            "response": response.content[0].text,
            "cost_usd": round(total_cost, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

# Test
agent = CostAwareAgent()
for i in range(2):
    result = agent.run(f"Question {i+1}: Tell me something interesting.")
    print(f"Query {i+1}: ${result['cost_usd']:.4f} ({result['output_tokens']} tokens)")

print(f"\nTotal cost: ${agent.costs['total_usd']:.4f}")
print(f"Total tokens: {agent.costs['input_tokens'] + agent.costs['output_tokens']}")


# ======================================================================
# ## Key Takeaways
# 1. **Agent = Perception → Reasoning → Action.** Agents receive input, think using LLM, then take action. This loop repeats.
# 2. **Agents can use tools.** ReAct pattern: agent decides which tool to use, calls it, observes result, and iterates.
# ======================================================================
