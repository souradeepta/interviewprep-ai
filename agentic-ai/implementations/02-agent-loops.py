"""
Auto-generated from 02-agent-loops.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Loops
# Learning objectives:
# - Understand the core loop structure: observe → think → act → observe
# - Implement basic synchronous loops
# ======================================================================

# ======================================================================
# ## Setup
# ======================================================================

# Optional: Install dependencies if needed
# !pip install anthropic langchain llama-index python-dotenv

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-key-here")

print("Setup complete. Ready to explore agent loops!")


# ======================================================================
# ## Level 1: Basic Agent Loop
# Simplest possible loop: LLM + one tool, repeating until goal.
# ======================================================================

# Level 1: Basic loop - minimal implementation

def basic_agent_loop(query: str, max_steps: int = 5):
    """Minimal agent loop showing core pattern"""
    client = Anthropic()
    
    tools = [
        {
            "name": "calculator",
            "description": "Evaluate math expressions",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    ]
    
    messages = [{"role": "user", "content": query}]
    
    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            tools=tools,
            messages=messages
        )
        
        # Early stopping: goal reached
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
        
        # Add assistant response
        messages.append({"role": "assistant", "content": response.content})
        
        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  Tool: {block.name}")
                print(f"  Args: {block.input}")
                
                # Execute tool
                try:
                    result = str(eval(block.input["expression"]))
                    print(f"  Result: {result}")
                except Exception as e:
                    result = f"Error: {e}"
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
    
    return "Max steps exceeded"

# Test
result = basic_agent_loop("What is 42 * 17 + 100?")
print(f"\n=== Final Answer ===\n{result}")


# ======================================================================
# ## Level 2: Advanced Loop with Multiple Tools and Error Handling
# Full loop with multiple tools, error handling, and loop detection.
# ======================================================================

# Level 2: Advanced loop with error handling and loop detection

from collections import defaultdict

class AdvancedAgent:
    def __init__(self, max_steps: int = 10, max_same_tool: int = 3):
        self.client = Anthropic()
        self.max_steps = max_steps
        self.max_same_tool = max_same_tool
        self.tool_calls = defaultdict(list)
    
    def run(self, query: str):
        tools = [
            {
                "name": "calculator",
                "description": "Evaluate math expressions",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]
                }
            },
            {
                "name": "lookup",
                "description": "Look up information",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
        
        messages = [{"role": "user", "content": query}]
        
        for step in range(self.max_steps):
            print(f"\n[Step {step + 1}/{self.max_steps}]")
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=tools,
                messages=messages
            )
            
            # Early stopping
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, 'text'):
                        return {"status": "success", "answer": block.text}
            
            messages.append({"role": "assistant", "content": response.content})
            
            # Process tool calls with loop detection
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Check for loops
                    self.tool_calls[block.name].append(step)
                    if len(self.tool_calls[block.name]) > self.max_same_tool:
                        return {
                            "status": "loop_detected",
                            "tool": block.name,
                            "count": len(self.tool_calls[block.name])
                        }
                    
                    print(f"  Tool: {block.name}")
                    
                    # Execute tool
                    try:
                        if block.name == "calculator":
                            result = str(eval(block.input["expression"]))
                        elif block.name == "lookup":
                            result = f"Information about {block.input['query']}"
                        else:
                            result = "Unknown tool"
                    except Exception as e:
                        result = f"Error: {str(e)}"
                    
                    print(f"  Result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        
        return {"status": "max_steps_exceeded", "steps": self.max_steps}

# Test
agent = AdvancedAgent(max_steps=10, max_same_tool=2)
result = agent.run("Calculate 25 * 4 and find info about that number")
print(f"\n=== Result ===\n{json.dumps(result, indent=2)}")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Explicit ReAct Loop (Reasoning + Acting)
# ======================================================================

# Level 3 Example 1: Explicit ReAct pattern
# Shows visible Thought → Action → Observation steps

def react_loop(query: str, max_steps: int = 10):
    """ReAct: Explicit reasoning + acting"""
    client = Anthropic()
    
    # Prompt that encourages ReAct pattern
    system_prompt = """You are an AI agent that solves problems step by step.
Use the following format for each step:
Thought: [what do you think?]
Action: [tool_name(args)]
Observation: [result from tool]
Then repeat until you reach the final answer.
"""
    
    tools = [
        {
            "name": "calculator",
            "description": "Evaluate math",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        },
        {
            "name": "lookup",
            "description": "Look up facts",
            "input_schema": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"]
            }
        }
    ]
    
    messages = [{"role": "user", "content": query}]
    
    for step in range(max_steps):
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages
        )
        
        # Print reasoning
        print(f"\n[Step {step + 1}]")
        for block in response.content:
            if hasattr(block, 'text'):
                print(f"Reasoning:\n{block.text}")
        
        if response.stop_reason == "end_turn":
            return response.content[-1].text if response.content else "No answer"
        
        messages.append({"role": "assistant", "content": response.content})
        
        # Execute tools
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "calculator":
                    result = str(eval(block.input["expression"]))
                else:
                    result = f"Fact: {block.input.get('fact', 'unknown')}"
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
    
    return "Max steps"

# Test
result = react_loop("Calculate 100 * 5 and tell me a fact about 500")
print(f"\n=== Final Answer ===\n{result}")


# ======================================================================
# ### Example 2: LangChain Agent Framework
# ======================================================================

# Level 3 Example 2: Using LangChain framework

try:
    from langchain.agents import Tool, initialize_agent, AgentType
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    
    # Define tools
    def math_tool(expr: str) -> str:
        return str(eval(expr))
    
    def search_tool(query: str) -> str:
        return f"Found: information about {query}"
    
    tools = [
        Tool(name="Math", func=math_tool, description="Math calculations"),
        Tool(name="Search", func=search_tool, description="Search for info")
    ]
    
    # Initialize agent
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    result = agent.run("What is 42 * 17?")
    print(f"Result: {result}")
    
except ImportError:
    print("[SKIPPED] LangChain not installed. Install with: pip install langchain openai")
except Exception as e:
    print(f"[SKIPPED] LangChain example failed: {e}")


# ======================================================================
# ### Example 3: Production Loop with Cost Tracking and Timeout
# ======================================================================

# Level 3 Example 3: Production-grade loop with cost tracking and timeout

import time
from typing import Dict, Any

class ProductionAgent:
    def __init__(self, max_steps: int = 10, timeout_seconds: float = 30):
        self.client = Anthropic()
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.metrics = {
            "total_steps": 0,
            "total_tokens": 0,
            "tool_calls": {},
            "elapsed_time": 0
        }
    
    def run(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        tools = [
            {
                "name": "calculator",
                "description": "Math calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]
                }
            }
        ]
        
        messages = [{"role": "user", "content": query}]
        
        for step in range(self.max_steps):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                return {
                    "status": "timeout",
                    "elapsed_time": elapsed,
                    "metrics": self.metrics
                }
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=tools,
                messages=messages
            )
            
            # Track metrics
            self.metrics["total_steps"] += 1
            self.metrics["total_tokens"] += response.usage.input_tokens + response.usage.output_tokens
            
            if response.stop_reason == "end_turn":
                self.metrics["elapsed_time"] = time.time() - start_time
                return {
                    "status": "success",
                    "answer": response.content[-1].text if response.content else "No answer",
                    "metrics": self.metrics
                }
            
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    self.metrics["tool_calls"][block.name] = self.metrics["tool_calls"].get(block.name, 0) + 1
                    
                    try:
                        result = str(eval(block.input["expression"]))
                    except Exception as e:
                        result = f"Error: {e}"
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        
        self.metrics["elapsed_time"] = time.time() - start_time
        return {
            "status": "max_steps_exceeded",
            "metrics": self.metrics
        }

# Test
agent = ProductionAgent(max_steps=10, timeout_seconds=30)
result = agent.run("What is 123 * 456 + 789?")
print(f"\nProduction Result:")
print(json.dumps(result, indent=2))


# ======================================================================
# ## Key Takeaways
# 1. **The Core Loop:** Observe → Think → Act → Observe → Repeat until goal or max_steps
# 2. **Early Stopping Matters:** Check if goal is reached after each iteration. Don't always run max_steps.
# ======================================================================
