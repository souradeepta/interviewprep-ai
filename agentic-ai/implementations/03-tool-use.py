"""
Auto-generated from 03-tool-use.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Tool Use
# Objectives: Tool definition and schemas, agent loop with tool calling, error handling, tool validation and fallbacks
# ======================================================================

import asyncio
import json
from typing import Dict, List, Any, Callable
from dataclasses import dataclass

# Level 1: Basic Tool Definition and Execution

@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    execute_fn: Callable

class SimpleToolExecutor:
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def execute(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute a tool."""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool not found: {tool_name}"}
        
        tool = self.tools[tool_name]
        try:
            result = tool.execute_fn(**tool_input)
            return {"success": True, "result": result, "tool": tool_name}
        except Exception as e:
            return {"success": False, "error": str(e), "tool": tool_name}

# Define tools
def calculate(expression: str) -> float:
    """Perform arithmetic calculation."""
    return eval(expression)

def search_web(query: str) -> List[str]:
    """Simulate web search."""
    return [f"Result 1 for {query}", f"Result 2 for {query}", f"Result 3 for {query}"]

def get_time() -> str:
    """Get current time."""
    import datetime
    return datetime.datetime.now().isoformat()

# Create tools
calculator_tool = Tool(
    name="calculator",
    description="Perform arithmetic calculations. Examples: '5 + 3', '10 * 4'. Returns number.",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    },
    execute_fn=calculate
)

web_search_tool = Tool(
    name="web_search",
    description="Search the web for information. Returns list of results.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    },
    execute_fn=search_web
)

time_tool = Tool(
    name="get_time",
    description="Get current time in ISO format.",
    input_schema={
        "type": "object",
        "properties": {},
        "required": []
    },
    execute_fn=get_time
)

# Test Level 1
print("Level 1 - Basic Tool Execution:\n")
executor = SimpleToolExecutor()
executor.register_tool(calculator_tool)
executor.register_tool(web_search_tool)
executor.register_tool(time_tool)

# Execute tools
result1 = executor.execute("calculator", {"expression": "10 + 5"})
print(f"✓ calculator: {result1}")

result2 = executor.execute("web_search", {"query": "AI trends"})
print(f"✓ web_search: {result2}")

result3 = executor.execute("get_time", {})
print(f"✓ get_time: {result3['result']}")


# Level 2: Agent Loop with Tool Use and Error Handling

class AgentWithTools:
    def __init__(self, tools: List[Tool], max_iterations: int = 5):
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.tool_calls = {}
    
    def validate_tool_input(self, tool_name: str, tool_input: Dict) -> tuple[bool, str]:
        """Validate tool input against schema."""
        if tool_name not in self.tools:
            return False, f"Tool not found: {tool_name}"
        
        tool = self.tools[tool_name]
        required = tool.input_schema.get("required", [])
        
        for param in required:
            if param not in tool_input:
                return False, f"Missing required parameter: {param}"
        
        return True, ""
    
    async def execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute tool with validation and error handling."""
        # Check call limit
        if tool_name not in self.tool_calls:
            self.tool_calls[tool_name] = 0
        
        if self.tool_calls[tool_name] >= 3:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' call limit reached (3)"
            }
        
        # Validate input
        valid, error_msg = self.validate_tool_input(tool_name, tool_input)
        if not valid:
            return {"success": False, "error": f"Validation failed: {error_msg}"}
        
        # Execute
        try:
            tool = self.tools[tool_name]
            result = tool.execute_fn(**tool_input)
            self.tool_calls[tool_name] += 1
            return {"success": True, "result": result, "tool": tool_name}
        except Exception as e:
            return {
                "success": False,
                "error": f"{tool_name} execution failed: {str(e)[:50]}",
                "retryable": isinstance(e, (TimeoutError, ConnectionError))
            }
    
    async def agent_loop(self, user_query: str):
        """Simulate agent reasoning loop with tool use."""
        print(f"User Query: {user_query}\n")
        
        messages = []
        
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}:")
            
            # Simulate LLM decision (in real system, call Claude API)
            # For demo: decide whether to use tools based on query
            if "calculate" in user_query.lower() or "math" in user_query.lower():
                tool_name = "calculator"
                tool_input = {"expression": "42 * 17"}
            elif "search" in user_query.lower() or "find" in user_query.lower():
                tool_name = "web_search"
                tool_input = {"query": user_query}
            else:
                # No tool needed
                print(f"  → Reasoning complete, no tools needed")
                return "Task completed"
            
            # Execute tool
            print(f"  → Calling {tool_name} with {tool_input}")
            result = await self.execute_tool(tool_name, tool_input)
            
            if result["success"]:
                print(f"  ✓ Result: {str(result['result'])[:60]}...")
                return result["result"]
            else:
                print(f"  ✗ Error: {result['error']}")
                if result.get("retryable"):
                    print(f"  → Error is retryable")
                else:
                    return "Tool execution failed"
        
        return "Max iterations reached"

# Test Level 2
print("\nLevel 2 - Agent Loop with Tool Use:\n")
tools = [calculator_tool, web_search_tool, time_tool]
agent = AgentWithTools(tools, max_iterations=3)

# Run agent
result = await agent.agent_loop("Calculate 42 * 17")
print(f"\nFinal result: {result}\n")


# Example 1: Tool Use with Fallback Strategy

class RobustToolExecutor:
    def __init__(self, tools: List[Tool], fallbacks: Dict[str, List[str]] = None):
        self.tools = {t.name: t for t in tools}
        self.fallbacks = fallbacks or {}
    
    async def execute_with_fallback(self, tool_name: str, tool_input: Dict, max_retries: int = 2) -> Dict:
        """Execute tool with fallback strategy."""
        print(f"Executing {tool_name}...")
        
        # Try primary tool
        for attempt in range(max_retries):
            try:
                tool = self.tools.get(tool_name)
                if not tool:
                    return {"success": False, "error": f"Tool not found: {tool_name}"}
                
                result = tool.execute_fn(**tool_input)
                return {"success": True, "result": result, "tool": tool_name, "attempt": attempt + 1}
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {str(e)[:40]}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
        
        # Try fallback tools
        fallback_list = self.fallbacks.get(tool_name, [])
        for fallback_name in fallback_list:
            print(f"  → Trying fallback: {fallback_name}")
            try:
                tool = self.tools.get(fallback_name)
                if tool:
                    result = tool.execute_fn(**tool_input)
                    print(f"  ✓ Fallback succeeded")
                    return {"success": True, "result": result, "tool": fallback_name, "fallback": True}
            except Exception as e:
                print(f"  Fallback failed: {str(e)[:40]}")
        
        return {
            "success": False,
            "error": f"All attempts failed for '{tool_name}'",
            "tool": tool_name
        }

# Define alternative tools (simulated)
def search_cache(query: str) -> List[str]:
    """Search cached results."""
    return [f"Cached result for {query}"]

# Create cache search tool
cache_tool = Tool(
    name="search_cache",
    description="Search cached results (fallback for web search).",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    },
    execute_fn=search_cache
)

# Test Example 1
print("\nExample 1 - Tool Fallback Strategy:\n")
tools = [web_search_tool, cache_tool, calculator_tool]
fallbacks = {"web_search": ["search_cache"]}
executor = RobustToolExecutor(tools, fallbacks)

result = await executor.execute_with_fallback("web_search", {"query": "machine learning"})
print(f"Result: {result}\n")


# Example 2: Multi-Step Tool Composition

class ToolCompositionAgent:
    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}
        self.execution_trace = []
    
    async def execute_composed_tools(self, 
                                    tool_sequence: List[tuple],
                                    initial_input: Dict = None) -> Dict:
        """Execute sequence of tools, passing results between them.
        
        tool_sequence: list of (tool_name, input_template) tuples
        """
        print(f"Executing composed tools: {[t[0] for t in tool_sequence]}\n")
        
        current_result = initial_input
        
        for tool_name, input_template in tool_sequence:
            tool = self.tools.get(tool_name)
            if not tool:
                return {"success": False, "error": f"Tool not found: {tool_name}"}
            
            # Prepare input (can use previous result)
            if callable(input_template):
                tool_input = input_template(current_result)
            else:
                tool_input = input_template
            
            try:
                print(f"Step: {tool_name}")
                result = tool.execute_fn(**tool_input)
                self.execution_trace.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": str(result)[:60]
                })
                print(f"  ✓ Result: {str(result)[:60]}...")
                current_result = result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"{tool_name} failed: {str(e)}",
                    "trace": self.execution_trace
                }
        
        return {
            "success": True,
            "result": current_result,
            "trace": self.execution_trace
        }

# Test Example 2
print("Example 2 - Multi-Step Tool Composition:\n")
agent = ToolCompositionAgent([calculator_tool, web_search_tool])

# Define composition: calculate first, then search for significance
composition = [
    ("calculator", {"expression": "42 * 17"}),
    ("web_search", lambda result: {"query": f"significance of number {result}"})
]

result = await agent.execute_composed_tools(composition)
print(f"\nFinal success: {result['success']}")
print(f"Execution trace:")
for trace in result['trace']:
    print(f"  {trace['tool']}: {trace['result']}\n")


# Example 3: Conditional Tool Use (Routing)

class ConditionalToolAgent:
    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}
    
    async def route_and_execute(self, task: str, task_type: str) -> Dict:
        """Route task to appropriate tool based on type."""
        print(f"Task: {task}")
        print(f"Type: {task_type}\n")
        
        # Route based on task type
        if task_type == "arithmetic":
            tool_name = "calculator"
            tool_input = {"expression": task}
        elif task_type == "search":
            tool_name = "web_search"
            tool_input = {"query": task}
        elif task_type == "time":
            tool_name = "get_time"
            tool_input = {}
        else:
            return {"success": False, "error": f"Unknown task type: {task_type}"}
        
        # Execute routed tool
        print(f"Routed to: {tool_name}")
        try:
            tool = self.tools[tool_name]
            result = tool.execute_fn(**tool_input)
            return {
                "success": True,
                "result": result,
                "tool": tool_name,
                "task_type": task_type
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }

# Test Example 3
print("Example 3 - Conditional Tool Routing:\n")
agent = ConditionalToolAgent([calculator_tool, web_search_tool, time_tool])

# Route different task types
tasks = [
    ("10 + 5 * 2", "arithmetic"),
    ("latest AI breakthroughs", "search"),
    ("current time", "time")
]

for task, task_type in tasks:
    result = await agent.route_and_execute(task, task_type)
    print(f"✓ Success: {result['success']}")
    print(f"  Result: {str(result.get('result', 'N/A'))[:60]}...\n")


# ======================================================================
# ## Key Takeaways
# **Tool Use Fundamentals:**
# 1. Define tools: name, description, input schema, function
# 2. Agent reasons if tool needed
# ======================================================================
