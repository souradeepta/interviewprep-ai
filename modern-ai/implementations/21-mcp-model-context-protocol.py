"""
Auto-generated from 21-mcp-model-context-protocol.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Model Context Protocol (MCP)
# ## Learning Objectives
# 1. Understand MCP server architecture with resource registry and tool discovery
# 2. Implement request/response marshaling with schema validation
# 3. Build clients that call MCP tools and handle responses
# 4. Test error handling and versioning in distributed tool systems
# ======================================================================

# Prerequisites & Imports
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import time
from abc import ABC, abstractmethod

# Device/config setup
print("MCP (Model Context Protocol) Implementation")
print(f"Python asyncio runtime ready")
print(f"Session ID: {uuid.uuid4().hex[:8]}")


# ======================================================================
# ## Level 1: Basic MCP Server with Resource Registry
# ======================================================================

# Level 1: Minimal MCP Server - Resource Registry + Tool Discovery

@dataclass
class MCPTool:
    """Represents a tool available in MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    version: str = "1.0"

class BasicMCPServer:
    """Minimal MCP server with tool registry."""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.server_id = uuid.uuid4().hex[:8]
    
    def register_tool(self, tool: MCPTool):
        """Register a tool in the server."""
        self.tools[tool.name] = tool
        print(f"✓ Registered tool: {tool.name}")
    
    def discover_tools(self) -> List[Dict]:
        """Return list of available tools (schema, description)."""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'input_schema': tool.input_schema,
                'version': tool.version
            }
            for tool in self.tools.values()
        ]
    
    def call_tool(self, tool_name: str, args: Dict) -> Dict:
        """Execute a registered tool."""
        if tool_name not in self.tools:
            return {'error': f'Tool {tool_name} not found'}
        return {'result': f'Called {tool_name} with {list(args.keys())}'}

# Test Level 1
server = BasicMCPServer()

# Register sample tools
calculator = MCPTool(
    name='add',
    description='Add two numbers',
    input_schema={
        'type': 'object',
        'properties': {
            'a': {'type': 'number'},
            'b': {'type': 'number'}
        },
        'required': ['a', 'b']
    }
)

search = MCPTool(
    name='search',
    description='Search the web',
    input_schema={
        'type': 'object',
        'properties': {
            'query': {'type': 'string'}
        },
        'required': ['query']
    }
)

server.register_tool(calculator)
server.register_tool(search)

# Tool discovery
tools = server.discover_tools()
print(f"\nDiscovered {len(tools)} tools:")
for tool in tools:
    print(f"  • {tool['name']}: {tool['description']}")

# Call a tool
result = server.call_tool('add', {'a': 5, 'b': 3})
print(f"\nTool call result: {result}")


# ======================================================================
# ## Level 2: Advanced MCP with Request/Response Marshaling, Versioning, Error Handling
# ======================================================================

# Level 2: Production MCP Server with Schema Validation, Versioning, Error Handling

class MCPRequest:
    """Marshaled request format."""
    def __init__(self, request_id: str, method: str, params: Dict):
        self.request_id = request_id
        self.method = method
        self.params = params
        self.timestamp = time.time()
    
    def to_json(self) -> str:
        return json.dumps({
            'jsonrpc': '2.0',
            'id': self.request_id,
            'method': self.method,
            'params': self.params
        })

class MCPResponse:
    """Marshaled response format."""
    def __init__(self, request_id: str, result: Any = None, error: str = None):
        self.request_id = request_id
        self.result = result
        self.error = error
        self.timestamp = time.time()
    
    def to_json(self) -> str:
        resp = {'jsonrpc': '2.0', 'id': self.request_id}
        if self.error:
            resp['error'] = {'code': -32603, 'message': self.error}
        else:
            resp['result'] = self.result
        return json.dumps(resp)

class AdvancedMCPServer:
    """Production MCP server with versioning and error handling."""
    
    PROTOCOL_VERSION = '1.0.0'
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.request_count = 0
        self.error_count = 0
    
    def register_tool(self, tool: MCPTool):
        self.tools[tool.name] = tool
    
    def validate_input(self, tool_name: str, args: Dict) -> Optional[str]:
        """Validate args against tool's input schema."""
        if tool_name not in self.tools:
            return f'Tool {tool_name} not found'
        
        tool = self.tools[tool_name]
        required_fields = tool.input_schema.get('required', [])
        
        for field in required_fields:
            if field not in args:
                return f'Missing required field: {field}'
        
        return None
    
    def call_tool(self, tool_name: str, args: Dict) -> Any:
        """Execute tool with validation."""
        if tool_name == 'add':
            return {'sum': args['a'] + args['b']}
        elif tool_name == 'multiply':
            return {'product': args['x'] * args['y']}
        elif tool_name == 'translate':
            return {'translated': f"{args['text']} (translated to {args['language']})"}
        return {'error': 'Unknown tool'}
    
    async def process_request(self, request_json: str) -> str:
        """Process incoming JSON-RPC request."""
        try:
            self.request_count += 1
            
            req_data = json.loads(request_json)
            request = MCPRequest(
                request_id=req_data['id'],
                method=req_data['method'],
                params=req_data.get('params', {})
            )
            
            # Validate
            validation_error = self.validate_input(request.method, request.params)
            if validation_error:
                self.error_count += 1
                response = MCPResponse(request.request_id, error=validation_error)
                return response.to_json()
            
            # Execute
            result = self.call_tool(request.method, request.params)
            response = MCPResponse(request.request_id, result=result)
            return response.to_json()
        
        except json.JSONDecodeError as e:
            self.error_count += 1
            return json.dumps({
                'jsonrpc': '2.0',
                'error': {'code': -32700, 'message': 'Parse error'}
            })
        except Exception as e:
            self.error_count += 1
            return json.dumps({
                'jsonrpc': '2.0',
                'error': {'code': -32603, 'message': str(e)}
            })

# Test Level 2
server = AdvancedMCPServer()

# Register tools with different schemas
server.register_tool(MCPTool(
    name='add',
    description='Add two numbers',
    input_schema={'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}}, 'required': ['a', 'b']}
))

server.register_tool(MCPTool(
    name='multiply',
    description='Multiply two numbers',
    input_schema={'type': 'object', 'properties': {'x': {'type': 'number'}, 'y': {'type': 'number'}}, 'required': ['x', 'y']}
))

server.register_tool(MCPTool(
    name='translate',
    description='Translate text',
    input_schema={'type': 'object', 'properties': {'text': {'type': 'string'}, 'language': {'type': 'string'}}, 'required': ['text', 'language']}
))

# Test valid requests
print("Testing valid requests:")
valid_req = json.dumps({
    'jsonrpc': '2.0',
    'id': '1',
    'method': 'add',
    'params': {'a': 10, 'b': 5}
})
result = await server.process_request(valid_req)
print(f"Response: {result}")

# Test invalid request (missing required param)
print("\nTesting invalid request (missing param):")
invalid_req = json.dumps({
    'jsonrpc': '2.0',
    'id': '2',
    'method': 'add',
    'params': {'a': 10}
})
result = await server.process_request(invalid_req)
print(f"Response: {result}")

# Test malformed JSON
print("\nTesting malformed JSON:")
result = await server.process_request('{invalid json')
print(f"Response: {result}")

print(f"\nServer stats: {server.request_count} requests, {server.error_count} errors")


# ======================================================================
# ## Real-World Example 1: MCP Server with 3 Tools
# ======================================================================

# Example 1: Full MCP Server with 3 Production Tools

class ProductionMCPServer(AdvancedMCPServer):
    """Server with realistic tool implementations."""
    
    def __init__(self):
        super().__init__()
        self.call_history = []
    
    def call_tool(self, tool_name: str, args: Dict) -> Any:
        """Execute tool with realistic behavior."""
        result = None
        
        if tool_name == 'calculator_eval':
            # Safe calculator: supports +, -, *, /
            try:
                a, b, op = args['a'], args['b'], args['operation']
                if op == '+':
                    result = {'value': a + b}
                elif op == '-':
                    result = {'value': a - b}
                elif op == '*':
                    result = {'value': a * b}
                elif op == '/':
                    if b == 0:
                        result = {'error': 'Division by zero'}
                    else:
                        result = {'value': a / b}
                else:
                    result = {'error': f'Unknown operation: {op}'}
            except Exception as e:
                result = {'error': str(e)}
        
        elif tool_name == 'web_search':
            # Mock web search
            query = args['query']
            result = {
                'query': query,
                'results': [
                    {'title': f'Result 1 for {query[:20]}', 'url': 'https://example.com/1'},
                    {'title': f'Result 2 for {query[:20]}', 'url': 'https://example.com/2'}
                ]
            }
        
        elif tool_name == 'translate_text':
            # Mock translator
            text = args['text']
            lang = args['target_language']
            result = {
                'original': text,
                'translated': f'[{lang.upper()}] {text}',
                'confidence': 0.95
            }
        
        else:
            result = {'error': f'Unknown tool: {tool_name}'}
        
        # Log call
        self.call_history.append({
            'tool': tool_name,
            'args': args,
            'result': result,
            'timestamp': time.time()
        })
        
        return result

# Setup server
server = ProductionMCPServer()

# Register tools
server.register_tool(MCPTool(
    name='calculator_eval',
    description='Evaluate arithmetic expression',
    input_schema={'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}, 'operation': {'type': 'string'}}, 'required': ['a', 'b', 'operation']}
))

server.register_tool(MCPTool(
    name='web_search',
    description='Search the web',
    input_schema={'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}
))

server.register_tool(MCPTool(
    name='translate_text',
    description='Translate text to target language',
    input_schema={'type': 'object', 'properties': {'text': {'type': 'string'}, 'target_language': {'type': 'string'}}, 'required': ['text', 'target_language']}
))

# Test calls
print("Example 1: MCP Server with 3 Tools\n")

# Test 1: Calculator
req1 = json.dumps({'jsonrpc': '2.0', 'id': '1', 'method': 'calculator_eval', 'params': {'a': 42, 'b': 8, 'operation': '*'}})
resp1 = await server.process_request(req1)
print(f"Calculator: 42 * 8 = {json.loads(resp1)['result']}")

# Test 2: Search
req2 = json.dumps({'jsonrpc': '2.0', 'id': '2', 'method': 'web_search', 'params': {'query': 'machine learning optimization'}})
resp2 = await server.process_request(req2)
result = json.loads(resp2)['result']
print(f"\nWeb Search: Found {len(result['results'])} results")

# Test 3: Translate
req3 = json.dumps({'jsonrpc': '2.0', 'id': '3', 'method': 'translate_text', 'params': {'text': 'Hello, world!', 'target_language': 'Spanish'}})
resp3 = await server.process_request(req3)
result = json.loads(resp3)['result']
print(f"\nTranslate: '{result['original']}' → '{result['translated']}'")

print(f"\nServer processed {server.request_count} requests, {len(server.call_history)} tool calls")


# ======================================================================
# ## Real-World Example 2: MCP Client that Calls Tools
# ======================================================================

# Example 2: MCP Client that Discovers and Calls Tools

class MCPClient:
    """Client that connects to MCP server and calls tools."""
    
    def __init__(self, server: AdvancedMCPServer):
        self.server = server
        self.request_id = 0
        self.tool_cache = None
    
    async def discover(self) -> List[Dict]:
        """Discover available tools from server."""
        self.tool_cache = self.server.tools
        return [
            {'name': t.name, 'description': t.description}
            for t in self.tool_cache.values()
        ]
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict:
        """Call a tool on the server."""
        self.request_id += 1
        
        request = {
            'jsonrpc': '2.0',
            'id': str(self.request_id),
            'method': tool_name,
            'params': kwargs
        }
        
        response_json = await self.server.process_request(json.dumps(request))
        response = json.loads(response_json)
        
        return response
    
    async def multi_tool_sequence(self, steps: List[Dict]) -> List[Any]:
        """Execute a sequence of tool calls (like a workflow)."""
        results = []
        for step in steps:
            tool_name = step['tool']
            params = step['params']
            
            response = await self.call_tool(tool_name, **params)
            
            if 'result' in response:
                results.append(response['result'])
                print(f"✓ {tool_name}: {response['result']}")
            else:
                error_msg = response.get('error', {}).get('message', 'Unknown error')
                print(f"✗ {tool_name}: {error_msg}")
                results.append(None)
        
        return results

# Setup client
client = MCPClient(server)

print("Example 2: MCP Client\n")

# Discover tools
tools = await client.discover()
print(f"Discovered {len(tools)} tools:")
for tool in tools:
    print(f"  - {tool['name']}: {tool['description']}")

# Execute multi-step workflow
print("\nExecuting workflow:")
workflow = [
    {'tool': 'calculator_eval', 'params': {'a': 100, 'b': 5, 'operation': '/'}},
    {'tool': 'translate_text', 'params': {'text': 'The answer is 20', 'target_language': 'French'}},
    {'tool': 'web_search', 'params': {'query': 'distributed systems'}}
]
results = await client.multi_tool_sequence(workflow)
print(f"\nCompleted workflow with {len([r for r in results if r])} successful calls")


# ======================================================================
# ## Real-World Example 3: Error Scenarios and Recovery
# ======================================================================

# Example 3: Error Handling and Recovery Strategies

class ResilientMCPClient(MCPClient):
    """Client with retry logic and error recovery."""
    
    def __init__(self, server: AdvancedMCPServer, max_retries: int = 3):
        super().__init__(server)
        self.max_retries = max_retries
        self.failed_calls = []
    
    async def call_tool_with_retry(self, tool_name: str, max_retries: int = None, **kwargs) -> Optional[Dict]:
        """Call tool with exponential backoff retry."""
        max_retries = max_retries or self.max_retries
        
        for attempt in range(max_retries):
            try:
                response = await self.call_tool(tool_name, **kwargs)
                
                # Check if response has error
                if 'error' in response:
                    error_msg = response['error'].get('message', 'Unknown error')
                    
                    # Retryable errors
                    if 'timeout' in error_msg.lower() and attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  ⏳ Retry attempt {attempt + 1}/{max_retries} after {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Non-retryable error
                    print(f"  ❌ Error: {error_msg}")
                    self.failed_calls.append({'tool': tool_name, 'error': error_msg})
                    return None
                
                return response['result']
            
            except Exception as e:
                print(f"  ⚠️  Exception: {str(e)[:50]}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        return None
    
    async def call_tool_with_fallback(self, tool_name: str, fallback_tool: str, **kwargs) -> Optional[Dict]:
        """Call tool, fallback to alternative if primary fails."""
        result = await self.call_tool_with_retry(tool_name, max_retries=2, **kwargs)
        
        if result is None:
            print(f"  → Falling back to {fallback_tool}")
            result = await self.call_tool_with_retry(fallback_tool, max_retries=2, **kwargs)
        
        return result

# Test error scenarios
print("Example 3: Error Handling and Recovery\n")

client = ResilientMCPClient(server, max_retries=3)

# Scenario 1: Invalid tool (non-retryable)
print("Scenario 1: Invalid tool")
result = await client.call_tool_with_retry('nonexistent_tool', x=1, y=2)
print(f"Result: {result}\n")

# Scenario 2: Missing required field
print("Scenario 2: Missing required field")
result = await client.call_tool_with_retry('calculator_eval', a=10)
print(f"Result: {result}\n")

# Scenario 3: Division by zero (handled by tool)
print("Scenario 3: Division by zero")
result = await client.call_tool_with_retry('calculator_eval', a=10, b=0, operation='/')
print(f"Result: {result}\n")

# Scenario 4: Valid call
print("Scenario 4: Valid call")
result = await client.call_tool_with_retry('calculator_eval', a=100, b=25, operation='*')
print(f"Result: {result}")

print(f"\nClient stats: {len(client.failed_calls)} failed calls")


# ======================================================================
# ## Comparison & Benchmarking
# ======================================================================

# Benchmark: Server response time by tool type
import matplotlib.pyplot as plt
import numpy as np

# Simulate server processing times
tool_names = ['calculator_eval', 'web_search', 'translate_text']
response_times = [0.005, 0.050, 0.030]
throughput = [1000, 200, 300]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Response time comparison
colors = ['#2ecc71', '#3498db', '#e74c3c']
ax1.bar(tool_names, response_times, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Response Time (seconds)', fontsize=11)
ax1.set_title('MCP Tool Response Times', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 0.06)
for i, v in enumerate(response_times):
    ax1.text(i, v + 0.002, f'{v*1000:.1f}ms', ha='center', fontsize=10)

# Throughput comparison
ax2.bar(tool_names, throughput, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Throughput (req/s)', fontsize=11)
ax2.set_title('MCP Tool Throughput', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1200)
for i, v in enumerate(throughput):
    ax2.text(i, v + 50, f'{v}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Error breakdown
error_types = ['Parse Error', 'Validation Error', 'Tool Error', 'Timeout']
error_counts = [5, 12, 3, 2]

plt.figure(figsize=(8, 5))
wedges, texts, autotexts = plt.pie(error_counts, labels=error_types, autopct='%1.1f%%', colors=['#e74c3c', '#f39c12', '#e67e22', '#c0392b'], startangle=90)
plt.title('Error Distribution in MCP Server', fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
plt.tight_layout()
plt.show()

print(f"Total errors by type:")
for error_type, count in zip(error_types, error_counts):
    print(f"  {error_type}: {count}")
print(f"\nFastest tool: {tool_names[0]} ({response_times[0]*1000:.1f}ms)")
print(f"Highest throughput: {tool_names[0]} ({throughput[0]} req/s)")


# ======================================================================
# ## Key Takeaways
# **MCP Architecture:**
# 1. Server maintains tool registry with metadata (schema, description, version)
# 2. Client discovers tools, then calls them via JSON-RPC 2.0
# 3. Request/response marshaling enables versioning and schema evolution
# **Error Handling Strategy:**
# - Parse errors: Client sent invalid JSON
# - Validation errors: Missing required fields or type mismatches
# - Tool errors: Execution failed (division by zero, timeout, etc.)
# - Retryable vs non-retryable: Distinguish timeout/transient from permanent failures
# **Production Patterns:**
# - Tool discovery before calling (prevents invalid tool names)
# - Schema validation per tool (fail fast on bad input)
# - Exponential backoff retry (for transient failures)
# - Fallback strategies (primary → alternative tool)
# - Call history tracking (observability, debugging)
# **When to use MCP:**
# - Model needs access to external tools (search, calculator, API)
# - Multiple clients may call same tools (centralized server)
# - Tool schema evolution required (versioning support)
# - Complex error scenarios (validation, timeouts, retries)
# **Related Concepts:** [[tool-use]], [[function-calling]], [[agent-architecture]], [[distributed-systems]]
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Extend Example 1:** Add a 4th tool (e.g., format_json or validate_url) with custom error handling
# 2. **Modify Example 2:** Implement tool chaining where output of one tool feeds into another
# 3. **Enhance Example 3:** Add circuit breaker pattern to prevent cascading failures
# ======================================================================
