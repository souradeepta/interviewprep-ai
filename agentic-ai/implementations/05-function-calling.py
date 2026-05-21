"""
Auto-generated from 05-function-calling.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Function Calling
# Objectives: OpenAI function definitions, JSON schema, parallel calls, parameter validation
# ======================================================================

import json
from typing import Dict, List, Any
from jsonschema import validate, ValidationError

# Level 1: Basic Function Definition and Parsing

class OpenAIFunctionCaller:
    def __init__(self):
        # Define functions using OpenAI JSON Schema format
        self.functions = [
            {
                "name": "calculate",
                "description": "Perform arithmetic calculations (e.g., '5+3', '10*4')",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Result limit (default 10)"}
                    },
                    "required": ["query"]
                }
            }
        ]
        self.function_map = {f["name"]: f for f in self.functions}
    
    def parse_function_call(self, llm_response: str) -> Dict:
        """Parse function call from simulated LLM response."""
        try:
            # In real system, extract from OpenAI response object
            data = json.loads(llm_response)
            return {
                "success": True,
                "function_name": data["function_name"],
                "arguments": data["arguments"]
            }
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {str(e)[:50]}"}
    
    def execute(self, function_name: str, arguments: Dict) -> Dict:
        """Execute function with arguments."""
        if function_name == "calculate":
            try:
                result = eval(arguments["expression"])
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif function_name == "web_search":
            query = arguments.get("query")
            limit = arguments.get("limit", 10)
            return {
                "success": True,
                "results": [f"Result {i+1} for {query}" for i in range(min(limit, 3))]
            }
        
        return {"success": False, "error": f"Unknown function: {function_name}"}

# Test Level 1
print('Level 1 - Basic Function Calling:\n')
caller = OpenAIFunctionCaller()

llm_response = json.dumps({
    "function_name": "calculate",
    "arguments": {"expression": "10 + 5"}
})

print(f'LLM Response: {llm_response}')
parsed = caller.parse_function_call(llm_response)

if parsed["success"]:
    result = caller.execute(parsed["function_name"], parsed["arguments"])
    print(f'✓ Executed: {parsed["function_name"]} → {result["result"]}\n')


# Level 2: Function Validation and Error Handling

class ValidatingFunctionCaller:
    def __init__(self, functions: List[Dict]):
        self.functions = {f["name"]: f for f in functions}
    
    def validate_arguments(self, function_name: str, arguments: Dict) -> tuple[bool, str]:
        """Validate arguments against function schema."""
        if function_name not in self.functions:
            return False, f"Function not found: {function_name}"
        
        schema = self.functions[function_name]["parameters"]
        try:
            validate(instance=arguments, schema=schema)
            return True, ""
        except ValidationError as e:
            return False, f"Validation error: {e.message}"
    
    def execute_with_validation(self, function_name: str, arguments: Dict) -> Dict:
        """Execute with full validation."""
        # Validate
        valid, error = self.validate_arguments(function_name, arguments)
        if not valid:
            return {"success": False, "error": error, "function": function_name}
        
        # Execute
        try:
            if function_name == "calculate":
                result = eval(arguments["expression"])
            else:
                result = f"Executed {function_name}"
            
            return {
                "success": True,
                "function": function_name,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "function": function_name
            }

# Test Level 2
print('Level 2 - Function Validation:\n')
caller = ValidatingFunctionCaller(caller.functions)

# Valid call
valid_result = caller.execute_with_validation("calculate", {"expression": "20 * 3"})
print(f'Valid call: {valid_result}')

# Invalid call (missing required parameter)
invalid_result = caller.execute_with_validation("calculate", {})
print(f'Invalid call: {invalid_result}\n')


# Example 1: Parallel Function Calls

class ParallelFunctionCaller:
    def __init__(self, functions: List[Dict]):
        self.functions = {f["name"]: f for f in functions}
    
    def parse_multiple_calls(self, llm_response: str) -> List[Dict]:
        """Parse multiple function calls from LLM response."""
        try:
            data = json.loads(llm_response)
            # LLM returns array of function calls
            if isinstance(data, list):
                return data
            else:
                return [data]
        except json.JSONDecodeError:
            return []
    
    def execute_all(self, calls: List[Dict]) -> Dict:
        """Execute multiple function calls."""
        results = []
        
        print(f'Executing {len(calls)} function calls in parallel:\n')
        
        for call in calls:
            func_name = call.get("function_name")
            args = call.get("arguments", {})
            
            # In real system: parallelize with asyncio
            if func_name == "calculate":
                try:
                    result = eval(args["expression"])
                    results.append({"success": True, "function": func_name, "result": result})
                except Exception as e:
                    results.append({"success": False, "function": func_name, "error": str(e)})
            
            elif func_name == "web_search":
                results.append({
                    "success": True,
                    "function": func_name,
                    "result": f"Results for {args.get('query')}"
                })
            
            print(f"  ✓ {func_name}: success={results[-1]['success']}")
        
        return {"all_success": all(r["success"] for r in results), "results": results}

# Test Example 1
print('Example 1 - Parallel Function Calls:\n')
parallel_caller = ParallelFunctionCaller(caller.functions)

llm_response = json.dumps([
    {"function_name": "calculate", "arguments": {"expression": "5 + 3"}},
    {"function_name": "web_search", "arguments": {"query": "AI trends"}}
])

calls = parallel_caller.parse_multiple_calls(llm_response)
result = parallel_caller.execute_all(calls)
print(f'\nAll successful: {result["all_success"]}\n')


# Example 2: Type Coercion and Flexible Parsing

class RobustFunctionCaller:
    def __init__(self, functions: List[Dict]):
        self.functions = {f["name"]: f for f in functions}
    
    def coerce_types(self, function_name: str, arguments: Dict) -> Dict:
        """Attempt type coercion for robustness."""
        if function_name not in self.functions:
            return arguments
        
        schema = self.functions[function_name]["parameters"]
        properties = schema.get("properties", {})
        coerced = {}
        
        for key, value in arguments.items():
            if key not in properties:
                coerced[key] = value
                continue
            
            expected_type = properties[key].get("type")
            
            # Coerce string to int if needed
            if expected_type == "integer" and isinstance(value, str):
                try:
                    coerced[key] = int(value)
                except ValueError:
                    coerced[key] = value  # Keep as is, let validation fail
            else:
                coerced[key] = value
        
        return coerced
    
    def execute(self, function_name: str, arguments: Dict) -> Dict:
        """Execute with type coercion."""
        # Coerce types
        arguments = self.coerce_types(function_name, arguments)
        
        if function_name == "web_search":
            query = arguments.get("query")
            limit = arguments.get("limit", 10)
            # Limit must be int
            if not isinstance(limit, int):
                limit = 10
            return {
                "success": True,
                "results": [f"Result {i+1}" for i in range(min(limit, 3))]
            }
        return {"success": False, "error": "Unknown function"}

# Test Example 2
print('Example 2 - Type Coercion:\n')
robust = RobustFunctionCaller(caller.functions)

# LLM sends limit as string instead of int
result = robust.execute("web_search", {
    "query": "machine learning",
    "limit": "5"  # String, should be int
})
print(f'✓ Coerced and executed: {result["success"]}\n')


# ======================================================================
# ## Key Takeaways
# **Function Calling Pattern (OpenAI):**
# 1. Define functions with JSON Schema
# 2. Pass to OpenAI chat completion API
# ======================================================================
