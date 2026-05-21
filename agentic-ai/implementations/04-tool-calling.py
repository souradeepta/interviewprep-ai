"""
Auto-generated from 04-tool-calling.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Tool Calling
# Objectives: Structured tool invocation formats, parsing patterns, parameter validation, fallback on syntax errors
# ======================================================================

import json
import re
from typing import Dict, List, Any

# Level 1: XML-Based Tool Calling Parser

class XMLToolCallParser:
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, name: str, fn: callable):
        self.tools[name] = fn
    
    def parse_xml_calls(self, llm_output: str) -> List[Dict]:
        """Parse XML-formatted tool calls from LLM output."""
        calls = []
        
        # Regex to find <invoke name="tool_name">...</invoke>
        pattern = r'<invoke\s+name=["\']([^"\']+ )["\']>(.*?)</invoke>'
        matches = re.findall(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        
        for tool_name, content in matches:
            # Extract parameters
            params = {}
            param_pattern = r'<parameter\s+name=["\']([^"\']+ )["\'][>]?([^<]*)</parameter>'
            param_matches = re.findall(param_pattern, content)
            
            for param_name, param_value in param_matches:
                params[param_name.strip()] = param_value.strip()
            
            calls.append({
                'tool': tool_name.strip(),
                'params': params
            })
        
        return calls
    
    def execute(self, calls: List[Dict]) -> List[Dict]:
        """Execute parsed tool calls."""
        results = []
        
        for call in calls:
            tool_name = call.get('tool')
            params = call.get('params', {})
            
            if tool_name not in self.tools:
                results.append({
                    'success': False,
                    'error': f'Tool not found: {tool_name}'
                })
                continue
            
            try:
                result = self.tools[tool_name](**params)
                results.append({
                    'success': True,
                    'tool': tool_name,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'tool': tool_name,
                    'error': str(e)
                })
        
        return results

# Register tools
parser = XMLToolCallParser()
parser.register_tool('calculator', lambda expression: eval(expression))
parser.register_tool('web_search', lambda query: f'Results for {query}')

# Test Level 1
print('Level 1 - XML Tool Calling:\n')
llm_output = '''<invoke name="calculator">
<parameter name="expression">10 + 5</parameter>
</invoke>'''

print(f'LLM Output:\n{llm_output}\n')
calls = parser.parse_xml_calls(llm_output)
print(f'Parsed calls: {calls}')
results = parser.execute(calls)
print(f'Execution results: {results}\n')


# Level 2: JSON-Based Tool Calling with Schema Validation

class JSONToolCallValidator:
    def __init__(self, tool_schemas: Dict[str, Dict]):
        self.schemas = tool_schemas
        self.tools = {}
    
    def register_tool(self, name: str, fn: callable):
        self.tools[name] = fn
    
    def validate_params(self, tool_name: str, params: Dict) -> tuple[bool, str]:
        """Validate parameters against schema."""
        if tool_name not in self.schemas:
            return False, f'Tool schema not found: {tool_name}'
        
        schema = self.schemas[tool_name]
        required = schema.get('required', [])
        properties = schema.get('properties', {})
        
        # Check required params
        for param in required:
            if param not in params:
                return False, f'Missing required parameter: {param}'
        
        # Check param types
        for param, value in params.items():
            if param in properties:
                expected_type = properties[param].get('type')
                # Simple type check
                if expected_type == 'string' and not isinstance(value, str):
                    return False, f'Parameter {param} must be string, got {type(value).__name__}'
        
        return True, ''
    
    def parse_and_execute(self, llm_output: str) -> Dict:
        """Parse JSON tool call, validate, execute."""
        try:
            # Extract JSON
            call_data = json.loads(llm_output)
            tool_name = call_data.get('tool')
            params = call_data.get('params', {})
            
            # Validate
            valid, error = self.validate_params(tool_name, params)
            if not valid:
                return {'success': False, 'error': f'Validation failed: {error}'}
            
            # Execute
            if tool_name not in self.tools:
                return {'success': False, 'error': f'Tool not found: {tool_name}'}
            
            result = self.tools[tool_name](**params)
            return {
                'success': True,
                'tool': tool_name,
                'result': result
            }
        
        except json.JSONDecodeError as e:
            return {'success': False, 'error': f'Invalid JSON: {str(e)[:50]}'}
        except Exception as e:
            return {'success': False, 'error': f'Execution error: {str(e)}'}

# Define schemas
schemas = {
    'calculator': {
        'type': 'object',
        'properties': {'expression': {'type': 'string'}},
        'required': ['expression']
    },
    'web_search': {
        'type': 'object',
        'properties': {'query': {'type': 'string'}},
        'required': ['query']
    }
}

validator = JSONToolCallValidator(schemas)
validator.register_tool('calculator', lambda expression: eval(expression))
validator.register_tool('web_search', lambda query: f'Found results for {query}')

# Test Level 2
print('\nLevel 2 - JSON Tool Calling with Validation:\n')

# Valid call
valid_call = json.dumps({
    'tool': 'calculator',
    'params': {'expression': '20 * 3'}
})
print(f'Valid call: {valid_call}')
result = validator.parse_and_execute(valid_call)
print(f'Result: {result}\n')

# Invalid call (missing required param)
invalid_call = json.dumps({
    'tool': 'calculator',
    'params': {}  # Missing expression
})
print(f'Invalid call: {invalid_call}')
result = validator.parse_and_execute(invalid_call)
print(f'Result: {result}\n')


# Example 1: Parallel Tool Calls (Multiple tools in one response)

class ParallelToolCaller:
    def __init__(self, validator):
        self.validator = validator
    
    def parse_multiple_calls(self, llm_output: str) -> List[Dict]:
        """Parse multiple JSON tool calls from LLM output."""
        calls = []
        
        # Try to parse as JSON array
        try:
            data = json.loads(llm_output)
            if isinstance(data, list):
                calls = data
            elif isinstance(data, dict):
                calls = [data]
        except json.JSONDecodeError:
            # Try regex to extract multiple JSON objects
            import re
            pattern = r'\{"tool"\s*:\s*["\']([^"\']+ )["\'][^}]*\}'
            matches = re.findall(pattern, llm_output)
            # Simplified; real implementation would be more robust
        
        return calls
    
    def execute_parallel(self, calls: List[Dict]) -> Dict:
        """Execute multiple tool calls (could be in parallel)."""
        results = []
        
        for call in calls:
            tool_name = call.get('tool')
            params = call.get('params', {})
            
            # Validate
            valid, error = self.validator.validate_params(tool_name, params)
            if not valid:
                results.append({'tool': tool_name, 'success': False, 'error': error})
                continue
            
            # Execute (in real system: parallelize with asyncio)
            if tool_name not in self.validator.tools:
                results.append({'tool': tool_name, 'success': False, 'error': 'Tool not found'})
                continue
            
            try:
                result = self.validator.tools[tool_name](**params)
                results.append({'tool': tool_name, 'success': True, 'result': result})
            except Exception as e:
                results.append({'tool': tool_name, 'success': False, 'error': str(e)})
        
        return {'success': all(r['success'] for r in results), 'results': results}

# Test Example 1
print('Example 1 - Parallel Tool Calls:\n')
caller = ParallelToolCaller(validator)
parallel_calls = json.dumps([
    {'tool': 'calculator', 'params': {'expression': '10 + 5'}},
    {'tool': 'web_search', 'params': {'query': 'AI trends'}}
])

print(f'Multiple calls: {parallel_calls}')
calls = json.loads(parallel_calls)
result = caller.execute_parallel(calls)
print(f'Results: {json.dumps(result, indent=2)}\n')


# Example 2: Retry on Parse/Validation Failure

class RobustToolCaller:
    def __init__(self, validator, max_retries: int = 3):
        self.validator = validator
        self.max_retries = max_retries
    
    def call_with_retry(self, initial_prompt: str, llm_responses: List[str]) -> Dict:
        """Attempt to parse and execute tool calls, retry on failure."""
        
        for attempt, response in enumerate(llm_responses[:self.max_retries]):
            print(f'Attempt {attempt + 1}:')
            
            # Try to parse
            try:
                result = self.validator.parse_and_execute(response)
                if result['success']:
                    print(f'  ✓ Success: {result}')
                    return result
                else:
                    print(f'  ✗ {result["error"]}')
                    if attempt < self.max_retries - 1:
                        print(f'  → Retrying with corrected format...')
            except Exception as e:
                print(f'  ✗ Exception: {str(e)[:50]}')
        
        return {'success': False, 'error': 'Max retries exceeded'}

# Test Example 2
print('Example 2 - Retry on Failure:\n')
caller = RobustToolCaller(validator, max_retries=3)

# Simulate LLM retry attempts
llm_attempts = [
    '{"tool": "calculator"}',  # Missing params (fails validation)
    '{"tool": "calculator", "params": {"expr": "5+3"}}',  # Wrong param name
    '{"tool": "calculator", "params": {"expression": "5+3"}}'  # Correct
]

result = caller.call_with_retry('Calculate 5+3', llm_attempts)
print(f'\nFinal result: {result}')


# ======================================================================
# ## Key Takeaways
# **Tool Calling Formats:**
# - XML: hierarchical, clear structure, verbose
# - JSON: lightweight, standard, less verbose
# ======================================================================
