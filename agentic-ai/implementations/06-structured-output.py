"""
Auto-generated from 06-structured-output.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Structured Output
# Objectives: Schema definition, output validation, type safety, composition of structured outputs
# ======================================================================

import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from jsonschema import validate, ValidationError

# Level 1: Basic Structured Output with Validation

class SimpleStructuredOutput:
    def __init__(self):
        self.schema = {
            'type': 'object',
            'properties': {
                'decision': {'type': 'string', 'enum': ['approve', 'reject']},
                'confidence': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                'reasoning': {'type': 'string'}
            },
            'required': ['decision', 'confidence', 'reasoning']
        }
    
    def validate(self, output_text: str) -> tuple[bool, Any, str]:
        """Validate LLM output against schema."""
        try:
            # Parse JSON
            data = json.loads(output_text)
            
            # Validate against schema
            validate(instance=data, schema=self.schema)
            return True, data, ""
        
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)[:50]}"
        
        except ValidationError as e:
            return False, None, f"Schema validation failed: {e.message}"
    
    def parse(self, output_text: str) -> Dict:
        """Parse and validate LLM output."""
        valid, data, error = self.validate(output_text)
        
        if valid:
            return {'success': True, 'data': data}
        else:
            return {'success': False, 'error': error}

# Test Level 1
print('Level 1 - Basic Structured Output Validation:\n')
validator = SimpleStructuredOutput()

# Valid output
valid_output = json.dumps({
    'decision': 'approve',
    'confidence': 0.92,
    'reasoning': 'Good risk profile'
})
print(f'Valid output: {valid_output}')
result = validator.parse(valid_output)
print(f'Result: {result}\n')

# Invalid output (wrong enum value)
invalid_output = json.dumps({
    'decision': 'maybe',  # Invalid: not in ['approve', 'reject']
    'confidence': 0.85,
    'reasoning': 'Uncertain'
})
print(f'Invalid output: {invalid_output}')
result = validator.parse(invalid_output)
print(f'Result: {result}\n')


# Level 2: Pydantic-Based Structured Output with Type Safety

from pydantic import BaseModel, Field, field_validator

class DecisionOutput(BaseModel):
    """Structured output for approval decisions."""
    decision: str = Field(..., description="approve or reject")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    reasoning: str = Field(..., description="Why this decision")
    alternative: Optional[str] = Field(None, description="Alternative approach")
    
    @field_validator('decision')
    @classmethod
    def validate_decision(cls, v):
        if v not in ['approve', 'reject']:
            raise ValueError(f'decision must be approve or reject, got {v}')
        return v

class PydanticStructuredOutput:
    def parse(self, llm_output: str) -> Dict:
        """Parse LLM output using Pydantic validation."""
        try:
            # Parse JSON
            data = json.loads(llm_output)
            # Validate with Pydantic
            result = DecisionOutput(**data)
            return {
                'success': True,
                'data': result.model_dump()
            }
        
        except json.JSONDecodeError as e:
            return {'success': False, 'error': f'Invalid JSON: {str(e)[:50]}'}
        
        except ValueError as e:
            return {'success': False, 'error': f'Validation failed: {str(e)[:100]}'}

# Test Level 2
print('Level 2 - Pydantic Structured Output:\n')
parser = PydanticStructuredOutput()

# Valid output
valid = json.dumps({
    'decision': 'approve',
    'confidence': 0.95,
    'reasoning': 'Strong indicators',
    'alternative': None
})
print(f'Valid: {valid}')
result = parser.parse(valid)
print(f'✓ Parsed: {result["success"]}\n')

# Invalid output (confidence > 1.0)
invalid = json.dumps({
    'decision': 'approve',
    'confidence': 1.5,  # Invalid: > 1.0
    'reasoning': 'Too confident'
})
print(f'Invalid: {invalid}')
result = parser.parse(invalid)
print(f'✗ Error: {result["error"]}\n')


# Example 1: Composition of Structured Outputs

class AnalysisOutput(BaseModel):
    user_id: int
    risk_score: float = Field(..., ge=0.0, le=1.0)
    analysis: str

class ApprovalOutput(BaseModel):
    user_id: int  # From analysis
    decision: str = Field(..., pattern='^(approve|reject)$')
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str

class CompositionExample:
    def analyze_user(self, user_id: int) -> Dict:
        """Step 1: Analyze user."""
        # Simulate LLM output
        llm_output = json.dumps({
            'user_id': user_id,
            'risk_score': 0.3,
            'analysis': 'Low risk profile'
        })
        
        try:
            analysis = AnalysisOutput(**json.loads(llm_output))
            return {'success': True, 'data': analysis}
        except ValueError as e:
            return {'success': False, 'error': str(e)}
    
    def make_decision(self, analysis: AnalysisOutput) -> Dict:
        """Step 2: Make decision based on analysis."""
        # Simulate LLM output using analysis data
        decision_prompt = f"User {analysis.user_id} has risk {analysis.risk_score}"
        
        llm_output = json.dumps({
            'user_id': analysis.user_id,  # Input from analysis
            'decision': 'approve' if analysis.risk_score < 0.5 else 'reject',
            'confidence': 0.92,
            'reason': 'Based on analysis'
        })
        
        try:
            approval = ApprovalOutput(**json.loads(llm_output))
            return {'success': True, 'data': approval}
        except ValueError as e:
            return {'success': False, 'error': str(e)}

# Test Example 1
print('Example 1 - Composition of Structured Outputs:\n')
composer = CompositionExample()

# Step 1: Analyze
analysis_result = composer.analyze_user(user_id=123)
if analysis_result['success']:
    analysis = analysis_result['data']
    print(f'✓ Analysis: risk={analysis.risk_score}\n')
    
    # Step 2: Decide
    decision_result = composer.make_decision(analysis)
    if decision_result['success']:
        approval = decision_result['data']
        print(f'✓ Decision: {approval.decision} (confidence={approval.confidence})\n')


# Example 2: Retry with Structured Output

class RobustStructuredOutput:
    def __init__(self, schema_class, max_retries: int = 3):
        self.schema_class = schema_class
        self.max_retries = max_retries
    
    def get_with_retry(self, llm_responses: list) -> Dict:
        """Attempt to parse structured output, retry on failure."""
        
        for attempt, response in enumerate(llm_responses[:self.max_retries]):
            try:
                print(f'Attempt {attempt + 1}: Parsing {str(response)[:60]}...')
                data = json.loads(response)
                result = self.schema_class(**data)
                print(f'  ✓ Success')
                return {'success': True, 'data': result}
            
            except (json.JSONDecodeError, ValueError) as e:
                error_msg = str(e)[:40]
                print(f'  ✗ Failed: {error_msg}')
                if attempt < self.max_retries - 1:
                    print(f'  → Retrying (attempt {attempt + 2})...')
        
        return {'success': False, 'error': 'Max retries exceeded'}

# Test Example 2
print('\nExample 2 - Retry on Validation Failure:\n')
retrier = RobustStructuredOutput(DecisionOutput, max_retries=3)

# Simulate LLM retry attempts
llm_attempts = [
    '{"decision": "maybe", "confidence": 0.8}',  # Missing reasoning, invalid decision
    '{"decision": "approve", "confidence": 1.5}',  # Confidence out of range
    '{"decision": "approve", "confidence": 0.88, "reasoning": "Good score"}'  # Valid
]

result = retrier.get_with_retry(llm_attempts)
if result['success']:
    print(f'\n✓ Final: {result["data"].decision}')
else:
    print(f'\n✗ Final: {result["error"]}')


# ======================================================================
# ## Key Takeaways
# **Schema-Based Output Validation:**
# 1. Define schema (JSON Schema, Pydantic, TypeScript)
# 2. Communicate to LLM (prompt + examples)
# ======================================================================
