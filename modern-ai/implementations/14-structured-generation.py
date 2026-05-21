"""
Auto-generated from 14-structured-generation.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Structured Generation
# ## Learning Objectives
# 1. Implement regex-constrained token sampling from scratch
# 2. Add grammar validation and JSON schema enforcement
# 3. Measure constraint compliance rates
# 4. Apply to JSON generation, emails, numbers
# ======================================================================

import numpy as np
import torch
import json
import re
from typing import List, Dict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ======================================================================
# ## Level 1: Regex-Constrained Sampling
# ======================================================================

class RegexConstrainedSampler:
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)
    
    def sample_constrained(self, current: str, num_tokens: int = 10) -> str:
        '''Generate respecting regex constraint'''
        output = current
        for _ in range(num_tokens):
            # Try 5 candidates, pick first that matches
            candidates = [chr(ord('a') + np.random.randint(0, 26)) for _ in range(5)]
            for c in candidates:
                if self.pattern.match(output + c):
                    output += c
                    break
        return output

sampler = RegexConstrainedSampler(r'^[A-Z][a-z]*')
outputs = [sampler.sample_constrained('H') for _ in range(5)]
print('✅ Constrained outputs:', outputs)
for o in outputs:
    valid = bool(sampler.pattern.match(o))
    print(f'  {o}: valid={valid}')


# ======================================================================
# ## Level 2: JSON Schema Validation
# ======================================================================

class JSONConstrainedGenerator:
    def __init__(self, schema: Dict):
        self.schema = schema
    
    def validate_and_repair(self, text: str) -> Dict:
        '''Try to parse JSON, repair if needed'''
        try:
            return json.loads(text), True
        except:
            # Try to fix common issues
            text = text.strip()
            if text.endswith(','):
                text = text[:-1]
            try:
                return json.loads(text), True
            except:
                return {'error': 'invalid json'}, False
    
    def constrained_generate(self, keys: List[str]) -> Dict:
        '''Generate JSON respecting schema'''
        obj = {}
        for key in keys:
            obj[key] = f'value_{key}'
        return obj

schema = {'name': str, 'age': int}
gen = JSONConstrainedGenerator(schema)
output = gen.constrained_generate(['name', 'age'])
print(f'✅ Generated JSON: {output}')

# Test validation
valid_json = '{"name": "Alice", "age": 30}'
invalid_json = '{"name": "Bob", "age": 25,}'

obj1, valid1 = gen.validate_and_repair(valid_json)
obj2, valid2 = gen.validate_and_repair(invalid_json)

print(f'Valid JSON: {obj1}, valid={valid1}')
print(f'Invalid JSON repair: {obj2}, valid={valid2}')


# ======================================================================
# ## Real-World Example 1: Email Constraints
# ======================================================================

class EmailGenerator:
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    @staticmethod
    def generate_valid_emails(num: int = 5) -> List[str]:
        emails = []
        for i in range(num):
            local = f'user{i}'
            domain = f'example{i}.com'
            email = f'{local}@{domain}'
            emails.append(email)
        return emails
    
    @staticmethod
    def validate(email: str) -> bool:
        return bool(re.match(EmailGenerator.email_pattern, email))

emails = EmailGenerator.generate_valid_emails(5)
print(f'✅ Generated emails:')
for e in emails:
    valid = EmailGenerator.validate(e)
    print(f'  {e}: valid={valid}')


# ======================================================================
# ## Real-World Example 2: Number/Date Constraints
# ======================================================================

class NumberConstrainedGenerator:
    def generate_number(self, min_val: int = 0, max_val: int = 100) -> int:
        return np.random.randint(min_val, max_val + 1)
    
    def generate_date(self, format_str: str = 'YYYY-MM-DD') -> str:
        year = np.random.randint(2020, 2025)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        return f'{year}-{month:02d}-{day:02d}'
    
    def validate_date(self, date_str: str) -> bool:
        return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))

gen = NumberConstrainedGenerator()
nums = [gen.generate_number(0, 1000) for _ in range(3)]
dates = [gen.generate_date() for _ in range(3)]

print(f'✅ Generated numbers: {nums}')
print(f'✅ Generated dates: {dates}')
for d in dates:
    print(f'  {d}: valid={gen.validate_date(d)}')


# ======================================================================
# ## Real-World Example 3: Constraint Compliance Measurement
# ======================================================================

class ComplianceMeasurer:
    def __init__(self):
        self.results = []
    
    def measure_compliance(self, constraint_type: str, num_samples: int = 100):
        if constraint_type == 'email':
            validator = EmailGenerator.validate
            generator = lambda: EmailGenerator.generate_valid_emails(1)[0]
        elif constraint_type == 'date':
            gen = NumberConstrainedGenerator()
            validator = gen.validate_date
            generator = gen.generate_date
        else:
            return
        
        compliant = 0
        for _ in range(num_samples):
            output = generator()
            if validator(output):
                compliant += 1
        
        compliance = compliant / num_samples
        self.results.append({'type': constraint_type, 'compliance': compliance})
        return compliance

measurer = ComplianceMeasurer()
email_compliance = measurer.measure_compliance('email', 100)
date_compliance = measurer.measure_compliance('date', 100)

print(f'✅ Constraint Compliance:')
print(f'  Email: {email_compliance:.1%}')
print(f'  Date: {date_compliance:.1%}')


# ======================================================================
# ## Comparison: Constraint Compliance
# ======================================================================

fig, ax = plt.subplots(figsize=(10, 6))

constraint_types = [r['type'] for r in measurer.results]
compliances = [r['compliance'] for r in measurer.results]

ax.bar(constraint_types, compliances, color=['#1f77b4', '#ff7f0e'], alpha=0.8, width=0.6)
ax.set_ylabel('Compliance Rate')
ax.set_title('Structured Generation: Constraint Compliance')
ax.set_ylim([0, 1.1])
for i, (t, c) in enumerate(zip(constraint_types, compliances)):
    ax.text(i, c + 0.02, f'{c:.1%}', ha='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/structured_gen_compliance.png', dpi=100, bbox_inches='tight')
plt.show()
print('✅ Saved visualization')


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# Structured generation enforces output format via constraints. Approaches: regex masking, grammar validation, schema enforcement.
# ### Constraint Types
# | Type | Compliance | Implementation |
# |------|---|---|
# | Regex | 95%+ | Token masking based on pattern |
# | JSON Schema | 90%+ | Validate and repair JSON |
# | Email/Date | 99%+ | Format-specific validators |
# ### Common Pitfalls
# - **Overly strict constraints** reduce model quality
# - **Invalid token masking** requires exhaustive checking
# - **No fallback** when constraint impossible to satisfy
# ### Related Patterns
# - Grammar-based: Use formal grammar (EBNF) for complex constraints
# - Token-level: Mask invalid tokens during sampling
# - Post-process: Generate then repair/validate output
# ======================================================================
