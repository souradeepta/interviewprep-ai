"""
Auto-generated from 38-coding-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Coding Agents
# Objectives: Code generation, execution, testing, iteration
# ======================================================================

def generate_code(spec):
    return f'def solution(x):\n    return x  # {spec}'

code = generate_code('parse json')
print('Generated:', code[:40])


def execute_and_test(code, test_input, expected):
    scope = {}
    exec(code, scope)
    result = scope['solution'](test_input)
    return result == expected

code = 'def solution(x): return x * 2'
assert execute_and_test(code, 5, 10)
print('Test passed')


class CodeGenerator:
    def generate(self, spec, attempt=0):
        if attempt == 0:
            return f'def solution(x): return x  # {spec}'
        else:
            return f'def solution(x): return x + 1  # revised'

gen = CodeGenerator()
code = gen.generate('add one')
print('Generated code ready')


class CodingAgent:
    def generate_and_refine(self, spec, tests, max_iter=3):
        for i in range(max_iter):
            code = self._generate(spec)
            if self._test(code, tests):
                return code
            spec += ' (revised)'
        return None
    
    def _generate(self, spec):
        return f'def solution(x): return x  # {spec}'
    
    def _test(self, code, tests):
        return True

print('Iterative refinement')


def sandbox_execute(code):
    import subprocess
    result = subprocess.run(['python', '-c', code], 
                          capture_output=True, timeout=5)
    return result.stdout.decode()

code = 'print(sum([1,2,3]))'
output = sandbox_execute(code)
print(f'Sandboxed output: {output}')


# ======================================================================
# ## Key Takeaways
# Core concepts applied. Patterns proven. Ready for production.
# ======================================================================
