# Coding Agents

## Detailed Explanation

Coding agents generate, execute, and refine code autonomously. Capabilities: code generation (write Python, JavaScript, etc.), execution in sandboxed environment, testing and validation, debugging errors, iterative refinement. Advantages: automate programming (boilerplate generation, bug fixes), assist developers, educational tool. Challenges: hallucinated code (looks right, doesn't work), security (executing LLM output is risky), sandbox limitations (restricted environment). Best for: code generation from specs, automated testing, learning programming, data transformation.

## Core Intuition

A pair programmer who writes, tests, and refixes code. Describe what you want, they generate code, run it, see if it works, fix if needed, deliver working solution.

## How It Works

Generate → Test → Validate → Iterate cycle:

1. **Specification** — User provides requirements
2. **Generation** — Agent generates code
3. **Execution** — Run in sandbox environment
4. **Testing** — Validate with test cases
5. **Iteration** — Debug and refine if needed
6. **Delivery** — Return working code

## Architecture / Trade-offs

**Language:** Python (easiest) vs JavaScript vs multi-language
**Execution:** Sandboxed (safe, limited) vs unrestricted (powerful, risky)
**Validation:** Syntax checking, type checking (mypy, TypeScript), test-driven

## Interview Q&A

**Q: What are the most important safety constraints for a coding agent running in production?**
A: Sandbox execution: run all generated code in isolated environments (containers, VMs) with no network access, limited file system access, and resource limits. Review before execution: for irreversible operations (database writes, file deletions, API calls with side effects), require explicit confirmation. Allowlist/blocklist: define which commands and libraries the agent can use. Rate limiting: prevent infinite loops or resource exhaustion. Never run generated code with production credentials until it has been reviewed.

**Q: How do you evaluate a coding agent's output quality automatically?**
A: Use test execution: run the generated code against a test suite and measure pass rate. For new functionality, generate tests first (TDD approach) and measure whether the code passes them. Static analysis: run linters, type checkers, and security scanners on generated code. Functional correctness: compare output of generated code against expected output on benchmark inputs. Track regression rate: does new code break existing tests?

**Q: What is the best approach to handle long coding tasks that span multiple files?**
A: Use a hierarchical planning approach: first create a high-level plan (which files to modify, what interfaces to create), then execute changes file by file. Maintain a context window that includes the current file plus relevant related files (interfaces, types it uses). After each file change, run the test suite to catch immediate regressions. Use a dependency graph to determine which files depend on changed files and proactively check them. For very long tasks, checkpoint progress and allow resumption.

**Q: How do you design a coding agent that can learn from code review feedback?**
A: Collect code review comments with accept/reject decisions and the reason. Fine-tune the agent on accepted vs. rejected changes as preference pairs (DPO). Categorize common review patterns: missing error handling, insufficient tests, naming inconsistencies—add these as explicit constraints in the system prompt or retrieval knowledge base. Track code review acceptance rate over time as the primary quality metric.

**Q: When should a coding agent refuse to implement a requested feature?**
A: Refuse when: the request requires introducing known security vulnerabilities (SQL injection patterns, unvalidated inputs), bypassing existing safety checks, implementing features that violate compliance requirements, or when the scope is unclear enough that implementation would require making unverifiable assumptions. The agent should explain why it's refusing and what information or clarification would allow it to proceed.

**Q: How do you handle tool use in a coding agent (file reads, code execution, web search)?**
A: Design tool use as atomic operations with clear inputs/outputs. For file operations: read before write to avoid clobber. For code execution: capture stdout/stderr, detect hangs (timeout), parse structured output (JSON) rather than free text. For search: cache results to avoid duplicate queries. Rate limit all external tools to prevent runaway API usage. Log all tool calls for debugging—the tool call trace is essential for understanding agent failures.


## Best Practices

1. Sandboxed execution always
2. Code review before production
3. Test-driven generation
4. Type hints for constraints
5. Error feedback for iteration
6. Guardrails on what code can do
7. Version control for generated code

## Code Examples

### Example 1: Simple Generator

```python
def generate_function(spec):
    code = f"""
def solution(data):
    '''Generated from: {spec}'''
    return data
"""
    return code
```

### Example 2: With Testing

```python
def execute_and_test(code, test_cases):
    local_scope = {}
    exec(code, local_scope)
    func = local_scope["solution"]
    
    for inputs, expected in test_cases:
        result = func(*inputs)
        assert result == expected, f"Got {result}, expected {expected}"
    
    return True
```

### Example 3: Iterative Refinement

```python
class CodingAgent:
    def generate_and_refine(self, spec, test_cases, max_iterations=3):
        for i in range(max_iterations):
            code = self._generate(spec)
            if self._test(code, test_cases):
                return code
            spec += f"\\n# Attempt {i+1} failed, revise"
        return None
```

## Related Concepts

- Error Recovery, Agent Loops, Tool Use, Observability
