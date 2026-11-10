# Coding Agents
**TL;DR:** A coding agent reads a bounded repository context and proposes changes that must pass deterministic checks and human review.

## Prerequisites
Agent loops, tool boundaries, and evaluation.

## Learning Outcomes
- Explain the main patch workflow workflow and its state boundary.
- Compare an automated path with review or fallback using measured outcomes.
- Diagnose a representative failure using trace evidence.

## Core Intuition
Coding Agents is best understood as a controlled workflow with explicit evidence and a bounded decision. The system should expose what it knows, what it is allowed to do, and when a person or deterministic checker must take over. This framing prevents a fluent answer from being confused with a verified outcome.

## How It Works
Pin a commit, scope files, inspect relevant symbols and tests, propose a diff, and run tests in an isolated worktree. Separate read and write capabilities. The model can explain or draft a change, but the test runner and reviewer provide independent evidence that the patch preserves behavior.
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
Measure accepted patch rate, test pass rate, regression rate, review time, and cost per merged change. Slice by task and repository; a high patch acceptance average can conceal unsafe dependency or security edits.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Deterministic baseline | Predictable behavior and low cost | Limited coverage of ambiguous cases |
| Model-assisted path | Handles flexible inputs | Requires evidence checks and evaluation |
| Human review | Resolves high-impact uncertainty | Capacity and response-time limits |
| Fallback or abstention | Bounds unsafe behavior | May defer useful work |

## Implementation
This runnable Python example shows a bounded policy gate relevant to the workflow. It models control logic only; it does not claim production quality.

```python
def validate_patch(changed_paths, allowed_paths, tests_pass):
    in_scope = set(changed_paths) <= set(allowed_paths)
    return in_scope and tests_pass

assert validate_patch(['src/a.py'], ['src/a.py', 'tests/'], True)
assert not validate_patch(['secrets.env'], ['src/a.py'], True)
```

## Failure Modes
- Coding Agents uses stale or incomplete input: log source and timestamp, validate freshness, and request a refresh before acting.
- Coding Agents produces an unsupported result: require evidence and a domain-specific validator; route unresolved cases to review.
- Coding Agents exceeds its resource or authority boundary: enforce explicit budgets and permissions in the controller, then report the stopped state.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| What are the most important safety constraints for a coding agent running in production? | Sandbox execution: run all generated code in isolated environments (containers, VMs) with no network access, limited file system access, and resource limits. Review before execution: for irreversible operations (database writes, file deletions, API calls with side effects), require explicit confirmation. Allowlist/blocklist: define which commands and libraries the agent can use. Rate limiting: prevent infinite loops or resource exhaustion. Never run generated code with production credentials until it has been reviewed. |
| How do you evaluate a coding agent's output quality automatically? | Use test execution: run the generated code against a test suite and measure pass rate. For new functionality, generate tests first (TDD approach) and measure whether the code passes them. Static analysis: run linters, type checkers, and security scanners on generated code. Functional correctness: compare output of generated code against expected output on benchmark inputs. Track regression rate: does new code break existing tests? |
| What is the best approach to handle long coding tasks that span multiple files? | Use a hierarchical planning approach: first create a high-level plan (which files to modify, what interfaces to create), then execute changes file by file. Maintain a context window that includes the current file plus relevant related files (interfaces, types it uses). After each file change, run the test suite to catch immediate regressions. Use a dependency graph to determine which files depend on changed files and proactively check them. For very long tasks, checkpoint progress and allow resumption. |

## Related Concepts
See [agent testing](28-agent-testing.md), [agent evaluation](29-agent-evals.md), and [agent monitoring](30-agent-monitoring.md).

## References
- [www.nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)
- [opentelemetry.io](https://opentelemetry.io/docs/)

## Further Topic Detail
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
