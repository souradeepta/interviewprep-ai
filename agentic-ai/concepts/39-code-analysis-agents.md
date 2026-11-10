# Code Analysis Agents
**TL;DR:** Static analyzers find mechanically defined defects; an analysis agent adds repository context, explains a likely defect, and can propose a patch for review.

## Prerequisites
AST parsing, unit testing, and tool-use boundaries.

## Learning Outcomes
- Separate deterministic lint and SAST checks from semantic review.
- Design findings with evidence, severity, confidence, and file-span provenance.
- Measure precision and reviewer time on a labeled repository sample.

## Core Intuition
A code analysis agent is a junior reviewer with a searchable codebase and a strict evidence rule: every claim must point to a line and a reproducible path. A linter can say a value is unused with near certainty; an agent may infer that a check is missing only after following callers, tests, and data flow. That distinction determines whether the output is a blocking diagnostic or a hypothesis for a human.

## How It Works
The pipeline starts by pinning a commit and calculating the exact files within scope. It parses files into symbols, extracts bounded context around a candidate, and may follow imports or call edges under a token and depth budget. Deterministic tools run first because they provide stable diagnostics and reduce model work. The agent then proposes a finding with a location, concrete path from input to effect, and confidence. A separate validator checks that the cited file and lines exist at the pinned revision; invalid evidence is rejected. Suggested patches are diffs and never silently applied.
Repository-level findings are harder than single-file explanations. A security claim may require tracing untrusted input through a wrapper to a sink, checking whether sanitization occurs, and considering whether the vulnerable path is reachable. Context retrieval should be relevance bounded: include definitions, callers, tests, and configuration that bear on the claim, not a repository dump. Preserve the revision identifier because line numbers and data flow change after each commit. For generated or vendored files, classify ownership before asking the model to recommend edits.
For an analysis finding, separate observation from interpretation: “this call accepts an unchecked path” is evidence; “an attacker can read secrets” is an impact claim that needs a reachable source and a sensitive sink. Rank by severity only after reachability and exploitability are checked. Deduplicate repeated patterns so maintainers see one representative trace and a count, not hundreds of near-identical alerts. A developer dismissal should carry a reason code that can improve retrieval or policy without automatically suppressing future reports.
Track precision among reviewed findings, actionable-finding rate, duplicate rate, and reviewer minutes per accepted issue. Recall requires a seeded benchmark or expert audit because dismissed reports alone cannot reveal missed bugs. Slice results by language, rule family, repository size, and finding severity; a high overall precision can conceal poor recall for security-critical categories. Compare against a deterministic baseline and require useful incremental findings before expanding the agent’s tool permissions.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Static checks first | Reproducible, cheap findings | Cannot infer product intent |
| Agent explanation | Connects code paths and tests | Can invent causal links |
| Patch proposal | Shortens repair loop | Review and regression risk |
| Human approval | Controls repository mutation | Adds queue time |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def finding(path, line, evidence, severity):
    if not path or line < 1 or not evidence:
        raise ValueError('finding requires source evidence')
    return {'path': path, 'line': line, 'evidence': evidence, 'severity': severity}

assert finding('src/app.py', 12, 'unchecked input reaches query', 'high')['line'] == 12
```

## Failure Modes
- Stale line evidence: revalidate file spans against the pinned commit and discard stale findings.
- False semantic claim: request a minimal data-flow trace and reproduce it with a test before raising severity.
- Patch breaks an invariant: run targeted tests and static checks in an isolated worktree; require human approval before merge.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| When does an LLM add value over a SAST tool? | When a finding depends on repository context or explanation; deterministic rules remain the first pass. Design follow-up: how will you measure incremental precision? |
| What must a finding contain? | A pinned revision, exact span, observed evidence, impact hypothesis, confidence, and a falsifiable next step. Debug follow-up: what do you do when the cited line moved? |
| How do you prevent dangerous edits? | Return a patch for review in an isolated branch and run targeted tests; do not grant direct write or merge authority. |

## Related Concepts
See [Coding agents](38-coding-agents.md), [agent debugging](27-agent-debugging.md), [agent testing](28-agent-testing.md).

## References
- [semgrep.dev](https://semgrep.dev/docs/)
- [arxiv.org](https://arxiv.org/abs/2308.00352)

## Further Topic Detail
## Core Intuition
A code analysis agent is a junior reviewer with a searchable codebase and a strict evidence rule: every claim must point to a line and a reproducible path. A linter can say a value is unused with near certainty; an agent may infer that a check is missing only after following callers, tests, and data flow. That distinction determines whether the output is a blocking diagnostic or a hypothesis for a human.


## How It Works
The pipeline starts by pinning a commit and calculating the exact files within scope. It parses files into symbols, extracts bounded context around a candidate, and may follow imports or call edges under a token and depth budget. Deterministic tools run first because they provide stable diagnostics and reduce model work. The agent then proposes a finding with a location, concrete path from input to effect, and confidence. A separate validator checks that the cited file and lines exist at the pinned revision; invalid evidence is rejected. Suggested patches are diffs and never silently applied.
Repository-level findings are harder than single-file explanations. A security claim may require tracing untrusted input through a wrapper to a sink, checking whether sanitization occurs, and considering whether the vulnerable path is reachable. Context retrieval should be relevance bounded: include definitions, callers, tests, and configuration that bear on the claim, not a repository dump. Preserve the revision identifier because line numbers and data flow change after each commit. For generated or vendored files, classify ownership before asking the model to recommend edits.
For an analysis finding, separate observation from interpretation: “this call accepts an unchecked path” is evidence; “an attacker can read secrets” is an impact claim that needs a reachable source and a sensitive sink. Rank by severity only after reachability and exploitability are checked. Deduplicate repeated patterns so maintainers see one representative trace and a count, not hundreds of near-identical alerts. A developer dismissal should carry a reason code that can improve retrieval or policy without automatically suppressing future reports.
Track precision among reviewed findings, actionable-finding rate, duplicate rate, and reviewer minutes per accepted issue. Recall requires a seeded benchmark or expert audit because dismissed reports alone cannot reveal missed bugs. Slice results by language, rule family, repository size, and finding severity; a high overall precision can conceal poor recall for security-critical categories. Compare against a deterministic baseline and require useful incremental findings before expanding the agent’s tool permissions.


## Detailed Explanation

Code analysis agents examine code: bug detection, optimization suggestions, refactoring recommendations, test generation. Mechanisms: (1) parse code, (2) apply rules/models, (3) suggest fixes. Advantages: tireless, consistent, catches patterns humans miss. Challenges: context matters (same pattern is bug in one context, feature in another), false positives (too many bad suggestions = ignored). Best for: security scanning, style checking, performance analysis, test generation.



## Interview Q&A

**Q: How do you scope a code analysis agent to avoid analyzing code it shouldn't touch?**
A: Define the analysis boundary explicitly: which files/directories are in scope, which patterns to include or exclude, maximum file size/complexity to analyze. Use gitignore-style patterns for exclusion. Implement hard limits on the number of files and total tokens analyzed per run. For security analysis, explicitly define the threat model (e.g., only analyze for OWASP Top 10, not theoretical vulnerabilities). Document scope decisions in the analysis report.

**Q: What is the difference between static analysis tools and LLM-based code analysis, and when do you use each?**
A: Static analysis (AST-based): deterministic, fast, low false positive rate for known patterns, can't understand context or intent. LLM analysis: understands code intent and complex patterns, handles novel issues static tools miss, but has higher false positive rate and is slower. Use static analysis first to catch obvious issues (linters, SAST tools), then LLM analysis for complex semantic issues (business logic bugs, architectural problems). Don't replace static analysis with LLMs—use them together.

**Q: How do you handle false positives in code analysis agent output?**
A: Track false positive rate by category: when developers dismiss findings without fixing, log the dismissal reason. Train a classifier on accepted vs. dismissed findings to predict false positives. Set confidence thresholds: only report findings above a certainty threshold (e.g., 70%). Implement suppression comments in code for known acceptable patterns. Review false positive rate in sprint retrospectives—systematic false positives indicate prompt or tool problems.

**Q: What context does a code analysis agent need to produce actionable findings?**
A: File content and surrounding code context (not just the flagged line), git history for the file (when was this introduced, by whom), existing tests that exercise the flagged code, related issues or PRs that modified this code, and the codebase's existing conventions and style. Without this context, findings often suggest solutions that are already impossible given other constraints. The agent should explicitly state which context it used to reach each finding.

**Q: How do you prioritize code analysis findings when there are hundreds of issues?**
A: Score by: severity (security vulnerability > correctness bug > performance issue > style), exploitability (reachable from user input vs. internal only), blast radius (how much code is affected), fixability (estimated effort to fix). Present findings in priority order. Cluster related findings: 50 instances of the same pattern should be one finding with "50 occurrences." Auto-assign to likely owners based on git blame. Focus code review time on high-severity, high-confidence findings first.

**Q: How do you ensure a code analysis agent's findings are explainable to developers?**
A: Each finding must include: the specific code location, what the agent detected and why it's a problem, a concrete example of how the issue could manifest, a suggested fix with code snippet, and links to relevant documentation or standards. Test explainability by having a developer unfamiliar with the issue read the finding and implement the fix—if they can't do it in 10 minutes without additional research, the finding needs more detail.
