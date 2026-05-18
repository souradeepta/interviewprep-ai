# Code Analysis Agents

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


## Best Practices

1. High precision (few false positives)
2. Explainable suggestions
3. Fix auto-generation (optional)
4. Integration with IDE
5. Severity levels
6. Suppression mechanism
7. False positive tracking
8. Continuous model improvement

## Code Examples

```python
class CodeAnalyzer:
    def detect_bugs(self, code):
        ast = self._parse(code)
        issues = []
        # Rule-based checks
        if self._has_unhandled_exception(ast):
            issues.append({'type': 'bug', 'severity': 'high'})
        # ML-based anomaly detection
        anomalies = self._detect_anomalies(ast)
        issues.extend(anomalies)
        return issues
    
    def suggest_optimization(self, code):
        perf_issues = self._profile(code)
        suggestions = []
        for issue in perf_issues:
            suggestion = self._suggest_fix(issue)
            suggestions.append(suggestion)
        return suggestions
```

## Related Concepts

- Coding Agents, Error Recovery, Monitoring
