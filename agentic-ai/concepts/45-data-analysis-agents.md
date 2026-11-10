# Data Analysis Agents
**TL;DR:** A data analysis agent should inspect a constrained dataset, run reproducible calculations, and report results with code, assumptions, and provenance.

## Prerequisites
Agent loops, tool boundaries, and evaluation.

## Learning Outcomes
- Explain the main analysis sandbox workflow and its state boundary.
- Compare an automated path with review or fallback using measured outcomes.
- Diagnose a representative failure using trace evidence.

## Core Intuition
Data Analysis Agents is best understood as a controlled workflow with explicit evidence and a bounded decision. The system should expose what it knows, what it is allowed to do, and when a person or deterministic checker must take over. This framing prevents a fluent answer from being confused with a verified outcome.

## How It Works
Validate schema and units before analysis, restrict executable code to a sandbox, and expose only approved tables or columns. The result should include query or code, row filters, missing-data treatment, and a distinction between observed association and causal claims. A chart is not evidence if its aggregation or denominator is hidden.
## Detailed Explanation

Data analysis agents explore data: SQL generation, statistical analysis, anomaly detection, insight generation. Mechanisms: (1) table understanding (schema), (2) query generation, (3) result interpretation. Advantages: fast exploration, pattern finding, automated insights. Challenges: SQL injection, hallucination (made-up columns), context (what does data mean?). Best for: BI, data exploration, automated reporting, anomaly alerts.


## Interview Q&A

**Q: What are the most common errors a data analysis agent makes and how do you catch them?**
A: Common errors: incorrect joins causing row multiplication, confusing count vs. count distinct, aggregating before filtering (wrong denominator), ignoring NULL values in averages, time zone errors in datetime columns, and misinterpreting column names. Catch with: code review by a data engineer for generated queries, automated tests on expected output shapes (number of rows should be less than source table), unit tests with synthetic data with known answers, and a human sanity-check for any metric that changed >20% from previous analysis.

**Q: How do you prevent a data analysis agent from accessing data it shouldn't?**
A: Implement data access controls at the database level (row-level security, column masking), not just at the agent prompt level. The agent should connect to the database with a service account that has only SELECT permissions on authorized tables. Maintain a schema registry that defines which tables/columns the agent can access, filtered by user role. Audit all queries run by the agent. For sensitive columns (PII, financial), require explicit justification in the analysis request before granting access.

**Q: What makes a data analysis agent's output trustworthy vs. untrustworthy?**
A: Trustworthy: shows the SQL/code used to generate the result (verifiable), includes sample rows to validate the interpretation, states assumptions explicitly (e.g., "excluding nulls"), provides confidence intervals or uncertainty ranges where appropriate, and references the data source and freshness. Untrustworthy: presents conclusions without showing methodology, rounds numbers suspiciously, doesn't acknowledge data limitations, or presents uncertain conclusions as definitive facts. Always show your work.

**Q: How do you handle requests for analysis that the available data cannot reliably answer?**
A: Be explicit about the limitation rather than generating a plausible-sounding but unreliable answer. State: "The available data cannot answer this question reliably because [reason]." Offer alternatives: "I can answer the related question [X] which may proxy for what you need." Suggest data collection: "To answer this properly, you would need to instrument [event] in your system." This is better than generating spurious correlations or extrapolating beyond the data's support.

**Q: How do you design a data analysis agent for iterative analysis vs. one-shot queries?**
A: For iterative analysis: maintain conversation state including previously run queries and their results, allow the user to refine questions naturally, offer suggestions for follow-up analysis based on current results, and maintain a notebook-like interface where each step builds on previous steps. For one-shot queries: optimize for clarity of output, include all context needed to interpret the result without reference to prior conversation, and produce a self-contained report. The key difference is context management and output format.

**Q: What happens when a data analysis agent's conclusion contradicts the user's hypothesis?**
A: Present the contradicting finding clearly and objectively—don't soften data to match expectations. Provide additional context: check whether the methodology is sound (was the hypothesis well-defined?), whether the time period is appropriate, whether there are confounding factors. Offer to investigate deeper: "The data shows X, which contradicts the hypothesis. Would you like me to investigate whether [confounding factor] explains this?" Data integrity requires honest reporting even when the result is unexpected.
Measure numerical correctness against reference queries, reproducibility, unsupported-claim rate, privacy leakage, and analyst review time. Test missing values, duplicate rows, unit changes, and empty subsets.

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
def validate_columns(requested, allowlist):
    unknown = set(requested) - set(allowlist)
    if unknown:
        raise ValueError(f'unapproved columns: {sorted(unknown)}')
    return requested

assert validate_columns(['region'], ['region', 'sales']) == ['region']
```

## Failure Modes
- Data Analysis Agents uses stale or incomplete input: log source and timestamp, validate freshness, and request a refresh before acting.
- Data Analysis Agents produces an unsupported result: require evidence and a domain-specific validator; route unresolved cases to review.
- Data Analysis Agents exceeds its resource or authority boundary: enforce explicit budgets and permissions in the controller, then report the stopped state.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| What are the most common errors a data analysis agent makes and how do you catch them? | Common errors: incorrect joins causing row multiplication, confusing count vs. count distinct, aggregating before filtering (wrong denominator), ignoring NULL values in averages, time zone errors in datetime columns, and misinterpreting column names. Catch with: code review by a data engineer for generated queries, automated tests on expected output shapes (number of rows should be less than source table), unit tests with synthetic data with known answers, and a human sanity-check for any metric that changed >20% from previous analysis. |
| How do you prevent a data analysis agent from accessing data it shouldn't? | Implement data access controls at the database level (row-level security, column masking), not just at the agent prompt level. The agent should connect to the database with a service account that has only SELECT permissions on authorized tables. Maintain a schema registry that defines which tables/columns the agent can access, filtered by user role. Audit all queries run by the agent. For sensitive columns (PII, financial), require explicit justification in the analysis request before granting access. |
| What makes a data analysis agent's output trustworthy vs. untrustworthy? | Trustworthy: shows the SQL/code used to generate the result (verifiable), includes sample rows to validate the interpretation, states assumptions explicitly (e.g., "excluding nulls"), provides confidence intervals or uncertainty ranges where appropriate, and references the data source and freshness. Untrustworthy: presents conclusions without showing methodology, rounds numbers suspiciously, doesn't acknowledge data limitations, or presents uncertain conclusions as definitive facts. Always show your work. |

## Related Concepts
See [agent testing](28-agent-testing.md), [agent evaluation](29-agent-evals.md), and [agent monitoring](30-agent-monitoring.md).

## References
- [www.nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)
- [opentelemetry.io](https://opentelemetry.io/docs/)

## Further Topic Detail
## Detailed Explanation

Data analysis agents explore data: SQL generation, statistical analysis, anomaly detection, insight generation. Mechanisms: (1) table understanding (schema), (2) query generation, (3) result interpretation. Advantages: fast exploration, pattern finding, automated insights. Challenges: SQL injection, hallucination (made-up columns), context (what does data mean?). Best for: BI, data exploration, automated reporting, anomaly alerts.


## Interview Q&A

**Q: What are the most common errors a data analysis agent makes and how do you catch them?**
A: Common errors: incorrect joins causing row multiplication, confusing count vs. count distinct, aggregating before filtering (wrong denominator), ignoring NULL values in averages, time zone errors in datetime columns, and misinterpreting column names. Catch with: code review by a data engineer for generated queries, automated tests on expected output shapes (number of rows should be less than source table), unit tests with synthetic data with known answers, and a human sanity-check for any metric that changed >20% from previous analysis.

**Q: How do you prevent a data analysis agent from accessing data it shouldn't?**
A: Implement data access controls at the database level (row-level security, column masking), not just at the agent prompt level. The agent should connect to the database with a service account that has only SELECT permissions on authorized tables. Maintain a schema registry that defines which tables/columns the agent can access, filtered by user role. Audit all queries run by the agent. For sensitive columns (PII, financial), require explicit justification in the analysis request before granting access.

**Q: What makes a data analysis agent's output trustworthy vs. untrustworthy?**
A: Trustworthy: shows the SQL/code used to generate the result (verifiable), includes sample rows to validate the interpretation, states assumptions explicitly (e.g., "excluding nulls"), provides confidence intervals or uncertainty ranges where appropriate, and references the data source and freshness. Untrustworthy: presents conclusions without showing methodology, rounds numbers suspiciously, doesn't acknowledge data limitations, or presents uncertain conclusions as definitive facts. Always show your work.

**Q: How do you handle requests for analysis that the available data cannot reliably answer?**
A: Be explicit about the limitation rather than generating a plausible-sounding but unreliable answer. State: "The available data cannot answer this question reliably because [reason]." Offer alternatives: "I can answer the related question [X] which may proxy for what you need." Suggest data collection: "To answer this properly, you would need to instrument [event] in your system." This is better than generating spurious correlations or extrapolating beyond the data's support.

**Q: How do you design a data analysis agent for iterative analysis vs. one-shot queries?**
A: For iterative analysis: maintain conversation state including previously run queries and their results, allow the user to refine questions naturally, offer suggestions for follow-up analysis based on current results, and maintain a notebook-like interface where each step builds on previous steps. For one-shot queries: optimize for clarity of output, include all context needed to interpret the result without reference to prior conversation, and produce a self-contained report. The key difference is context management and output format.

**Q: What happens when a data analysis agent's conclusion contradicts the user's hypothesis?**
A: Present the contradicting finding clearly and objectively—don't soften data to match expectations. Provide additional context: check whether the methodology is sound (was the hypothesis well-defined?), whether the time period is appropriate, whether there are confounding factors. Offer to investigate deeper: "The data shows X, which contradicts the hypothesis. Would you like me to investigate whether [confounding factor] explains this?" Data integrity requires honest reporting even when the result is unexpected.
