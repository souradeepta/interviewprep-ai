# Data Analysis Agents

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


## Best Practices

1. Schema validation
2. Query sandboxing
3. Confidence scores
4. Human review for decisions
5. Audit logs
6. Error handling
7. Performance monitoring
8. Metadata tracking

## Code Examples

```python
class DataAnalysisAgent:
    def __init__(self, database_schema):
        self.schema = database_schema
    
    def generate_query(self, question):
        # Validate schema tables/columns exist
        query = self._llm_generate_query(question, self.schema)
        validated = self._validate_query(query)
        if not validated:
            return None
        return query
    
    def analyze(self, question):
        query = self.generate_query(question)
        if not query:
            return {'error': 'Could not generate valid query'}
        
        results = self._execute(query)
        insights = self._interpret(results)
        return insights
```

## Related Concepts

- Retrieval-Augmented Generation, Error Recovery, Monitoring
