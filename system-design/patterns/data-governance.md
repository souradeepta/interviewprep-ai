# Data governance

## TL;DR
Core ML system design pattern for production.

## Core Intuition
[Intuitive explanation]

## How It Works
[Technical details]

## Key Properties / Trade-offs
- Property 1
- Property 2

## Common Mistakes / Gotchas
- Mistake 1
- Mistake 2

## Best Practices
- Classify all data assets by sensitivity (PII, confidential, public) before building pipelines
- Document data lineage from source through transformations to model features
- Implement access controls at the data layer, not just application layer
- Audit all data access for compliance — log who accessed what and when
- Define data retention policies and automate deletion
- Version datasets like code — immutable snapshots with metadata
- Conduct regular data quality audits and alert on schema drift

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
