# Model registry

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
- Store model artifacts, metadata, and metrics together as a single versioned unit
- Implement promotion workflows: experiment → staging → production with approval gates
- Use tags for searchability (model type, task, dataset, team)
- Automate metric comparison before promotion — require improvement over current prod
- Integrate registry with CI/CD for automatic registration on successful training runs
- Store model cards and bias evaluation alongside model artifacts
- Keep registry access audited — log all reads, writes, and promotions

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
