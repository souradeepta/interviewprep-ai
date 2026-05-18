# Privacy preserving ml

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
- Apply differential privacy with epsilon < 1.0 for high-sensitivity data
- Use federated learning when data cannot leave client devices
- Anonymize training data before use — pseudonymization is not anonymization
- Apply privacy budget tracking across multiple model versions
- Use secure aggregation in federated settings to prevent server from seeing individual updates
- Conduct privacy audits using membership inference attack benchmarks
- Encrypt model weights at rest and in transit for sensitive domains

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
