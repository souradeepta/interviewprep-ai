# Federated learning

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
- Apply differential privacy at the client level to protect individual contributions
- Use secure aggregation to prevent server from seeing individual client updates
- Implement client selection strategies to handle heterogeneous data and compute
- Use FedAvg with momentum for faster convergence
- Communicate only model deltas, not full weights, to reduce bandwidth
- Monitor for Byzantine clients and implement robust aggregation
- Test with heterogeneous data (non-IID) distributions — federated data is never IID

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
