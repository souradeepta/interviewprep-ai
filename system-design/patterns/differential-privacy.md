# Differential privacy

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
- Start with epsilon=1.0 as a baseline and tighten based on privacy requirements
- Use RDP (Rényi Differential Privacy) accountant for tight composition bounds
- Clip gradients before adding noise — gradient norm must be bounded
- Apply DP at the model level, not the dataset level, for training
- Use DP-SGD (opacus library for PyTorch) for end-to-end differentially private training
- Publish epsilon values alongside model performance metrics
- Test that privacy guarantees hold with membership inference attack evaluation

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
