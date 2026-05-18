# Fairness metrics

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
- Measure fairness across multiple metrics simultaneously — no single metric captures all notions of fairness
- Use disaggregated evaluation by subgroup before and after model training
- Apply fairness constraints during training when post-processing is insufficient
- Document which fairness definition was optimized and why
- Use Aequitas or Fairlearn libraries for systematic fairness auditing
- Monitor fairness metrics in production — distribution shift can re-introduce bias
- Involve domain experts and affected communities in defining fairness criteria

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
