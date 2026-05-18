# Bias detection

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
- Define protected attributes and fairness criteria before building models
- Measure bias on held-out test set, not training data
- Disaggregate metrics by subgroup combinations (intersectionality), not just single attributes
- Use statistical tests to determine if performance differences are significant
- Run bias evaluation as part of CI/CD pipeline — not one-time audits
- Monitor bias metrics in production — distribution shift causes bias to reappear
- Involve domain experts in interpreting bias metrics — not all disparities are problematic

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
