# Feature importance tracking

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
- Track feature importances across model versions to detect distribution shift in inputs
- Alert when a previously unimportant feature becomes top-ranked — often indicates data issues
- Combine multiple importance methods (permutation, SHAP, gradient) for robustness
- Track feature importance stability across cross-validation folds
- Use importance tracking to guide feature engineering for the next model iteration
- Visualize importance trends over time, not just point-in-time values
- Separate importance from model version metadata

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
