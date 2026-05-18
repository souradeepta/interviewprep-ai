# Interpretability

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
- Choose interpretability method matching model complexity — LIME for local, SHAP for global
- Always validate feature importances against domain knowledge — spurious correlations look important
- Use SHAP summary plots for global importance, waterfall plots for individual predictions
- Test explanations on adversarial examples to verify faithfulness
- Provide explanations in the user's language, not ML jargon
- Document which features are most influential for regulatory compliance
- Combine model-agnostic methods with model-specific interpretations

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
