# Model explainability

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
- Choose explanation method based on stakeholder: SHAP values for data scientists, natural language for end users
- Validate explanations against domain knowledge — implausible explanations signal model issues
- Use local explanations (LIME, SHAP waterfall) for individual predictions, global for model behavior
- Generate explanations at inference time for production use cases requiring regulatory compliance
- Test explanation stability — similar inputs should yield similar explanations
- Include counterfactual explanations ('if X were different, outcome would change')
- Document explanation limitations alongside the explanations themselves

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
