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

## Interview Q&A

**Q: What is the difference between model interpretability and model explainability?**
A: Interpretability: understanding why a model makes predictions by examining its internal structure—inherent property of simple models (linear models, decision trees) that humans can directly inspect. Explainability: post-hoc techniques applied to black-box models to approximate their behavior for specific predictions—SHAP, LIME, attention weights. Interpretable models are always explainable; explainable models may not be interpretable (SHAP provides per-prediction explanations but doesn't make a neural network "interpretable"). Use interpretability when regulatory requirements demand it; explainability when debugging or communicating decisions.

**Q: When is SHAP preferable to LIME for explaining individual predictions?**
A: SHAP is preferable when: you need globally consistent feature importance (SHAP values satisfy axioms like efficiency and symmetry that LIME doesn't guarantee), you have tree-based models (TreeSHAP is exact and fast), or you need to aggregate explanations across many predictions. LIME is preferable when: SHAP is too slow (SHAP on deep neural networks requires approximation), you want local linearity (LIME's local linear model is easier to communicate), or you need custom input perturbations for complex input types (text, images with domain-specific perturbations).

**Q: How do you use explainability in production to debug model failures?**
A: For each reported model failure (wrong prediction, customer complaint): compute SHAP values for the failing example, identify which features have unusually high attribution, compare the feature values and attributions to similar correct predictions. Build a "failure explanation dashboard": aggregate explanations for the worst predictions, identify systematic patterns (feature X has unexpectedly high negative attribution in 60% of failures). This turns individual failure analysis into systematic model improvement signal.

**Q: What are the limitations of SHAP-based explanations that you should communicate to stakeholders?**
A: SHAP explains predictions in terms of feature contributions, but it cannot: prove causality (high SHAP for a feature doesn't mean changing it would change the prediction), guarantee that the explanation reflects the model's actual decision process for deep networks (it approximates), or account for feature correlations fully (correlated features may have their importance split arbitrarily). Also: explanations are for single predictions, not the model's general behavior. Communicate these limitations explicitly when sharing explanations to prevent overconfidence in their meaning.

**Q: How do you build explainability into a model from the design phase rather than as an afterthought?**
A: Design choices: prefer simpler models when accuracy is comparable (a gradient boosted tree with 94% accuracy is more explainable than a neural network with 95%); use attention mechanisms for text models (attention weights give rough explanations); design features to be human-interpretable (one-hot encode rather than embed categorical features when explanation matters); include explanation-specific features in the architecture (auxiliary objectives that force the model to learn interpretable representations). Document the explanation strategy in the model card before training.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
