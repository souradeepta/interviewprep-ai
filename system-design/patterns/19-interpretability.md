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

## Interview Q&A

**Q: When does a model need to be interpretable by regulatory requirement?**
A: Regulations requiring interpretability: EU GDPR Article 22 (right to explanation for automated decisions that significantly affect individuals), US Equal Credit Opportunity Act (ECOA requires adverse action notices explaining credit decisions), EU AI Act (high-risk systems must provide documentation and explanations), and various financial regulations (SR 11-7 guidance for model risk management). Healthcare (FDA SaMD): not strictly required but expected in submissions. When unsure, consult legal/compliance before choosing an opaque model for regulated decisions.

**Q: What are the trade-offs between model complexity and interpretability?**
A: Linear regression: fully interpretable, limited expressivity—use for low-dimensional, linear problems. Gradient boosted trees: partially interpretable via SHAP, high expressivity—good default for tabular data. Neural networks: black-box, highest expressivity—use when expressivity is essential and interpretability can be provided post-hoc. The interpretability-accuracy trade-off is smaller than often assumed: for many tabular tasks, a tuned gradient boosted tree matches neural network accuracy. Reach for the most interpretable model that meets accuracy requirements.

**Q: How do concept-based explanations (TCAV) differ from feature-based explanations (SHAP)?**
A: SHAP: explains predictions in terms of input feature importance—good for tabular data, less meaningful for images (pixel importance is hard to interpret). TCAV (Testing with Concept Activation Vectors): trains probes to detect human-defined concepts (e.g., "striped texture" for animal classification), then measures how much each concept influences model predictions. Better for: vision models, complex feature spaces where individual features lack semantic meaning, and testing for known biases (does the model use "woman in kitchen" as a concept for chef predictions?).

**Q: How do you evaluate whether an explanation is faithful to the model's actual decision process?**
A: Faithfulness tests: (1) feature ablation—remove the top-k features by explanation importance and measure prediction change (should change significantly); (2) roar/FRESH test—retrain with important features masked and compare accuracy drop; (3) sanity checks—perturb random (unimportant) features and verify explanation doesn't change. Beware: many explanation methods pass visual inspection but fail faithfulness tests—LIME explanations in particular can be unfaithful to the model's actual reasoning for complex models.

**Q: How do you communicate model explanations to non-technical stakeholders?**
A: Use concrete language, not statistical terms: "The model predicted high fraud risk because the transaction occurred at an unusual time (3am) and in a city different from the customer's registered address" rather than "SHAP values for time_of_day=-0.34 and location_delta=0.52." Visualize: waterfall charts for feature contributions, decision trees for simple rules. Provide action: "To reduce fraud risk, verify transactions that occur [condition]." Validate communication with target audience before using in production.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
