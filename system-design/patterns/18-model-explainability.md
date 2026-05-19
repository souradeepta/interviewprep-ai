# Model Explainability

## TL;DR
Explain why model made a prediction. Techniques: SHAP (feature importance), LIME (local approximation), decision trees (inherently interpretable). Required for: regulated domains (healthcare, finance), high-stakes decisions.

## Core Intuition
Black-box model predicts 0.92 (approve loan). Why? Explainability answers: "because income=high (impact: +0.3), credit_score=low (impact: -0.1)..."

## How It Works

**Three approaches:**

1. **SHAP (SHapley Additive exPlanations):**
   - Compute contribution of each feature to prediction
   - Global: which features matter most overall
   - Local: for this specific prediction, which features drove it

2. **LIME (Local Interpretable Model-agnostic Explanations):**
   - Fit simple linear model to predict same as black-box
   - Use linear model coefficients as explanation

3. **Decision Trees:**
   - Inherently interpretable: path from root to leaf explains prediction
   - Downside: low accuracy vs neural nets

| Method | Global Explanations | Local Explanations | Accuracy |
|--------|---|---|---|
| SHAP | Yes | Yes | Neutral |
| LIME | No | Yes | Neutral |
| Trees | Yes | Yes | Lower |

## Key Properties / Trade-offs
- Accuracy vs interpretability: trees interpretable but less accurate
- Computation: SHAP expensive, LIME cheap
- Trust: explanations must be correct (garbage in → garbage explanation)

## Common Mistakes / Gotchas
- Explanation wrong: SHAP values computed incorrectly
- Over-confident: explanation seems clear but isn't (confounding)
- Ignoring domain: explanation makes sense mathematically but not in business

## Best Practices
- **Validate explanations:** use human review to check explanations make sense
- **Global + local:** understand overall model and specific predictions
- **Sanity checks:** flip feature value, does explanation change appropriately?
- **Domain review:** have domain expert validate explanations

## Code Example
```python
import shap

# Train model
model = train_model(X, y)

# SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global: average absolute SHAP per feature
shap.summary_plot(shap_values, X)  # Feature importance

# Local: for one prediction
shap.force_plot(explainer.expected_value, shap_values[0], X[0])
```

## Interview Q&A
**Q: Regulatory requirement: explain each prediction. Approach?**
A: Use SHAP or LIME. SHAP more rigorous (Shapley values mathematically grounded), LIME faster. For loan approvals: "Approved because income=$100K (impact +0.4), credit_score=750 (impact +0.3), ..."

**Q: Explanation says "age important". Trust it?**
A: Validate. (1) Flip age value, does prediction change? (2) Is age causally important or correlated with actual driver? (3) Does domain expert agree? Don't trust until validated.

## Interview Quick-Reference
| Method | Speed | Rigor | When |
|--------|-------|-------|------|
| SHAP | Slow | High | Healthcare (need rigor) |
| LIME | Fast | Medium | Quick explanation |
| Trees | Fast | High | Interpretability critical |

## Related Topics
- [Model Debugging](17-model-debugging.md)
- [Fairness Metrics](25-fairness-metrics.md)

## Resources
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME: Local Interpretable Models](https://github.com/marcotcr/lime)
