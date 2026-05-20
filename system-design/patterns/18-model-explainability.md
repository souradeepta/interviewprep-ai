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

## Detailed Trade-off Analysis

| Aspect | SHAP | LIME | Decision Trees | Prototype-Based |
|--------|------|------|----------------|-----------------|
| Computation time | 10 min per 1K samples | 10 sec per sample | Instant | 1 min per sample |
| Rigor (Shapley values) | Yes | No | Yes | No |
| Local explanations | Yes | Yes | Yes | Yes |
| Global explanations | Yes | No | Yes | Yes |
| Model agnostic | No | Yes | No | Yes |
| Latency (per-prediction) | Prohibitive | 5-10ms | <1ms | 10-50ms |

**Decision:** Batch analysis → SHAP. Real-time API → LIME or trees. Compliance required → SHAP or trees.

---

## Production Failure Scenarios

**Scenario 1: Explanation misleads on causality**
- SHAP shows "age important", model always predicts high for old users. Team assumes age causes decisions.
- Root cause: age correlated with income. Real driver: income, not age.
- Fix: Validate explanations with domain experts. Test by removing feature, retraining, checking if accuracy changes.

**Scenario 2: SHAP values computed incorrectly**
- SHAP assumes feature independence. But features correlated (age + years_employed). Explanation invalid.
- Fix: Use conditional SHAP (TreeExplainer) or document assumptions.

**Scenario 3: Explanations overly complex**
- SHAP explanation: "Average impact of this feature across 10K background samples is +0.034". Customer doesn't understand.
- Fix: Simplify for end users. Use LIME (linear approximation) or prototype examples ("similar approved loans had income >$X").

**Scenario 4: Explanation contradicts business logic**
- Model says "approved because high income". But manual review finds model is rejecting high-income applicants 30% of the time.
- Root cause: LIME locally linear, but model is non-linear. Local approximation wrong.
- Fix: Use SHAP for rigor. Or add sanity checks (does explanation flip appropriately when feature changes?).

---

## Implementation Guidance

**Wrong:** Trust SHAP explanation without validation. Assume it's causally correct.
**Right:** Validate with domain experts. Test by modifying feature, retraining, checking if explanation changes.

**Wrong:** Use SHAP for every prediction in production (too slow).
**Right:** SHAP for batch analysis and auditing. LIME or rules-based for real-time API.

---

## Sophisticated Interview Q&A

**Q1: SHAP explanation takes 5 minutes per prediction. Production needs <100ms. Solution?**
A: Tiering. (1) Real-time: use simpler method (LIME, decision rules, prototypes) <100ms. (2) Audit: SHAP for sampled predictions, batch processing offline. (3) Caching: precompute SHAP for common inputs.

**Q2: Feature A shows +0.3 impact in SHAP. Does it cause prediction or correlate?**
A: Can't tell from SHAP alone. Validate: (1) ablate feature A, retrain, check accuracy. (2) Does domain expert agree? (3) Test on holdout: if A important, removing A should hurt. If explanation wrong, accuracy stays same.

**Q3: Regulator requires explanations for every loan decision. SHAP too slow. What do you do?**
A: Hybrid approach. (1) Train interpretable model (e.g., Generalized Additive Model) as surrogate. (2) Use surrogate for real-time explanations (<100ms). (3) Validate surrogate against complex model (should agree >95% of time). (4) Periodically recompute SHAP for audit/validation.

**Q4: LIME local model R² = 0.6. Is explanation trustworthy?**
A: No. Low R² means linear approximation doesn't fit local model behavior. Either (1) use SHAP instead, (2) use more complex local model (e.g., small decision tree), (3) increase local neighborhood size, or (4) accept lower confidence.

---

## Cost & Resource Analysis

**SHAP computation:** 10 min for 1K samples. $100-500 per 1M samples (cloud compute).
**LIME:** Real-time capable. Infrastructure: <$10/month (negligible compute).
**Model surrogate:** Train once, <$100. Validation: 2 engineer hours = $500.

**Compliance cost:** Missing explanations = regulatory fine ($1M+). Cost of explainability infrastructure: $5-20K/year. Break-even on first incident prevented.

---

## Monitoring & Observability

**Key metrics:** Explanation stability (same prediction, same explanation?), explanation-model agreement (does SHAP match local behavior?), explanation complexity (average number of features), explanation latency, audit coverage (% of predictions explained)

**Alerts:** Low agreement between explanation and model (indicates wrong explanation), latency spike on explanation computation, explanation confidence drops (model became non-linear), regulatory audit failures

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
