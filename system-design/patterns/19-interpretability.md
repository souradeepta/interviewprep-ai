# Interpretability

## TL;DR
Model outputs understandable to humans. Methods: feature importance (which features matter), decision rules (if-then logic), counterfactuals (what would change prediction). vs Explainability (why prediction), interpretability is (what prediction means).

## Core Intuition
Prediction = 0.92 (approve). Interpretability: "because these features [income, age]". Explainability: "because income was high relative to baseline, age was low relative to baseline".

## How It Works

**Interpretability techniques:**

1. **Feature importance:** rank features by contribution
   - TreeSHAP: importance from tree structure
   - Permutation: drop feature, measure performance drop
   - Correlation: correlation with target

2. **Decision rules:** extract if-then logic
   - Example: if age>30 AND income>$50K THEN approve

3. **Counterfactuals:** "to flip this prediction, change X by Y"
   - Example: "to deny this loan, reduce income by $5K"

| Technique | Output | Use Case |
|-----------|--------|----------|
| Feature importance | "income is most important" | Understand model overall |
| Decision rules | "if income>$50K approve" | Simple to implement |
| Counterfactuals | "reduce income $5K → flip prediction" | Show user how to change outcome |

## Key Properties / Trade-offs
- Accuracy: interpretable models (trees) less accurate
- Complexity: explaining complex models requires complex methods
- Computational cost: SHAP expensive, feature importance cheap

## Detailed Trade-off Analysis

| Approach | Interpretability | Accuracy | Computation | Regulatory | User UX |
|----------|------------------|----------|-------------|-----------|---------|
| Decision trees | High | 85-90% | <1ms | Good | Excellent |
| GLM (GAM) | High | 80-88% | <1ms | Excellent | Good |
| Complex + SHAP | Medium-High | 95%+ | 100-1000ms | Acceptable | Fair |
| Decision rules | Very High | 70-80% | <1ms | Excellent | Excellent |
| Counterfactuals | High | Neutral | 500-2000ms | Good | Excellent |

**Decision:** Real-time low-stakes → trees. Compliance required → decision rules. Maximum accuracy needed → complex + explanations.

---

## Production Failure Scenarios

**Scenario 1: Simplified rule omits edge cases**
- Rule: "if income>$50K, approve". But for income=$50.1K with debt=$100K, should deny.
- Fix: Use conjunction rules (if-AND). Or admit rule is simplification, reserve human review for edge cases.

**Scenario 2: Feature importance misleads on actionability**
- "income" is most important. But customer can't change income quickly.
- User sees interpretation, tries to increase income, gives up.
- Fix: Counterfactual interpretation: "to approve, would need income $10K higher" (clear, actionable).

**Scenario 3: Interpretability-accuracy trade-off accepted, but accuracy target missed**
- Switch to decision tree for interpretability. Accuracy drops from 95% to 85%.
- Business impact higher than expected (feature predictions used for routing).
- Fix: Hybrid model. Use tree for explainability, complex model for routing. Explain complex model post-hoc.

**Scenario 4: Feature importance changes with season**
- Winter: "weather" most important. Summer: "temperature" most important. Rule inconsistent.
- Fix: Stratify importance by season. Report seasonal interpretability.

---

## Implementation Guidance

**Wrong:** Maximize accuracy (neural net), then try to interpret. Interpretation often uninformative.
**Right:** If interpretability required, build interpretable-first (trees, GLM), accept modest accuracy trade-off, explain complex model if needed for specific cases.

**Wrong:** Use correlation as feature importance. Confounded variables mislead.
**Right:** Use permutation importance or SHAP (accounts for correlations). Validate with domain expert.

---

## Sophisticated Interview Q&A

**Q1: Model 95% accurate (neural net). Regulatory requirement: interpretability. Options?**
A: (1) Surrogate model: train interpretable model to mimic complex model (drop to 90% but interpretable). (2) Post-hoc: use SHAP for specific predictions on demand. (3) Hybrid: use complex model for predictions, tree for high-stakes decisions (require consistency). Pick based on accuracy vs interpretability trade-off tolerance.

**Q2: Decision rule "if income>$50K, approve" covers 60% of cases. 40% need human review. Cost?**
A: (1) If human review = $10 per decision, 40% × $10 = overhead. (2) Tighten rule to 80% coverage (accept some errors). (3) Add more conditions (income>$50K AND debt<$100K) to increase rule coverage. Balance: simplicity vs coverage.

**Q3: Permutation importance says "feature A most important". But model works without A. Why?**
A: Feature A correlated with other features. Permutation importance drops when A shuffled, but model learns A's pattern from correlates. Root cause: multicollinearity. Fix: (1) Use conditional importance (condition on correlates). (2) Remove correlates, recompute. (3) Use SHAP instead.

**Q4: Counterfactual says "reduce income $20K to deny loan". Is this useful for customer?**
A: (1) Not actionable (customer can't reduce income to get denied). (2) Better: show what WOULD change decision (increase debt, or reduce savings). (3) For acceptance: show what increases approval likelihood (increase income, or reduce debt). Make counterfactuals actionable.

---

## Cost & Resource Analysis

**Decision trees:** Train <1s, no explanation overhead. Accuracy trade-off: 5-10% vs complex models.
**Post-hoc SHAP:** Overhead 100-1000ms per prediction. $100-500/month if 10K decisions/month.
**Surrogate model:** Train once $500, validation $1K. Ongoing: inference free (mimics complex model speed).
**Counterfactuals:** Compute on demand, 500-2000ms. $1-2K/month infrastructure if high volume.

**ROI:** Missing interpretation = regulatory fine ($1M+) or customer churn. Cost of interpretability infrastructure: $5-50K/year. Break-even at 1 compliance incident prevented.

---

## Monitoring & Observability

**Key metrics:** Model interpretability score (% of decisions covered by rules), interpretation quality (expert review rate), accuracy of surrogate vs main model, counterfactual actionability (% of suggested changes feasible)

**Alerts:** Accuracy drop of surrogate model (indicates model drifted, explanation invalid), interpretation consistency drops (model behavior changed), rule coverage drops (rules becoming less applicable)

## Common Mistakes / Gotchas
- Correlation ≠ causation: feature important but not causal
- Over-simplification: "if-then" rule misses edge cases
- Model cheating: importance high but feature easily gamed

## Best Practices
- **Combine methods:** use both feature importance + decision rules
- **Domain validation:** have expert review interpretations
- **Counterfactuals:** show actionable paths to flip prediction
- **Fairness check:** ensure interpretations reveal bias (if any)

## Code Example
```python
from sklearn.inspection import permutation_importance

# Feature importance via permutation
result = permutation_importance(model, X_test, y_test)
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': result.importances_mean
}).sort_values('importance', ascending=False)

print(importance_df)  # Top features
```

## Interview Q&A
**Q: Interpretable vs Accurate: choose one. Which?**
A: Accuracy for low-stakes (recommendations), interpretability for high-stakes (healthcare, criminal justice). Or: build accurate model + explain it (SHAP, LIME).

**Q: Decision rule says "if age>30, approve". But 25-year-olds sometimes approved. Rule wrong?**
A: Rule is simplification. Use in combination with other features. Or: find tighter rule with domain expert input.

## Interview Quick-Reference
| Scenario | Best Technique |
|----------|---|
| Regulatory compliance | Decision rules + feature importance |
| User understanding | Counterfactuals |
| Model debugging | Feature importance |

## Related Topics
- [Explainability](18-model-explainability.md)
- [Bias Detection](24-bias-detection.md)

## Resources
- [Interpretable ML Book](https://christophgerstner.github.io/interpretable_machine_learning_book/)
