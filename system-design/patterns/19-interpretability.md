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
