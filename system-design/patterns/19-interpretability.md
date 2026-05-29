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

## Failure Scenarios

### Failure 1: Attention Weights Misread as Feature Importance
**Symptom:** The team uses raw attention weights from a BERT-based classifier to identify which input tokens "caused" a classification. They debug in the wrong direction for two weeks based on these rankings.
**Root Cause:** Attention weights measure correlation between token representations at a specific layer, not causal importance. A token can receive high attention and have zero causal influence on the output, particularly when attention heads are diffuse or redundant.
**Detection:** Compare attention-based token rankings against gradient × attention (attention rollout) rankings on 50 known examples where the correct attribution is established by domain experts. If rank correlation is below 0.6, attention weights alone are not reliable importance signals.
**Fix:** Always use gradient × input, Integrated Gradients, or attention rollout for token-level importance. Document in team guidelines: "raw attention weights are not feature importance." Remove any dashboard that displays raw attention as importance without qualification.

### Failure 2: Probing Accuracy Misread as Model Understanding
**Symptom:** A probing classifier achieves 85% accuracy at predicting the concept "sentiment" from an intermediate layer. The team concludes the model "understands sentiment" and uses this to justify deployment in a sentiment-sensitive application without further testing.
**Root Cause:** Probing measures linear separability of a concept in the representation space — it reveals correlation, not causal encoding. A representation might be linearly separable for sentiment while the model's predictions are driven by lexical shortcuts (specific words) rather than sentiment understanding.
**Detection:** Supplement probing with a causal intervention test: zero out the representation of the probed concept using activation patching (ROME or causal tracing). If zeroing the concept representation does not affect model outputs, the representation is not causally involved in predictions.
**Fix:** Require causal intervention validation before concluding a model "has" a concept. Use probing as a necessary but not sufficient condition. Document findings as "evidence of linear separability for X" rather than "model understands X."

### Failure 3: Interpretability Tool Incompatibility with Production Architecture
**Symptom:** The team selects GradCAM for visual attribution based on a well-cited paper. Implementation begins, but GradCAM requires named convolutional layers with specific hook-able structure. The production model uses a vision transformer (ViT) with attention blocks, not convolutions. GradCAM is incompatible.
**Root Cause:** The tool was selected based on paper results without verifying compatibility with the production model architecture.
**Detection:** Run each candidate interpretability tool against the production model in a sandbox environment before committing to it. Failure modes include: missing hook support, shape mismatches, and methods requiring specific layer types.
**Fix:** Maintain a tested-compatibility matrix: {model architecture} × {interpretability tool} × {tested: yes/no}. Before selecting a method, verify it runs end-to-end on the production model with a toy input. For ViTs, use attention rollout or DINO-style self-attention visualization instead of GradCAM.

### Failure 4: Misleading Saliency Maps
**Symptom:** A GradCAM saliency map for an image classifier confidently highlights the image background (sky) rather than the object (bird). The model's accuracy is high. The team trusts the model anyway, but later discovers it fails on images with unusual backgrounds.
**Root Cause:** Gradient-based saliency methods are noisy for high-confidence predictions where gradients are near-zero (saturation problem). Background texture is a spurious shortcut the model learned, but the saliency map does not reveal this clearly because the gradient signal is weak.
**Detection:** Compare GradCAM vs Integrated Gradients vs SmoothGrad on the same 100 inputs. If the three methods disagree on the top-highlighted region for more than 30% of examples, saliency reliability is in question.
**Fix:** Use multiple attribution methods and report agreement. Flag any prediction where top-attributed region across methods disagrees for manual review before trusting the explanation. For shortcut detection specifically, use spurious correlation audits (test on datasets where the shortcut is absent) rather than relying on saliency maps.

### Failure 5: Interpretability-Accuracy Trade-off Underestimated
**Symptom:** The team switches from a gradient-boosted model (AUC 0.96) to a decision tree (AUC 0.87) to meet an interpretability requirement. Post-deployment, the 9-point AUC drop results in a 15% increase in false negative rate for fraud detection. Fraud losses increase by $2M/month.
**Root Cause:** The business impact of the accuracy trade-off was modeled only in abstract terms ("9 points of AUC") without translating to business outcome (false negatives × average fraud loss). The decision was made on accuracy numbers, not economic impact.
**Detection:** Before accepting an interpretability-accuracy trade-off, convert the accuracy difference to business outcomes: Δ(false_negative_rate) × volume × average_loss_per_missed_fraud.
**Fix:** Explore hybrid approaches before accepting pure accuracy loss: (1) use the complex model for decisions, the interpretable model for explanations, and validate surrogate agreement (> 95%) before deploying; (2) use GAMs (Generalized Additive Models), which offer near-neural-net accuracy with full interpretability for tabular data; (3) accept interpretability in exchange for human review on the ambiguous decision band (middle 20% of predictions), rather than globally degrading accuracy.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Activation analysis compute (GPU) | $2/hr | 10 hr/week | $80 |
| Probing classifier training runs | $0.50/run | 20 runs/week | $40 |
| Saliency map generation (vision models) | $0.001/image | 10,000 images/day | $300 |
| Surrogate interpretable model retraining | $0.50/run | 4 runs/month | $2 |
| Interpretability audit (quarterly, engineer time) | $200/hr | 40 hr/quarter | $2,700/quarter (~$900/month) |
| Counterfactual explanation compute | $0.002/request | 500 req/day | $30 |
| **Total** | | | **~$1,352/month** |

For most teams, interpretability infrastructure costs are dominated by the quarterly audit engineering time ($900/month amortized), not compute. Saliency map generation is the largest ongoing compute cost at $300/month but scales linearly with prediction volume. Teams processing > 1M images/day should pre-compute saliency maps for a 1% sample ($30/month) rather than on-demand for all predictions.

---

## Interview Q&A

**Q1: Model 95% accurate (neural net). Regulatory requirement: interpretability. Options?**
A: Three viable paths: (1) surrogate model — train an interpretable model (GAM, shallow decision tree) to mimic the neural net; target > 95% agreement on held-out data; use surrogate for explanations; (2) post-hoc SHAP — keep the neural net, compute SHAP for sampled or on-demand predictions; acceptable for audit, not real-time; (3) hybrid — use neural net for predictions, require human review for decisions in the high-risk confidence band (0.4-0.6) where the tree also disagrees.

**Q2: Decision rule "if income > $50K, approve" covers 60% of cases. 40% need human review. Cost?**
A: If human review costs $10/decision, 40% human review = $4 overhead per 10 decisions. To reduce: add a second rule condition ("if income > $50K AND debt < $80K, approve") to cover 75% of cases, reducing human review to 25%. The trade-off is rule complexity vs coverage; balance by estimating cost of human review against accuracy loss from a tighter rule.

**Q3: Permutation importance says "feature A most important." But model works without A. Why?**
A: Multicollinearity. Feature A is correlated with features B and C. When A is permuted, the model partially recovers A's signal from B and C — but permutation importance still shows a drop because B and C alone are noisier. This is the known limitation of marginal permutation importance. Fix: use conditional importance (condition on B and C when permuting A), or switch to SHAP which accounts for feature interactions.

**Q4: Counterfactual says "reduce income $20K to deny loan." Is this useful for the customer?**
A: No — this counterfactual tells the customer how to get denied, which is never their goal. For customer-facing explanations, generate actionable counterfactuals in the approval direction: "Increasing your credit score by 50 points or reducing your debt-to-income ratio from 0.45 to 0.35 would result in approval." Filter counterfactuals to only surface feasible, positive-direction changes. Use DiCE (Diverse Counterfactual Explanations) with actionability constraints.

**Q5: When would you choose a decision tree over SHAP-explained gradient boosting?**
A: When you need: (1) zero-latency explanations (tree path is instant; SHAP adds 100ms); (2) explanations that are globally consistent (tree rules apply identically to every prediction; SHAP values are locally computed and can appear inconsistent); (3) regulatory environments requiring a single auditable decision path rather than a post-hoc statistical attribution. Accept 5-10% accuracy loss as the trade-off.

**Q6: What breaks first when interpretability scales to 10× prediction volume?**
A: Saliency map compute scales linearly — 10× images means $3,000/month instead of $300/month. Switch to sampling-based saliency (compute for 1% of predictions) before the cost becomes prohibitive. Counterfactual generation is the second scaling bottleneck: at 10× volume with 500ms per counterfactual, you need horizontal scaling of the counterfactual compute cluster.

**Q7: Interpretable vs. Accurate — choose one. Which?**
A: Accuracy for low-stakes, high-volume decisions (content ranking, recommendations) where errors are cheap and reversible. Interpretability for high-stakes decisions (healthcare diagnosis, credit approval, criminal risk scoring) where errors are costly and must be explained to affected parties. In most real scenarios, a GAM or surrogate-based hybrid avoids the binary choice entirely.

**Q8: Decision rule says "if age > 30, approve." But 25-year-olds are sometimes approved. Rule wrong?**
A: The rule is a simplification — it covers the majority case but has exceptions. Handle this in two ways: (1) add a second condition ("if age > 30 OR (age >= 25 AND income > $80K), approve") to cover the exceptions explicitly; (2) document the rule's coverage rate and error rate, accept human review for out-of-rule cases. A rule that is 90% accurate with clear scope is preferable to a black-box model for compliance purposes.

## Related Topics
- [Explainability](18-model-explainability.md)
- [Bias Detection](24-bias-detection.md)

## Resources
- [Interpretable ML Book](https://christophgerstner.github.io/interpretable_machine_learning_book/)
