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

## Failure Scenarios

### Failure 1: SHAP on Full Dataset Is Too Slow to Be Actionable
**Symptom:** SHAP computation is scheduled to run nightly and takes 48 hours, making results 48 hours stale. By the time explanations are available, the debugging context has changed and the team has moved on.
**Root Cause:** TreeExplainer runs in O(n × features) time. On 10M samples with 200 features, wall-clock time is prohibitive even on a multi-core machine.
**Detection:** Measure wall-clock time from the moment SHAP is triggered to the moment results are available in the dashboard. If > 2 hours, results are not actionable for same-day debugging.
**Fix:** For exploration and debugging, run SHAP on a stratified 1,000-sample subset — results are available in under 5 minutes and are representative. For compliance reports that require population-level SHAP, run full computation weekly as a batch job (not on-demand). Cache the weekly results in a queryable store so analysts do not re-trigger expensive computation.

### Failure 2: Non-Reproducible Explanations (Compliance Violation)
**Symptom:** The same loan application receives a different LIME explanation on two consecutive API calls. A regulatory audit flags this as inconsistent treatment.
**Root Cause:** LIME generates explanations by sampling random perturbations of the input. Without a fixed random seed, each call produces a different neighborhood sample and different coefficients.
**Detection:** Run LIME 10 times on the same input. Compute the standard deviation of feature importance rankings across runs. If any top-5 feature changes rank by more than 2 positions across runs, reproducibility is insufficient for compliance.
**Fix:** Fix the random seed per model version: `seed = hash(model_version + input_hash) % 2^32`. Log the seed alongside every explanation so any explanation can be reproduced from the audit log. Update the seed when the model version changes (a new model legitimately produces different explanations, which is expected).

### Failure 3: Spurious Feature Importance Discovered Post-Deployment
**Symptom:** SHAP analysis run three months after deployment reveals that `browser_timezone_offset` is the second most important feature for loan approval decisions. This is legally problematic — it is a proxy for geographic location.
**Root Cause:** The feature slipped through feature selection because it genuinely has predictive power (timezone correlates with regional economic conditions). SHAP was not run before deployment to surface this.
**Detection:** Run SHAP feature importance on the validation set before every deployment. Compare the top-10 features against a domain knowledge checklist of approved and prohibited features.
**Fix:** Add a mandatory SHAP audit to the deployment checklist — block promotion if any top-10 feature is flagged as a prohibited proxy. Assign a domain expert reviewer to approve the top-10 feature list before each release.

### Failure 4: Explanation-Reality Mismatch (Rashomon Effect)
**Symptom:** The team trains three models with equivalent accuracy (all within 0.5% AUC of each other). SHAP analysis shows completely different top features for each model. Business stakeholders lose confidence in all explanations.
**Root Cause:** Multiple models fit the data equally well through different feature combinations — the Rashomon effect. SHAP correctly describes each individual model, but the explanations are model-specific and do not represent ground truth about the data-generating process.
**Detection:** Compare SHAP feature importance rankings across an ensemble of your top-3 candidate models. If rank correlation (Spearman) between any two models is below 0.7, the Rashomon effect is present.
**Fix:** Document explicitly that explanations are model-specific, not causal ground truth. For compliance reporting, report the range of importance rankings across the top-3 models to convey uncertainty. Consider switching to an inherently interpretable model (GAM, decision tree) if consistent explanations are a hard business requirement.

### Failure 5: LIME Local Approximation Incorrect for Non-Linear Model
**Symptom:** LIME explanation says "income was the top positive factor." However, manual inspection of similar predictions with slightly different income values shows the model actually has a threshold effect — predictions flip sharply at $75K income, a behavior LIME's linear approximation cannot capture.
**Root Cause:** LIME fits a linear model in the neighborhood of the prediction. For models with sharp non-linearities, the linear approximation has low R² and is structurally wrong.
**Detection:** Check LIME's local R² for every explanation. If R² < 0.7, the explanation is unreliable. For threshold-heavy features, compare LIME coefficients against SHAP values — a large disagreement indicates non-linearity that LIME cannot represent.
**Fix:** Use SHAP instead of LIME for models with known non-linearities (gradient boosting, neural networks). If LIME is required for latency reasons, increase the local neighborhood size and add polynomial interaction terms to the local model.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| SHAP batch compute (weekly full run, GPU) | $2/hr | 4 hr/run × 4 runs/month | $32 |
| LIME on-demand (per-prediction API) | $0.005/request | 1,000 req/day | $150 |
| Explanation storage (S3) | $0.023/GB | 2 GB/day | $1.40 |
| Compliance report generation (engineer time) | $200/hr | 4 hr/month | $800 |
| Domain expert review of top-10 features | $300/hr | 2 hr/deployment | $600 (per deployment) |
| **Total** | | | **~$983/month + $600/deployment** |

Explainability infrastructure is inexpensive compared to the compliance risk it mitigates. A single regulatory fine for unexplained credit decisions can exceed $1M — the $12K/year infrastructure cost has a break-even of less than two weeks of avoided regulatory exposure. The dominant non-compliance cost is engineer and domain expert review time, not compute.

---

## Interview Q&A

**Q1: SHAP explanation takes 5 minutes per prediction. Production needs < 100ms. Solution?**
A: Use a three-tier strategy: (1) real-time API: use LIME (< 10ms) or a pre-trained surrogate decision tree (< 1ms) for immediate explanations; (2) async audit: compute SHAP in the background for a sampled 10% of predictions and store in a queryable log; (3) compliance reporting: run full-population SHAP weekly as a batch job. Never block the request path with SHAP.

**Q2: Feature A shows +0.3 impact in SHAP. Does it cause the prediction or correlate?**
A: SHAP measures predictive contribution, not causation. To test causality: (1) ablate feature A (set it to baseline), retrain, and check if accuracy drops; (2) test with domain expert (does A have a plausible causal mechanism?); (3) run a natural experiment if available (find users where A changed exogenously and check if predictions changed as expected). SHAP alone cannot answer the causal question.

**Q3: Regulator requires explanations for every loan decision. SHAP too slow. What do you do?**
A: Train a Generalized Additive Model (GAM) as a surrogate that mimics the complex model. GAMs are inherently interpretable (output = sum of smooth functions of individual features), produce explanations in < 1ms, and can be validated against the complex model for agreement rate (target > 95% on held-out test set). Use the GAM for real-time explanations; periodically recompute SHAP for the complex model to validate that the GAM surrogate remains accurate.

**Q4: LIME local model R² = 0.6. Is the explanation trustworthy?**
A: No. R² = 0.6 means the linear approximation explains only 60% of the model's local behavior — the other 40% is non-linear structure that LIME cannot capture. Options: (1) switch to SHAP, which handles non-linearities via Shapley values; (2) use a local decision tree (depth 3-4) instead of a linear model for LIME's surrogate; (3) increase the local neighborhood size; (4) document the R² value alongside every explanation so users understand the confidence level.

**Q5: When would you NOT use SHAP?**
A: (1) When explanations must be produced in real time at high QPS (SHAP is too slow for > 10 req/sec without pre-computation); (2) when the model is a neural network without tree structure (KernelSHAP is very slow; DeepSHAP has approximation errors); (3) when the compliance requirement is counterfactual explanations ("what would you need to change to get approved?"), which SHAP does not provide — use DiCE or a counterfactual generation library instead.

**Q6: What breaks first when explainability scales to 10× more predictions?**
A: LIME on-demand cost scales linearly with requests — at 10× volume, LIME costs increase from $150 to $1,500/month, which is still manageable. The real bottleneck is explanation storage: at 10× volume, 20 GB/day of explanation storage costs $14/day ($420/month), and querying historical explanations requires a proper columnar store (Parquet + Athena) rather than raw JSON in S3.

**Q7: Explanation says "age important." Trust it?**
A: Validate first: (1) flip the age value for the same input and check if the prediction changes in the expected direction; (2) check whether age is a causal driver or a proxy for another variable (e.g., years_of_credit_history); (3) review with a domain expert. SHAP values reflect the model's learned associations, not causal ground truth.

**Q8: Regulatory requirement: explain each prediction. Approach?**
A: Use SHAP for batch audit reporting (run weekly). Use LIME or a surrogate interpretable model for real-time per-decision explanations. Format: "Approved because income = $120K (impact: +0.4), credit score = 750 (impact: +0.3), debt-to-income ratio = 0.25 (impact: +0.2)." Store both the explanation and the model version that produced it for auditability.

## Related Topics
- [Model Debugging](17-model-debugging.md)
- [Fairness Metrics](25-fairness-metrics.md)

## Resources
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME: Local Interpretable Models](https://github.com/marcotcr/lime)
