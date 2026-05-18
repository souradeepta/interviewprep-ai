# Model Debugging

## TL;DR
Model performs poorly: analyze predictions, feature distributions, edge cases. Tools: confusion matrix, feature importance, SHAP values, error analysis.

## Core Intuition
Model black box. Debug by: inspect what it learned, what it's confused about, why it fails.

## How It Works
**Steps:**
1. Confusion matrix: which classes confused?
2. Feature importance: which features drive predictions?
3. Error analysis: why does it fail on these examples?
4. SHAP values: contribution of each feature per prediction
5. Subgroup analysis: does it fail for specific groups?

**Example:**
- Model: 90% accuracy, but 40% error on ~elderly users
- Root cause: few elderly examples in training
- Fix: collect more elderly data, oversample, or retrain

## Common Mistakes / Gotchas
- **Reporting only overall metrics:** 90% accuracy but 20% on minority class hidden
- **No slicing:** bugs in subgroups invisible if you only look at aggregate
- **Not acting:** identify issue but don't fix it

## Interview Q&A

**Q: How do you systematically debug a production model that starts producing wrong predictions?**
A: Step 1: Isolate the scope—is it all predictions or a specific subset (one feature value, one user segment, one time period)? Step 2: Check data pipeline—are features computed correctly, is there new data quality issue? Step 3: Check model inputs—are any features outside training distribution? Step 4: Check model version—did a recent deployment change anything? Step 5: Check the prediction itself—run the input through the model with logging to see the reasoning chain. Start with the simplest hypothesis (data issue) before assuming model bug.

**Q: What information should you log for model predictions to enable debugging?**
A: Log: request ID, timestamp, model version, all input features (or their hashes for PII), raw model output (logits/probabilities, not just final prediction), prediction latency, and the session/user context. For LLMs: log the full prompt, the completion, token counts, and model confidence if available. Store these logs in a queryable format (not just as text). Retention: 30 days for full logs, then aggregated statistics indefinitely. Never debug production issues without this data—it's the equivalent of application error logging for ML systems.

**Q: What are the most common root causes of sudden model performance degradation?**
A: In order of frequency: (1) data pipeline change—upstream feature changed schema, computation bug, missing data; (2) traffic distribution shift—new user segment, marketing campaign changes input distribution; (3) code deployment—serving code change introduced bug; (4) model registry issue—wrong model version deployed; (5) infrastructure change—hardware, library version change affects output; (6) label/feedback loop—model's predictions affect future training data. Check these in order before concluding "the model got worse."

**Q: How do you identify which features are causing a model to make incorrect predictions?**
A: Use SHAP values for individual predictions: compute per-feature contribution for the incorrect predictions and compare to correct ones. Look for: features with unusually high or low values in the failure cases, features where the contribution sign is reversed compared to typical predictions. Use slice-based analysis: segment predictions by feature bins and compute accuracy per slice—the slice with worst accuracy identifies the problematic feature range. A/B compare the feature distributions between your true positives, true negatives, false positives, and false negatives.

**Q: How do you debug a model that is biased toward one class in production?**
A: Check: class distribution in recent production predictions vs. training data, threshold calibration (is the decision threshold still appropriate for current traffic?), class distribution in recent training data (if model was recently retrained), and feature distributions for each class (are class distributions shifting differently?). If the class distribution in predictions has shifted without a corresponding shift in true class distribution, the model may need recalibration. If true class distribution has shifted (concept drift), retraining is needed.

## Interview Quick-Reference
**Debug model?** Confusion matrix, feature importance, error analysis, SHAP. Slice by subgroups.

## Related Topics
- [Model Explainability](18-model-explainability.md)
- [Interpretability](19-interpretability.md)

## Resources
- [SHAP](https://github.com/slundberg/shap)
- [Error Analysis](https://www.microsoft.com/en-us/research/publication/towards-accountable-ai-systems-mechanisms-for-supporting-verifiable-claims/)
