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

## Interview Quick-Reference
**Debug model?** Confusion matrix, feature importance, error analysis, SHAP. Slice by subgroups.

## Related Topics
- [Model Explainability](model-explainability.md)
- [Interpretability](interpretability.md)

## Resources
- [SHAP](https://github.com/slundberg/shap)
- [Error Analysis](https://www.microsoft.com/en-us/research/publication/towards-accountable-ai-systems-mechanisms-for-supporting-verifiable-claims/)
