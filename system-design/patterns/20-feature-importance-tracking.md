# Feature importance tracking

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
- Track feature importances across model versions to detect distribution shift in inputs
- Alert when a previously unimportant feature becomes top-ranked — often indicates data issues
- Combine multiple importance methods (permutation, SHAP, gradient) for robustness
- Track feature importance stability across cross-validation folds
- Use importance tracking to guide feature engineering for the next model iteration
- Visualize importance trends over time, not just point-in-time values
- Separate importance from model version metadata

## Interview Q&A

**Q: How do you use feature importance to identify when a model should be retrained?**
A: Track feature importance over time using a sliding window of recent predictions. When important features' importance ranks change significantly (feature that was always top-5 drops to top-20), investigate: has the feature distribution changed? Has its correlation with the target shifted? Use importance change as a leading indicator of model degradation. Set alerts when top-5 feature importances shift >30% from historical baseline. Importance change often precedes measurable accuracy degradation by days to weeks, enabling proactive retraining.

**Q: What are the limitations of permutation importance vs. SHAP importance?**
A: Permutation importance: shuffle one feature, measure accuracy decrease—captures total effect including correlation with other features. Limitation: correlated features split importance between them unpredictably. SHAP importance: average absolute SHAP values—captures marginal contribution while accounting for correlation. More reliable for correlated features. Use permutation importance for quick feature selection; use SHAP for debugging and communication. Both can mislead when features are highly correlated—be skeptical of any single importance metric for correlated feature sets.

**Q: How do you use feature importance to detect data leakage?**
A: Leakage indicator: a feature has suspiciously high importance (>3x the next feature), especially a feature that shouldn't logically cause the target. Examples: a transaction timestamp in a fraud model (fraud transactions may be processed later), a post-event feature in a before-event prediction. For each high-importance feature, ask: "Could a value of this feature only be known if we already knew the outcome?" If yes, it's leakage. Verify by checking whether removing the suspicious feature causes a large accuracy drop—it will if it's leaking the target.

**Q: How do feature importance values change between model versions and what do you do with that information?**
A: Changes to expect: new features introduced in a retrain may rank highly, displacing existing features. Features whose distribution has changed may rank differently. Important features that now rank low may indicate the model has learned a different solution (possibly through shortcuts). Compare feature importance between model versions before promoting: if a version has a completely different feature importance ranking, investigate even if accuracy is comparable—it may have learned a different (possibly less robust) function.

**Q: How do you use feature importance for feature selection in production models?**
A: Eliminate features with near-zero importance if: (1) they're expensive to compute (API calls, complex aggregations), (2) they add noise that reduces interpretability, or (3) you want to reduce serving complexity. Never eliminate features based on importance alone without measuring the impact on model accuracy—low-importance features may still contribute to edge-case performance. Implement a feature ablation test: retrain without the candidate features and measure accuracy on a held-out test set. Remove only features where ablation shows <1% accuracy change.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
