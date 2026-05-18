# Drift Detection

## TL;DR
Detect when model's assumptions break: input distribution shifts (covariate drift), output distribution shifts (label drift), or P(Y|X) changes (concept drift). Alert for retraining.

## Core Intuition
Model trained on past data. World changes. Model assumptions break. Detect this and retrain.

## How It Works
**Covariate drift:** P(X) changed but P(Y|X) same
- Example: more mobile users than before
- Detection: compare feature distributions (KS test, Wasserstein)

**Label shift:** P(Y) changed but P(X|Y) same
- Example: more fraud attempts
- Detection: monitor label distribution

**Concept drift:** P(Y|X) changed
- Example: user preferences shifted
- Detection: model accuracy drops

## Common Mistakes / Gotchas
- **No baseline:** can't tell if drift significant without baseline
- **Alert fatigue:** too-sensitive thresholds → false positives
- **Not acting:** detect drift but don't retrain → still broken

## Interview Q&A

**Q: What is the difference between data drift, concept drift, and model drift?**
A: Data drift (covariate shift): the distribution of input features changes, but the relationship between features and target stays the same—retrain with new data. Concept drift: the relationship between features and target changes (e.g., user behavior patterns shift after a major event)—need to collect new labels and retrain, not just update feature distributions. Model drift: model performance degrades over time due to either data or concept drift—the observable symptom, caused by one of the above. Diagnosing which type is occurring determines the remediation.

**Q: How do you detect drift without labeled production data?**
A: Unsupervised drift detection uses only input features (no labels needed): compare production feature distributions against training data using statistical tests (KS test for continuous, chi-square for categorical, MMD for multivariate). Population Stability Index (PSI) measures distributional shift for each feature. Reconstruction error from an autoencoder trained on training data is high for out-of-distribution inputs. These detect input drift immediately; output drift (model prediction distribution changes) can also be monitored without labels. Labeled validation (comparing predictions to outcomes) is the gold standard but requires waiting for labels.

**Q: How do you set thresholds for drift alerts without generating too many false alarms?**
A: Set thresholds based on empirical distribution of the metric in a stable period. Compute PSI or KS statistic for each day in a 3-month historical window, take the 95th percentile as the alert threshold. This accounts for natural seasonal variation and traffic patterns. Use dynamic thresholds that adjust for seasonality (a metric may look "drifted" every holiday season but it's predictable). Alert on sustained drift (3+ consecutive days above threshold) rather than single-day spikes.

**Q: How do you prioritize which features to monitor for drift?**
A: Monitor: features with high importance (top 10 by SHAP/permutation importance), features known to be operationally unstable (external data sources, calculated fields), and features that have drifted before. Don't monitor exhaustively—with 100+ features, even 5% false positive rate means 5 false alarms per day. Use feature importance to prioritize: a drift in a low-importance feature has minimal impact on model performance, while drift in a top-5 feature is likely to cause significant degradation.

**Q: What is the appropriate response to detected drift and how do you automate it?**
A: Immediate response: alert the model owner with drift details and estimated impact on model performance. Automated triage: run model quality metrics on recent predictions to quantify actual performance degradation (not just distribution shift). If degradation is significant: trigger retraining pipeline, route traffic to fallback model if retraining takes too long. Not all drift requires action: investigate whether the drift represents a permanent change (retrain) or a temporary spike (wait and monitor). Build a drift response runbook with decision criteria for each response level.

## Interview Quick-Reference
**Drift types?** Covariate (X changed), label (Y changed), concept (relationship changed).

## Related Topics
- [Monitoring & Observability](16-monitoring-and-observability.md)
- [Model Versioning](06-model-versioning.md) — rolling back when drift detected

## Resources
- [Learning Under Concept Drift](https://arxiv.org/abs/1010.6241)
