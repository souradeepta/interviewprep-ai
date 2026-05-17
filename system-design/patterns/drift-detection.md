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

## Interview Quick-Reference
**Drift types?** Covariate (X changed), label (Y changed), concept (relationship changed).

## Related Topics
- [Monitoring & Observability](monitoring-and-observability.md)
- [Model Versioning](model-versioning.md) — rolling back when drift detected

## Resources
- [Learning Under Concept Drift](https://arxiv.org/abs/1010.6241)
