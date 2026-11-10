# Data and Concept Drift Detection

**TL;DR:** Detecting a changed input distribution is not the same as proving model quality declined; pair feature-shift signals with delayed labels, slice analysis and a reversible operational response.

## Prerequisites

[Model monitoring](17-monitoring-alert-design.md), [evaluation metrics](12-evaluation-metrics.md), and basic probability.

## Learning Outcomes

- Distinguish covariate shift, label shift and concept drift.
- Compute a binned population stability index (PSI) from reference and current samples.
- Use matured labels to measure outcome-rate or model-quality movement separately from feature movement.
- Set a measurable investigation threshold without treating drift as automatic retraining evidence.

## Core Intuition

A smoke detector notices smoke, not the cause of the smoke or the damage to the building. A feature-distribution monitor is similar: it notices that today's traffic differs from a reference period. The shift may be harmless seasonality, a new customer segment, a schema error or a real change in the relationship between inputs and outcomes. The monitor should initiate diagnosis and connect signals to business impact, not make an unsupported claim that the model is wrong.

## How It Works

Covariate shift means the input distribution P(X) changed. Label shift means P(Y) changed, while the conditional relationship P(X|Y) may remain similar. Concept drift refers to a change in P(Y|X), so the mapping from features to outcome changes. These can coexist. Feature-only telemetry can detect some covariate changes immediately, but it cannot establish concept drift because labels are required to evaluate the conditional relationship. Delayed or selectively observed labels introduce additional bias.

This lesson uses the population stability index (PSI) as an interpretable binned shift score. Given reference proportions p_i and current proportions q_i over fixed bins, PSI is Σ(q_i-p_i) log(q_i/p_i). A small epsilon prevents log(0), but binning choices and sample size influence the score. PSI is sensitive to bin edges and does not provide a universal significance threshold. Choose bins from a stable reference, preserve missing and out-of-range buckets, and compare like-for-like populations. The implementation also reports labeled outcome-rate shift as a separate signal so feature changes are not confused with target changes.

For numeric features, establish fixed edges using a representative baseline period. Compute proportions for current data using those same edges; recomputing quantile bins each day can hide movement because the boundaries move with the data. For categorical features, include a stable “other/new category” bucket. Track sample counts and confidence ranges, especially for low-volume segments. A single global statistic can hide opposite changes in two cohorts, so monitor carefully chosen segments such as geography, device class or acquisition channel.

An operational policy should define a trigger and next action. For instance, PSI above 0.20 for two daily windows and at least 1,000 examples can open an investigation ticket; it does not by itself trigger retraining. A high missing-rate or impossible-value signal may be more urgent because it can identify a data contract breach. If mature-label AUC drops by a predeclared amount with adequate coverage, the team can consider rollback or retraining after validating logging and cohort comparability. Thresholds must be calibrated against historical normal variation and the cost of missed incidents.

Reference windows need governance. A single training snapshot may be stale or unrepresentative. Compare against both a seasonally matched baseline and a recent rolling baseline where appropriate, and version the reference data and bin definitions. Do not update a baseline automatically during an unresolved incident: doing so can teach the monitor to accept the failure as normal. Keep baseline refreshes reviewed and auditable.

Missingness and schema changes deserve their own checks rather than being folded into distribution distance. A categorical feature whose new values all land in an “other” bucket may appear stable in a coarse histogram even though an upstream producer changed its vocabulary. Monitor null rate, unknown-category rate, range violations and freshness beside the PSI score. For numeric values, explicitly track underflow and overflow beyond the fixed bin edges; clipping these values into edge bins without counting them can hide an instrumentation bug.

When labels mature, compare metrics on aligned cohorts with the same eligibility criteria. If a fraud label is only observed after a chargeback, the current week contains fewer observable negatives and positives than older weeks. Report label coverage and maturity by cohort before comparing rates. Reweighting can sometimes correct selection, but only when the selection mechanism is understood and overlap assumptions hold. If those assumptions fail, treat observed metrics as partial evidence and gather more data rather than issuing a precise but misleading drift verdict.

Finally, define the response ladder before a threshold fires. A modest feature shift may open a data investigation; a schema violation can quarantine a batch; a confirmed quality decline may pause rollout; and a severe safety regression may require rollback. Record the evidence that advances the incident from one level to the next. A drift detector should not silently initiate retraining, because retraining on an anomalous or mislabeled window can institutionalize the defect and erase a useful baseline.

## Trade-offs

| Method | Strength | Limitation |
|---|---|---|
| PSI | Simple to explain and compute from bins | Depends on edges and lacks universal interpretation |
| KS statistic | Sensitive to numeric distribution changes | Assumes suitable sample interpretation and ignores model impact |
| Label-rate monitoring | Connects to outcomes | Can be biased by label delay and selection |
| Model-quality slices | Measures impact more directly | Requires timely labels and enough examples |

Use PSI or another feature metric as an early-warning signal. Use outcome and quality analysis to decide whether model behavior actually needs intervention.

## Implementation with Python

Run `python3 mlops/implementations/18-drift-detection.py`. The example computes PSI on fixed numeric bins and a separate change in labeled positive rate. It has no external dependencies and rejects empty samples and values outside the declared domain.

```python
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("drift", "mlops/implementations/18-drift-detection.py")
drift = module_from_spec(spec); spec.loader.exec_module(drift)
result = drift.psi([.1, .2, .3, .4], [.1, .2, .3, .4], [0, .25, .5, 1])
assert result["psi"] == 0
```

## Failure Modes

- Moving bin edges conceal real shifts: freeze and version edges from a reference period.
- Seasonality triggers repeated investigations: compare seasonal peers and use a sustained-window rule.
- Unlabeled PSI is treated as model failure: require matured outcome or quality evidence before model rollback.
- A global score hides segment changes: add preselected meaningful slices and enforce minimum sample counts.

## Interview Quick Reference

| Question | Expected answer | Follow-up |
|---|---|---|
| Does feature drift prove concept drift? | No; concept drift concerns P(Y|X) and needs labels or a valid proxy. | How would you debug a PSI increase with stable outcomes? |
| How do you choose a PSI threshold? | Backtest normal and incident periods, account for bins and sample size, and define a response. | Design a policy for a low-volume feature. |
| What reference distribution would you use? | A versioned, representative baseline, often with seasonal comparison. | How do you prevent baseline refresh from masking an outage? |

## Related Concepts

See [data validation](03-data-validation.md), [model monitoring](17-monitoring-alert-design.md), [model testing](09-model-testing.md), and [MLOps interview practice](../interview-prep/mlops-interview-questions.md#reliability-governance-and-cost).

## References

[Evidently: Data Drift](https://docs.evidentlyai.com/metrics/explainer_drift) and [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf).
