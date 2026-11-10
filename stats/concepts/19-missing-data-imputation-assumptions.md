# Missing Data and Imputation Assumptions

TL;DR: Missing-data methods are valid only under assumptions about why values are absent, so describe and test those assumptions before choosing deletion, imputation, or weighting.

## Prerequisites

Review [probability fundamentals](01-probability-fundamentals.md), [maximum likelihood estimation](04-maximum-likelihood-estimation.md), and [linear regression](../../ai/concepts/06-linear-regression.md).

## Learning Outcomes

- Distinguish missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR).
- Explain how complete-case analysis and simple imputation affect estimates and uncertainty.
- Choose multiple imputation or inverse-probability weighting under explicit assumptions.
- Plan diagnostics and sensitivity analyses for assumptions that are not testable from observed data alone.

## Core Intuition

Think of every measured value as a ticket in a box, with a separate mechanism deciding which tickets are visible. If visibility is independent of both observed and missing values, the data are MCAR. If visibility depends on observed variables but becomes independent of the missing value after conditioning on those observed variables, the data are MAR. If visibility still depends on the unseen value after conditioning, the mechanism is MNAR. These labels describe assumptions about the data-generating process, not categories that a missingness test can conclusively identify.

For example, a follow-up survey response may be less likely for people with shorter observed engagement. If engagement is recorded for everyone and response depends on that observed engagement, MAR may be plausible conditional on engagement. Responders overrepresent highly engaged users, so the unweighted responder mean is biased for the full population. Weighting responders inversely by their estimated response probability can recover the target under correct response-model specification and positivity. If response instead depends on an unobserved dissatisfaction score, observed data alone cannot distinguish the resulting MNAR process from alternative explanations.

## How It Works

Let Y be an outcome, X observed covariates, and R indicate whether Y is observed. MCAR requires (R \perp (X,Y)); MAR requires (R \perp Y \mid X); MNAR allows dependence on Y after conditioning on observed X. Under MCAR, complete cases can be unbiased but inefficient. Under MAR, complete-case analysis can still be biased if response depends on observed variables related to Y. Likelihood-based methods, multiple imputation, or inverse-probability weighting can identify target quantities under MAR plus model and positivity assumptions.

Multiple imputation creates M plausible completed datasets by drawing missing values from a predictive distribution conditional on observed data and uncertainty in its parameters. Analyze each completed dataset, then combine estimates: the pooled point estimate is the average, while total variance combines within-imputation variance and between-imputation variation. Deterministically filling missing values with a single mean creates false certainty and compresses variance. Imputation models should include variables related to missingness, outcomes, treatment, and analysis, while respecting variable types and structure.

Inverse-probability weighting estimates response propensity (hat\pi_i=P(R_i=1\mid X_i)) and weights observed cases by (1/hat\pi_i). This reconstructs the covariate mix of the target population if the propensity model is correct and every relevant type of unit has a nonzero probability of response. Very small propensities create unstable large weights; trimming changes the estimand or introduces bias and must be reported. Clustered response requires cluster-aware variance estimation.

No statistical test on the observed portion can generally prove MAR rather than MNAR. Diagnostics can compare missingness rates across observed covariates, inspect overlap, and test sensitivity to imputation models. MNAR sensitivity analysis introduces explicit departures, such as shifting imputed missing outcomes by a delta, and shows how conclusions change. Report which values are absent, why they might be absent, how much uncertainty the method adds, and which conclusions depend strongly on assumptions.

## Trade-offs

| Method | Assumption / use | Strength | Main limitation |
|---|---|---|---|
| Complete-case analysis | MCAR, or special model conditions | Simple and transparent | Can lose power and bias under MAR/MNAR |
| Single mean imputation | Rarely defensible for inference | Easy to compute | Understates variance and distorts relationships |
| Multiple imputation | MAR conditional on included predictors and a suitable model | Propagates imputation uncertainty | Model complexity; MNAR remains unresolved |
| Inverse-probability weighting | MAR and positivity with a suitable response model | Reweights observed units to target mix | Large weights and propensity misspecification can destabilize estimates |
| Sensitivity analysis | Plausible MNAR departures | Makes untestable assumptions visible | Does not identify a unique answer from observed data |

## Implementation

This synthetic example has two observed groups with different outcome means and different response probabilities. Complete cases distort the target group mix. Inverse-probability weighting corrects the mix because response depends only on observed group, a MAR mechanism built into the simulation.

```python
import random

random.seed(19)
rows = []
for group, mean, response_prob in ((0, 10, 0.9), (1, 20, 0.3)):
    for _ in range(1000):
        y = mean + random.gauss(0, 2)
        observed = random.random() < response_prob
        rows.append((group, y if observed else None, response_prob))

target = sum(10 if group == 0 else 20 for group, _, _ in rows) / len(rows)
complete = [y for _, y, _ in rows if y is not None]
complete_mean = sum(complete) / len(complete)
weighted = [(y / p, 1 / p) for _, y, p in rows if y is not None]
ipw_mean = sum(value for value, _ in weighted) / sum(weight for _, weight in weighted)
assert abs(ipw_mean - target) < abs(complete_mean - target)
print("target, complete-case, weighted:", target, complete_mean, ipw_mean)
```

In practice, response probabilities are estimated rather than supplied. Fit the response model using observed predictors, inspect propensity overlap, calculate effective sample size, and use robust variance that accounts for estimated weights and sampling design. Multiple imputation can be more efficient than weighting under correct models, but should produce multiple draws and pooled uncertainty rather than one filled-in table.

## Failure Modes

- **Calling MAR because the missingness pattern looks regular:** Mechanism assumptions involve unobserved values and are not directly testable. Document domain reasoning and vary assumptions through sensitivity analysis.
- **Dropping all incomplete rows:** Complete-case analysis changes the population mix and may be biased under MAR. Compare missingness by observed predictors, estimate response propensity, and justify the target population.
- **Single-value imputation:** Replacing every missing value with a mean suppresses uncertainty and may weaken correlations. Use multiple imputation or a model-based method and combine uncertainty.
- **Positivity failure:** Some strata have near-zero response probabilities, causing huge weights. Inspect weight distribution/effective sample size and redefine the estimand or gather more data rather than hiding instability.
- **Imputation leakage:** In predictive ML, imputing before train/test separation lets held-out information influence preprocessing. Fit imputation on training data only and apply that transform to validation/test.
- **Treating MNAR as solved by more data:** A larger sample does not identify an unobserved missingness mechanism. Use external information, selection models, or transparent sensitivity bounds.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| What distinguishes MAR from MNAR? | Under MAR, observed predictors explain selection conditional on them; MNAR still depends on missing values. | Can observed data prove MAR? |
| Why is mean imputation problematic? | It suppresses variance and distorts relationships, yielding overconfident inference. | What uncertainty does multiple imputation add? |
| When does inverse-probability weighting fail? | Positivity or response-model assumptions fail, often producing extreme weights. | Debug a tiny effective sample size. |

## Related Concepts

Continue with [confidence intervals](06-confidence-intervals.md), [regression diagnostics](20-regression-diagnostics-robust-standard-errors.md), and [bootstrap resampling](16-bootstrap-resampling.md). The paired notebook is [19-missing-data-imputation-assumptions.ipynb](../notebooks/19-missing-data-imputation-assumptions.ipynb). Practice the outcome-missingness scenario in [Statistics interview practice](../interview-prep/statistics-interview-questions.md#15-diagnose-informative-missingness).

References: [Rubin, “Inference and Missing Data” (1976)](https://doi.org/10.1093/biomet/63.3.581) and [National Research Council, *The Prevention and Treatment of Missing Data in Clinical Trials*](https://nap.nationalacademies.org/catalog/12955/the-prevention-and-treatment-of-missing-data-in-clinical-trials).
