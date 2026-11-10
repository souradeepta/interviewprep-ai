# Regression Diagnostics and Robust Standard Errors

TL;DR: Regression diagnostics reveal whether the conditional mean, variance, dependence, and influence assumptions are plausible, while robust standard errors address only specific uncertainty-model violations.

## Prerequisites

Review [estimation](04-maximum-likelihood-estimation.md), [hypothesis testing](05-hypothesis-testing.md), and [linear regression](../../ai/concepts/06-linear-regression.md).

## Learning Outcomes

- Interpret residual, leverage, and influence diagnostics for a fitted regression.
- Distinguish heteroskedasticity from nonlinearity, dependence, and influential observations.
- Explain how HC standard errors change inference without changing the OLS coefficient estimates.
- Select heteroskedasticity-consistent, cluster-robust, or HAC uncertainty estimates based on the sampling design.

## Core Intuition

Regression is a compact story about conditional averages. A fitted line is useful when it captures how the expected outcome changes with predictors. Residuals are the part of observations left unexplained by that story. Looking only at one summary statistic such as R² is like checking an average temperature while ignoring a room with a broken heater: the aggregate can look fine while the structure of errors violates the way uncertainty was calculated.

Residuals plotted against fitted values can reveal curvature, changing spread, or clusters. A fan shape suggests residual variance grows with the expected outcome. Curvature suggests the mean function may be wrong. A few distant predictor values can have high leverage; if they also have large residuals, they may strongly influence coefficients. These are distinct problems. Heteroskedasticity-consistent covariance estimation can adjust standard errors under heteroskedasticity, but it cannot make an omitted nonlinear term appear or prevent a high-leverage point from pulling the line.

## How It Works

For design matrix X and outcomes y, ordinary least squares estimates (hat\beta=(X^TX)^{-1}X^Ty), when the inverse exists. Under a correctly specified conditional mean and independent, homoskedastic errors with variance σ², the covariance is (sigma^2(X^TX)^{-1}). Heteroskedasticity changes the true sandwich form to ((X^TX)^{-1}X^T\Omega X(X^TX)^{-1}), where Ω has observation-specific variances. White’s HC estimators replace unknown Ω with functions of squared residuals; HC0 is a basic large-sample version, while HC1–HC3 apply finite-sample or leverage adjustments.

Robust covariance estimates leave (hat\beta) unchanged. They affect standard errors, confidence intervals, and tests. If errors are correlated within users, schools, or treatment clusters, HC0 still assumes independence across rows and may understate uncertainty. Cluster-robust covariance aggregates score contributions at the independent cluster level and relies on having enough independent clusters. For serial correlation in ordered time series, HAC estimators such as Newey–West account for dependence across nearby time points using a bandwidth choice.

Diagnostics should match the failure being considered. Residual-versus-fitted and residual-versus-predictor plots help assess mean form and variance. Q-Q plots examine residual tail shape for small-sample Gaussian inference, though normality is not necessary for OLS coefficients to be unbiased under exogeneity. Leverage (h_{ii}) measures unusual predictor geometry; Cook’s distance combines leverage and residual size as an influence diagnostic. These signals prompt investigation, not automatic row deletion.

Complexity is usually O(np² + p³) for dense least-squares fitting with n rows and p predictors, while residual diagnostics are O(np). HC covariance uses residual-weighted matrix products and is roughly O(np²) for a dense design. The statistical risk often matters more than compute: when p approaches n, leverage and unstable matrix inverses complicate both fit and diagnostics.

## Trade-offs

| Method | Error structure addressed | Point estimates change? | Main limitation |
|---|---|---:|---|
| Classical OLS covariance | Independent, constant variance | No | Miscalibrated under heteroskedasticity or dependence |
| HC robust covariance | Heteroskedastic, independent observations | No | Does not fix dependence, mean misspecification, or small samples universally |
| Cluster-robust covariance | Arbitrary within-cluster dependence | No | Needs enough independent clusters; few-cluster corrections matter |
| HAC / Newey–West | Heteroskedasticity and serial dependence | No | Bandwidth and stationarity assumptions affect results |
| Weighted least squares | Known/modelled variance structure | Yes, relative to OLS | Incorrect weights can reduce efficiency or change estimand behavior |

## Implementation

This standard-library example fits a simple intercept-plus-slope model and compares the homoskedastic slope standard error to an HC0 sandwich estimate. The synthetic noise variance grows with x. This diagnostic comparison does not include small-sample corrections or clustered dependence.

```python
import math
import random

random.seed(20)
x = [i / 10 for i in range(1, 101)]
y = [2 + 3 * value + random.gauss(0, 0.02 + 0.04 * value**3) for value in x]
n = len(x)
xbar, ybar = sum(x) / n, sum(y) / n
sxx = sum((value - xbar) ** 2 for value in x)
slope = sum((a - xbar) * (b - ybar) for a, b in zip(x, y)) / sxx
intercept = ybar - slope * xbar
residuals = [b - (intercept + slope * a) for a, b in zip(x, y)]
sigma2 = sum(e * e for e in residuals) / (n - 2)
classical_se = math.sqrt(sigma2 / sxx)
hc0_se = math.sqrt(sum(((a - xbar) * e) ** 2 for a, e in zip(x, residuals)) / sxx**2)
low = sum(e*e for a, e in zip(x, residuals) if a < 5) / sum(a < 5 for a in x)
high = sum(e*e for a, e in zip(x, residuals) if a >= 5) / sum(a >= 5 for a in x)
assert high > low * 5  # the residual spread changes sharply with predictor level
assert classical_se > 0 and hc0_se > 0
assert hc0_se > classical_se * 1.08
print("slope, classical SE, HC0 SE:", slope, classical_se, hc0_se)
```

When diagnosing a real model, visualize residuals against fitted values and important predictors before selecting covariance estimates. If a nonlinear pattern appears, improve the conditional-mean specification and re-evaluate. If residuals cluster by user, choose the cluster as the inferential unit. Robust inference should follow a diagnosis rather than substitute for one.

## Failure Modes

- **Treating robust errors as a model fix:** HC errors leave coefficients and predictions unchanged. Diagnose curvature, omitted interactions, and influential points separately; refit only when justified and validate out of sample.
- **Clustering at the wrong level:** Row-level robust errors do not repair cluster assignment or repeated-measure dependence. Cluster at the independent assignment/sampling unit and report cluster count.
- **Few clusters:** Conventional cluster-robust asymptotics can be poor with few clusters. Use a justified small-sample correction or randomization/wild-cluster procedure and temper conclusions.
- **Deleting influential cases automatically:** High influence may reveal data error or a meaningful population tail. Verify provenance and report sensitivity with and without justified corrections.
- **Using robust errors after data-driven model selection as if fixed:** Selection uncertainty is omitted. Use held-out confirmation, selective inference methods, or clearly label exploratory estimates.
- **Assuming normal residuals are the key OLS condition:** Exogeneity and design assumptions drive unbiasedness; normality supports exact small-sample inference under a model. Check the relevant assumption for the decision.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| What changes with HC standard errors? | Estimated coefficient uncertainty changes; OLS point estimates do not. | What if errors are clustered by user? |
| What does a fan-shaped residual plot suggest? | Heteroskedasticity, so inspect variance structure and robust inference. | Does robust covariance fix a curved residual pattern? |
| How do you debug one influential point? | Check data validity, leverage, residual, and sensitivity without automatic deletion. | Design a robust sensitivity report. |

## Related Concepts

Continue with [missing-data assumptions](19-missing-data-imputation-assumptions.md), [time-series statistics](14-time-series-statistics.md), and [confidence intervals](06-confidence-intervals.md). The paired notebook is [20-regression-diagnostics-robust-standard-errors.ipynb](../notebooks/20-regression-diagnostics-robust-standard-errors.ipynb). Practice the autocorrelation scenario in [Statistics interview practice](../interview-prep/statistics-interview-questions.md#20-debug-autocorrelated-residuals).

References: [White, “A Heteroskedasticity-Consistent Covariance Matrix Estimator” (1980)](https://doi.org/10.2307/1912934) and [NIST/SEMATECH e-Handbook, Regression Diagnostics](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm).
