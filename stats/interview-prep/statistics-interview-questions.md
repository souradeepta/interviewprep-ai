# Statistics Interview Questions

These prompts target statistics for ML engineering, applied science, and data science. Scenario and debugging questions make the candidate state assumptions and propose checks, rather than recite definitions.

## Foundations and estimation

### 1. Choose a distribution for event counts
**Prompt:** A service records the number of failed requests per minute. What distribution would you start with, and what would make you reject it?
**Role/domain:** Data science; probability. **Difficulty:** Easy.
**Expected answer:** Start with a Poisson model when events are independent, exposure is fixed, and the rate is stable; its mean and variance are both λ. Overdispersion suggests a negative-binomial model, while clustering, changing exposure, or excess zeros require further structure.
**Evaluation:** Assumptions, rate/exposure distinction, diagnostic for dispersion.
**Follow-up:** How would you model different request volumes across minutes?
**Common mistake:** Selecting a distribution from the variable type alone without checking its data-generating assumptions.

### 2. Estimate conversion uncertainty
**Prompt:** In 200 independent visits, 30 convert. Give a point estimate and describe an uncertainty interval.
**Role/domain:** Product data science; estimation. **Difficulty:** Easy.
**Expected answer:** The sample proportion is 0.15. A Wilson or exact binomial interval is preferable to a naive normal interval when samples or rates are small; the interval quantifies uncertainty under independent, representative Bernoulli sampling.
**Evaluation:** Correct estimate, interval choice, sampling assumptions.
**Follow-up:** What changes if visits are clustered by user?
**Common mistake:** Reporting 15% as the population truth or treating repeat visits from one user as independent.

### 3. Interpret a confidence interval
**Prompt:** Explain the frequentist meaning of a 95% confidence interval for a treatment effect.
**Role/domain:** Applied science; inference. **Difficulty:** Easy.
**Expected answer:** Under repeated sampling and the stated procedure’s assumptions, 95% of constructed intervals would contain the fixed true parameter. Once this interval is observed, it is not correct to assign a 95% probability to the parameter being inside it.
**Evaluation:** Long-run coverage, fixed parameter, assumptions.
**Follow-up:** How would a Bayesian credible interval differ?
**Common mistake:** Saying there is a 95% probability that the fixed parameter lies in this realized interval.

### 4. Plan power for a small lift
**Prompt:** An experiment must detect a 1% relative conversion lift. What inputs do you need before estimating sample size?
**Role/domain:** Product experimentation; power. **Difficulty:** Medium.
**Expected answer:** Specify baseline conversion, minimum detectable effect on the absolute scale, variance/model, type-I error, target power, allocation, unit of randomization, and expected attrition. Compute the required sample under those assumptions, then adjust for clustering or multiple primary outcomes.
**Evaluation:** Inputs, effect-scale clarity, randomization and attrition.
**Follow-up:** What would you change if traffic is capped?
**Common mistake:** Treating power as a property of the test alone or using relative percent without baseline rate.

### 5. Debug a low-powered experiment
**Prompt:** A team ran a two-week test and got p = 0.31. They conclude the new ranking has no effect. How do you review that conclusion?
**Role/domain:** Experimentation; debugging. **Difficulty:** Medium.
**Expected answer:** A non-significant result does not establish no effect. Inspect the confidence interval against a prespecified practical-effect range, achieved sample versus planned power, metric variance, assignment integrity, exposure, missingness, and whether the test stopped or changed outcomes opportunistically.
**Evaluation:** Uncertainty interpretation, operational diagnostics, practical significance.
**Follow-up:** How could an equivalence test answer a more useful question?
**Common mistake:** Equating failure to reject the null with evidence that the effect is zero.

### 6. Select a bootstrap procedure
**Prompt:** You need an interval for median session duration, whose distribution is strongly skewed. How would you bootstrap it?
**Role/domain:** Applied statistics; resampling. **Difficulty:** Medium.
**Expected answer:** Resample independent observational units with replacement, recompute the median for each replicate, and use a suitable percentile or BCa interval. If sessions are clustered within users, resample users (or use a justified cluster bootstrap), not individual sessions.
**Evaluation:** Resampling unit, statistic recomputation, assumptions and interval caveat.
**Follow-up:** What does the bootstrap fail to fix?
**Common mistake:** Resampling rows independently despite dependence, or assuming bootstrap removes selection bias.

### 7. Compare bootstrap and analytic intervals
**Prompt:** When would you prefer an analytic standard error over a bootstrap interval, and when would you do the reverse?
**Role/domain:** Data science; estimation. **Difficulty:** Medium.
**Expected answer:** Use a validated analytic result when its model assumptions fit and speed or stable tail inference matters. Bootstrap when the estimator’s sampling distribution is awkward but representative independent units can be resampled. Both inherit data and design problems; bootstrap can be poor for tiny samples or extreme tails.
**Evaluation:** Assumption fit, computational cost, limitations.
**Follow-up:** What resampling design fits repeated measurements per subject?
**Common mistake:** Calling bootstrap assumption-free.

## Experiments and causal reasoning

### 8. Debug an A/B test with a sample ratio mismatch
**Prompt:** A nominal 50/50 experiment has 54/46 assignment counts and a large conversion lift. What do you do first?
**Role/domain:** Product experimentation; debugging. **Difficulty:** Hard.
**Expected answer:** Pause interpretation and investigate assignment logs, eligibility filters, bucketing, exposure instrumentation, retries, and sample-ratio mismatch by platform/time/segment. Verify the analysis population follows assignment (intention-to-treat); only resume causal claims after the imbalance has an understood, non-outcome-driven explanation.
**Evaluation:** Integrity-first response, concrete checks, causal estimand.
**Follow-up:** Why is conditioning on actual exposure potentially biased?
**Common mistake:** Applying a post-hoc weight or segment filter until the counts look balanced.

### 9. Design randomization for network effects
**Prompt:** A collaboration feature causes users to invite one another. How should you randomize its experiment?
**Role/domain:** Causal inference; experiment design. **Difficulty:** Hard.
**Expected answer:** Individual assignment risks spillover and violates independent treatment. Consider cluster or network-component randomization, or a saturation design to estimate direct and spillover effects; trade statistical power against contamination and define the target estimand.
**Evaluation:** Interference recognition, design options, power and estimand.
**Follow-up:** What if the number of independent clusters is small?
**Common mistake:** Treating millions of users as the sample size when treatment was assigned to a small number of clusters.

### 10. Diagnose confounding in an observational launch
**Prompt:** Customers who enabled a feature retain better. Can you call the feature causal, and what analysis would help?
**Role/domain:** Causal inference; observational study. **Difficulty:** Medium.
**Expected answer:** No: adoption is selected and may reflect prior engagement or need. Define a causal graph and target effect, measure pre-treatment confounders, then consider adjustment, matching/weighting, or a credible instrument/natural experiment; state unmeasured-confounding and positivity limits and run sensitivity checks.
**Evaluation:** Selection bias, assumptions, appropriate method and caveats.
**Follow-up:** Why must covariates be measured before treatment?
**Common mistake:** Treating regression adjustment as automatic removal of confounding.

### 11. Explain a randomized test with noncompliance
**Prompt:** Some users assigned to treatment never see it. Which effect does intention-to-treat estimate?
**Role/domain:** Experimentation; causal estimands. **Difficulty:** Medium.
**Expected answer:** ITT estimates the effect of assignment among randomized units, preserving randomization and reflecting rollout effectiveness. A treatment-on-treated estimate needs extra assumptions or an instrumental-variable analysis; comparing exposed users directly can reintroduce selection bias.
**Evaluation:** Estimand distinction, randomization preservation, assumptions.
**Follow-up:** Under what assumptions can assignment instrument exposure?
**Common mistake:** Dropping unexposed assigned users from the treatment arm.

### 12. Review a peeking experiment
**Prompt:** A dashboard checks p-values hourly; the team stops at the first p < 0.05. What is wrong and what would you replace it with?
**Role/domain:** Experimentation; sequential inference. **Difficulty:** Hard.
**Expected answer:** Repeated unadjusted testing inflates false-positive probability. Predefine a fixed horizon, use a valid group-sequential alpha-spending design, or use a calibrated always-valid sequential procedure; log stopping rules and avoid outcome-driven restarts.
**Evaluation:** Error-rate mechanism, valid alternatives, operational governance.
**Follow-up:** Does a Bayesian posterior automatically make arbitrary stopping harmless?
**Common mistake:** Believing a fixed-horizon p-value remains calibrated under unrestricted optional stopping.

### 13. Handle multiple metric tests
**Prompt:** A launch is evaluated on 40 metrics and three are significant at 0.05. How do you report this?
**Role/domain:** Product science; multiple comparisons. **Difficulty:** Medium.
**Expected answer:** Clarify which metrics were prespecified and identify the multiplicity family. For confirmatory claims, control family-wise error (for example Holm) or false discovery rate (for example Benjamini–Hochberg) as appropriate; report adjusted results, effect sizes, intervals, and exploratory status.
**Evaluation:** Family definition, procedure fit, transparent reporting.
**Follow-up:** What if the metrics are strongly dependent?
**Common mistake:** Reporting the three raw p-values as independent confirmations.

### 14. Choose an adjustment for many discoveries
**Prompt:** A genomics screen tests thousands of features and the goal is to prioritize candidates for follow-up. Which error criterion might fit?
**Role/domain:** Applied science; multiple testing. **Difficulty:** Medium.
**Expected answer:** False discovery rate control may be appropriate because the goal tolerates some false leads while limiting their expected proportion among discoveries. Benjamini–Hochberg has dependence conditions; use a more conservative method or validate assumptions when dependence is problematic.
**Evaluation:** FDR versus family-wise distinction, conditions, downstream validation.
**Follow-up:** When would family-wise error be the better choice?
**Common mistake:** Assuming FDR means each selected feature has exactly the target probability of being false.

### 15. Diagnose informative missingness
**Prompt:** Outcomes are missing more often for users with long sessions. What assumptions and sensitivity analyses matter?
**Role/domain:** Data science; missing data. **Difficulty:** Hard.
**Expected answer:** Distinguish MCAR, MAR conditional on observed data, and MNAR depending on unobserved values. Model the missingness process, compare observed covariates, use multiple imputation or weighting under justified MAR assumptions, and perform MNAR sensitivity/bounds; do not silently complete-case analyze.
**Evaluation:** Mechanisms, method-to-assumption link, sensitivity analysis.
**Follow-up:** Can a missingness indicator solve MNAR bias?
**Common mistake:** Assuming missing at random merely because missingness correlates with an observed field.

### 16. Debug a missing-outcome dashboard
**Prompt:** A treatment arm has 8% missing outcomes versus 2% in control after a logging change. What should be reported?
**Role/domain:** Experimentation; debugging. **Difficulty:** Hard.
**Expected answer:** Check whether outcome capture changed by assignment, device, and time, and whether missingness is a treatment-affected post-randomization event. Report missingness rates, avoid treating absent outcomes as zero, assess bounds/sensitivity under plausible outcomes, and repair instrumentation before interpreting the point estimate.
**Evaluation:** Instrumentation diagnosis, missingness mechanism, uncertainty and decision impact.
**Follow-up:** When could inverse-probability weighting help?
**Common mistake:** Comparing only complete cases without assessing assignment-dependent attrition.

### 17. Interpret a difference-in-differences estimate
**Prompt:** A policy launches in one region. What does difference-in-differences require for a causal interpretation?
**Role/domain:** Causal inference; quasi-experiment. **Difficulty:** Hard.
**Expected answer:** In the absence of treatment, treated and comparison groups would have followed parallel trends; also examine no anticipation, stable composition, and spillovers. Plot pre-trends/event studies, choose a credible comparison, and use inference robust to treatment-level clustering; pre-trends cannot prove the assumption.
**Evaluation:** Identifying assumptions, diagnostics, inference unit.
**Follow-up:** What is problematic about a staggered rollout with heterogeneous effects?
**Common mistake:** Treating similar pre-period levels as proof of parallel trends.

### 18. Interpret a regression coefficient
**Prompt:** A linear regression reports a positive coefficient for a feature after adjustment. What can you conclude?
**Role/domain:** Data science; regression. **Difficulty:** Medium.
**Expected answer:** It is the model’s conditional association per unit change given included covariates and functional form, not automatically a causal effect. Check coding, scale, interactions, residual structure, collinearity, and whether adjustment introduced collider or post-treatment bias.
**Evaluation:** Conditional interpretation, causal restraint, diagnostics.
**Follow-up:** How would an interaction alter the meaning of the coefficient?
**Common mistake:** Calling an adjusted coefficient “the effect” without a causal design.

### 19. Choose robust standard errors
**Prompt:** Residual variance grows with fitted value, but coefficient estimates are otherwise plausible. What changes?
**Role/domain:** Applied statistics; regression diagnostics. **Difficulty:** Medium.
**Expected answer:** Inspect model form and influential points; heteroskedasticity-robust standard errors can protect large-sample inference for coefficients without changing the OLS point estimates. They do not fix omitted-variable bias, dependence, nonlinearity, or poor predictions; cluster at the assignment/dependence unit when needed.
**Evaluation:** Estimate versus uncertainty, diagnostics, dependence.
**Follow-up:** What would robust errors not repair if residuals are serially correlated?
**Common mistake:** Treating robust standard errors as a cure for a misspecified mean model.

### 20. Debug autocorrelated residuals
**Prompt:** A regression on daily demand has a strong residual pattern by weekday and autocorrelation. What do you investigate?
**Role/domain:** Forecasting; time series diagnostics. **Difficulty:** Hard.
**Expected answer:** Add or model calendar structure, trend/seasonality and relevant lags, check leakage and regime changes, and validate with time-ordered splits. Use uncertainty estimates that account for serial dependence; do not random-shuffle observations or infer independent errors from a high R².
**Evaluation:** Residual diagnosis, temporal validation, dependence-aware inference.
**Follow-up:** How can differencing help, and what information can it remove?
**Common mistake:** Using random train/test splits that leak future structure into the past.
