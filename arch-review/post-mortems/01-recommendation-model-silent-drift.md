# Post-Mortem: Recommendation Model Silent Drift (COVID Behavior Shift)

## Incident Summary
**Date:** 2020-06-12
**Duration:** ~90 days to detect, 2 weeks to fully resolve
**Business Impact:** Revenue -8% for approximately 3 months; estimated total loss $4.2M
**Severity:** P2 (Silent degradation — no alarms fired during drift)

---

## Timeline

| Time | Event |
|------|-------|
| 2020-03-15 | COVID-19 stay-at-home orders begin across major markets |
| 2020-03-15 | User behavior shifts: home purchases up, commute-related purchases down |
| 2020-03-15 | `hour_of_day` feature distribution drifts significantly (peak moves from commute hours to midday/evening) |
| 2020-03-20 | PSI monitoring runs on 5 monitored features — none flag above 0.2 threshold |
| 2020-04-01 | Revenue trend begins declining; attributed by business team to "macro conditions" |
| 2020-05-01 | Engineering team starts new A/B test: "Model v2" vs existing production model |
| 2020-06-01 | A/B test results show Model v2 delivers +0% improvement vs baseline |
| 2020-06-08 | Analyst discovers baseline (production model) is significantly underperforming historical benchmarks |
| 2020-06-10 | Feature drift analysis begins; `hour_of_day` PSI = 0.38 (well above meaningful threshold) |
| 2020-06-12 | Incident declared; root cause confirmed as model drift |
| 2020-06-26 | Model retrained on recent 60-day data; revenue recovers to baseline |

---

## What Happened (Technical)

The recommendation model was trained on 18 months of historical user interaction data ending February 2020. `hour_of_day` was the 12th most important feature by gain, representing when users typically browse — heavily weighted toward commute hours (7-9am, 5-7pm) and weekend patterns.

When COVID stay-at-home orders began in mid-March, user behavior shifted dramatically. Commute-time browsing collapsed; midday and evening traffic increased. The feature distribution for `hour_of_day` shifted from a bimodal commute-hour pattern to a broad midday-centered distribution. The model continued scoring users with time-of-day assumptions that no longer reflected reality, producing systematically miscalibrated rankings.

The monitoring system was configured to track PSI (Population Stability Index) on 5 features selected at model launch: `user_age`, `session_length`, `device_type`, `purchase_count_30d`, and `category_affinity`. None of these features drifted significantly. Feature #12 (`hour_of_day`) was not in the monitored set and showed PSI=0.38 — well above the 0.2 flag threshold — but was never checked. Additionally, no offline accuracy monitoring existed: the team was not computing model performance on a held-out labeled sample on any regular cadence.

The degradation remained invisible for 90 days because revenue was also suppressed by macro conditions (COVID economic uncertainty), making it impossible to attribute the 8% shortfall to the model. The signal that finally caught it was an A/B test that should have shown improvement but showed zero lift — because the "control" group (production model) was already severely degraded.

---

## Root Cause Analysis

**Contributing factors:**
1. PSI monitoring covered only 5 of the top 30 features; drift in feature #12 went undetected
2. PSI threshold set at 0.2 (common default); the actual alert-worthy level for this feature should have been 0.1
3. No model accuracy monitoring: no held-out labeled sample with ground-truth labels computed on a weekly basis
4. No shadow model running: the team had no "always-on" comparison to detect relative degradation
5. Revenue attribution was confounded by external macro factors, preventing human detection
6. A/B test was measuring lift over a degraded baseline, not over a fixed historical anchor

**5 Whys:**

Why did revenue fall 8%?
The recommendation model was ranking products poorly for users with shifted behavior.

Why was the model ranking poorly?
The `hour_of_day` feature distribution shifted dramatically but the model used pre-COVID assumptions.

Why wasn't the drift detected?
PSI monitoring only tracked 5 features; `hour_of_day` (feature #12) was not monitored.

Why were only 5 features monitored?
At model launch, the team chose "the most important features" without a systematic methodology; capacity for monitoring setup was limited.

Why was there no fallback signal?
No accuracy monitoring on held-out data was in place, and no shadow model existed to surface comparative degradation automatically.

---

## What Went Well

- The A/B test framework existed and was being used — even though it caught the problem indirectly
- The post-hoc feature drift analysis was fast once triggered (~2 days to confirm root cause)
- Retraining infrastructure was in place and retraining was completed quickly once triggered
- The team had a model registry with the original training data snapshot, enabling clean root cause analysis

---

## Action Items

| Item | Owner | Due | Status |
|------|-------|-----|--------|
| Expand PSI monitoring to all top-30 features (not just top-5) | ML Infra | +2 weeks | Done |
| Reduce PSI alert threshold from 0.2 to 0.1 for time-based features | ML Infra | +1 week | Done |
| Implement weekly accuracy eval on 10K held-out labeled samples | ML Platform | +3 weeks | Done |
| Deploy always-on shadow model (retrained 30-day rolling window) | ML Platform | +6 weeks | In progress |
| Add A/B test guardrail: compare to historical anchor, not just current control | Experimentation | +4 weeks | Done |
| Document PSI monitoring methodology and feature selection criteria | Documentation | +2 weeks | Done |

---

## Interview Discussion Points

**What would you have done differently?**
Monitor ALL top-30 features for drift, not a manually curated subset. Use a systematic threshold (e.g., features with importance > 1%) rather than subjective selection. Add a fixed historical anchor for A/B comparisons so a degraded control doesn't hide lift.

**How would you prevent this category of failure (silent model drift)?**
Three defenses: (1) comprehensive feature drift monitoring with tiered thresholds (0.1 for high-importance, 0.2 for low-importance features), (2) weekly offline accuracy eval on a labeled holdout sample that gives you an absolute quality signal, (3) a shadow model always running on a rolling retrain schedule so you have a "freshness baseline" for comparison.

**What monitoring gaps does this reveal?**
The biggest gap is relying only on operational metrics (latency, error rate) for model health. Silent drift degrades model quality with no operational signal. You need a separate "model quality monitoring" layer that checks prediction distributions, feature distributions, and periodic accuracy evaluation on fresh labeled data.

**What is PSI and when does it fail as a drift signal?**
PSI (Population Stability Index) = Σ (actual% - expected%) * ln(actual%/expected%) compares current distribution to reference distribution. PSI < 0.1 = no change; 0.1–0.25 = moderate change (investigate); > 0.25 = major shift. It fails when: (a) you're not monitoring the right features, (b) the threshold is too loose, (c) the feature shifts in a way that PSI doesn't capture (e.g., correlation changes without marginal distribution change).
