# Post-Mortem: Training-Serving Skew at Model Launch

## Incident Summary
**Date:** 2023-09-05
**Duration:** 48 hours to detect (user complaints), 6 hours to diagnose, 4 hours to resolve
**Business Impact:** CTR dropped from 4.2% (expected) to 1.9%; estimated $800K revenue loss over 2 days
**Severity:** P1 (Production model severely underperforming; user-facing quality impacted)

---

## Timeline

| Time | Event |
|------|-------|
| 2023-08-01 | Model v2 training begins on 6 months of historical data |
| 2023-08-28 | Offline evaluation: AUC = 0.940 (vs 0.880 baseline); launch approved |
| 2023-09-04 | Canary launch begins: 5% of production traffic routed to Model v2 |
| 2023-09-04 | Operational metrics (latency, error rate) all nominal |
| 2023-09-05 | Full traffic cutover to Model v2 (100%) |
| 2023-09-05 09:00 | User feedback begins arriving: recommendations feel random, irrelevant |
| 2023-09-06 08:00 | Support ticket volume 3x normal; PM escalates to engineering |
| 2023-09-06 10:00 | On-call engineer begins investigation; AUC check on production logs shows 0.71 |
| 2023-09-06 11:30 | Feature value comparison: production `income_normalized` mean=0.03, training mean=0.51 |
| 2023-09-06 14:00 | Root cause confirmed: scaler artifact mismatch |
| 2023-09-06 18:00 | Production scaler replaced with training scaler artifact; AUC recovers to 0.93 |

---

## What Happened (Technical)

During training, a `StandardScaler` was fitted on the full training dataset (6 months, 2.1M rows) and applied to normalize all numerical features including `income_normalized`, `session_time_scaled`, and `age_normalized`. The training pipeline stored the fitted scaler in memory and used it to transform features before feeding them to the model.

When the model was deployed to production, the serving pipeline needed a scaler to normalize incoming request features in real time. The engineer who wrote the serving code assumed the scaler should be refitted on "recent" data for freshness. They wrote a startup routine that fitted a new `StandardScaler` on the last 1,000 production requests when the serving pod started.

The result: the training scaler had mean(income)=52,400 and std(income)=28,600 (computed on 2.1M users). The production scaler had mean(income)=41,200 and std(income)=15,800 (computed on 1,000 recent users who skewed toward high-frequency shoppers). The normalized feature values were systematically different. The model received inputs outside the range it was trained on, producing miscalibrated predictions with AUC degrading from 0.94 to 0.71.

Critically, the canary stage did not catch this because the 5% canary segment had the same scaler mismatch as production — both serving environments used the incorrectly re-fitted scaler. The parity test in staging also missed it because staging was configured with a fresh full-dataset scaler (different environment setup than production).

---

## Root Cause Analysis

**Contributing factors:**
1. Scaler artifact was not saved with the model; serving code independently re-fitted the scaler
2. Scaler was re-fitted on a non-representative 1,000-sample window (recent high-frequency users) vs. 2.1M training samples
3. Canary test did not include a "parity check": verify that feature distributions after normalization match training-time distributions
4. Staging environment had different scaler configuration than production, so staging tests did not catch the mismatch
5. No automated comparison of training-time vs. serving-time feature statistics was run at deploy time

**5 Whys:**

Why did AUC drop from 0.94 to 0.71 in production?
The model received feature values with different scale/distribution than it was trained on.

Why were feature values at wrong scale?
The `StandardScaler` fitted in production used different parameters (mean, std) than the training scaler.

Why did the serving pipeline have a different scaler?
The serving code independently re-fitted the scaler at startup rather than loading the scaler artifact saved during training.

Why wasn't the scaler artifact saved and loaded?
The model artifact (pickled sklearn model) was saved; the scaler was treated as a "preprocessing step," not as part of the model artifact, and no explicit handoff process existed.

Why did the canary and staging tests miss this?
Canary had the same mismatch; staging used a different (correct) scaler setup. No automated parity assertion compared normalized feature statistics between training and serving environments.

---

## What Went Well

- On-call engineer had access to production feature logs, enabling quick comparison of production vs. training feature distributions
- Model registry retained the training run metadata, making it easy to retrieve the exact training dataset statistics
- Resolution was fast once root cause was found: 4 hours from diagnosis to fix
- The incident created strong momentum to fix the artifact management process

---

## Action Items

| Item | Owner | Due | Status |
|------|-------|-----|--------|
| Save scaler (and all preprocessing transformers) as part of the model artifact bundle | ML Infra | +1 week | Done |
| Add "parity test" to deployment checklist: compare mean/std of normalized features in staging vs training | ML Platform | +2 weeks | Done |
| Add feature distribution check to serving startup: log mean/std per feature, alert if >2 sigma from training baseline | ML Infra | +3 weeks | Done |
| Audit all existing production models for scaler artifact mismatch | ML Platform | +4 weeks | Done (2 other models found with same issue) |
| Mandatory staging-production config parity check in CI/CD pipeline | DevOps | +3 weeks | In progress |

---

## Interview Discussion Points

**What would you have done differently?**
Treat ALL preprocessing steps (scalers, encoders, imputers) as part of the model artifact. Save them alongside the model weights and load them atomically at serving time. Use an sklearn `Pipeline` object or a serving framework like BentoML/Seldon that enforces this pattern. Add a parity test that runs at deploy time: serialize 1,000 sample features, transform them with the saved scaler, and assert their distribution matches training-time statistics.

**How would you prevent training-serving skew?**
Four practices: (1) artifact bundling — scaler + model in one file (pickle, ONNX with preprocessing, or MLflow Model format), (2) parity test in CI — identical code path from raw feature to model input in both training and serving, (3) feature validation at serving time — log normalized feature stats and alert on drift from training baseline, (4) shadow mode before cutover — run new model on shadow traffic and compare feature values to production model's feature values for the same request.

**What monitoring gaps does this reveal?**
Operational metrics (latency, error rate) are insufficient for catching model quality regressions. You need a "model input health" monitoring layer that checks: (a) raw feature null rates, (b) feature value distributions after preprocessing, (c) prediction score distributions. A sudden shift in any of these is an early warning signal before user-facing metrics degrade.

**Why did the 5% canary miss this?**
Canary testing validates the new model handles load correctly and doesn't crash — it doesn't validate model quality unless you explicitly measure business KPIs during the canary phase. A 5% canary over 24 hours may not generate enough statistical power to detect a 50% AUC regression, especially if the canary segment is not representative. The lesson: include model quality metrics (AUC on logged predictions with delayed labels, or prediction score distribution checks) in canary acceptance criteria.
