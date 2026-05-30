# Post-Mortem: Cascade Failure from Miscalibrated Model Confidence

## Incident Summary
**Date:** 2023-11-14
**Duration:** 2 hours of total expert-path outage; 45 minutes of degraded fast-classifier serving
**Business Impact:** 100% of out-of-distribution queries received wrong predictions for 2 hours; fast classifier served incorrect responses to OOD inputs for 45 minutes; estimated 12,000 affected user sessions
**Severity:** P1 (Cascade failure, partial outage, incorrect model predictions served at scale)

---

## Timeline

| Time | Event |
|------|-------|
| 2023-11-01 | Two-model cascade deployed: fast classifier (XGBoost) + slow specialist (fine-tuned BERT) |
| 2023-11-01 | Fast classifier routes queries scoring confidence > 0.9 to itself; confidence < 0.9 to specialist |
| 2023-11-10 | System operating normally; specialist handling ~15% of traffic |
| 2023-11-14 09:00 | Marketing campaign launches; drives unusual query patterns (OOD inputs) at 8x normal volume |
| 2023-11-14 09:00 | Fast classifier assigns confidence=0.98 to all OOD inputs (miscalibrated) |
| 2023-11-14 09:00 | OOD inputs NOT routed to specialist (confidence too high); fast classifier serves wrong predictions |
| 2023-11-14 09:15 | Specialist load drops to near zero; no alert fires (unusual but not alarming) |
| 2023-11-14 09:45 | Marketing campaign continues; volume increases |
| 2023-11-14 09:45 | Fast classifier pods overwhelmed by volume; some OOD inputs finally spill to specialist |
| 2023-11-14 09:50 | Specialist pod overwhelmed by sudden burst (no warm-up); specialist crashes (OOM) |
| 2023-11-14 09:55 | Fast classifier circuit breaker not configured for specialist fallback; all specialist-bound requests fail |
| 2023-11-14 10:00 | Fast classifier serving all traffic with wrong predictions; specialist down |
| 2023-11-14 11:00 | On-call engineer alerted; investigation begins |
| 2023-11-14 11:45 | Specialist pods restarted with correct resource limits; traffic stabilizing |
| 2023-11-14 12:00 | System fully recovered |

---

## What Happened (Technical)

The system was designed as a two-stage cascade: a fast XGBoost classifier handled high-confidence cases (> 0.9 threshold) directly, while a slow but more accurate BERT specialist handled edge cases (< 0.9 confidence). This pattern is common in production ML systems to balance throughput and quality.

The fast classifier's confidence calibration was never validated after deployment. In training, confidence scores were well-calibrated on in-distribution data. However, for out-of-distribution inputs — queries with word patterns not seen during training — the XGBoost model produced systematically high confidence scores (around 0.97-0.99). This is a well-known failure mode: tree-based models extrapolate constant predictions in feature space regions not covered by training data, and the resulting predictions are high-confidence by construction.

When the marketing campaign drove unusual query patterns (OOD inputs at 8x volume), the fast classifier intercepted all of them with confidence = 0.98, preventing them from reaching the specialist. The fast classifier served incorrect predictions for 45 minutes before its pods became overloaded.

When a fraction of OOD traffic finally spilled over to the specialist (due to fast classifier pod saturation), the specialist received a sudden burst it hadn't seen before. Because the specialist was cold (no warm-up traffic), its pod resource limits were set for steady-state load. The burst overwhelmed the specialist's pods, causing OOM crashes.

With the specialist down, the fast classifier's circuit breaker was not configured to degrade gracefully — it propagated the failure upward rather than falling back to cached results or simplified predictions.

---

## Root Cause Analysis

**Contributing factors:**
1. Fast classifier confidence was never calibrated for out-of-distribution inputs; OOD always gets high confidence from tree models
2. No OOD detection mechanism existed; the system had no way to flag inputs outside the training distribution
3. Specialist pod autoscaling was slow; no pre-warming mechanism for sudden traffic bursts
4. Circuit breaker for specialist failure was not configured; fast classifier had no graceful degradation path
5. Confidence threshold A/B test was never conducted; 0.9 was set arbitrarily at launch
6. Specialist load monitoring did not alert on the drop to near-zero (could have been an early warning signal)

**5 Whys:**

Why did the cascade fail and serve wrong predictions for 2 hours?
The specialist (which could handle OOD inputs correctly) was first bypassed, then overwhelmed and crashed.

Why was the specialist bypassed?
Fast classifier assigned confidence=0.98 to all OOD inputs, so they never met the routing threshold to the specialist.

Why did the fast classifier assign high confidence to OOD inputs?
Tree-based models produce constant-region predictions outside training data; confidence calibration was never validated for OOD inputs.

Why wasn't OOD detection in place?
OOD detection (e.g., Mahalanobis distance, ensemble disagreement, or input density estimation) was not part of the original system design.

Why wasn't the specialist protected against sudden load bursts?
Autoscaling was configured for steady-state load; no pre-warming or burst-buffer capacity planning was done for the specialist.

---

## What Went Well

- The on-call engineer was alerted through user-facing error monitoring (error rate spike)
- Specialist pods were recoverable (OOM, not data corruption); restart time was 45 minutes
- Post-incident, the team had detailed request logs to reconstruct the failure sequence

---

## Action Items

| Item | Owner | Due | Status |
|------|-------|-----|--------|
| Monthly confidence calibration check: compute ECE (Expected Calibration Error) on production data | ML Infra | +2 weeks | Done |
| Add OOD detection layer: use Mahalanobis distance from training distribution; flag high-distance inputs | ML Research | +4 weeks | In progress |
| Implement confidence threshold A/B test: run 0.7, 0.8, 0.9 thresholds and measure specialist load vs quality | Experimentation | +3 weeks | Done |
| Configure circuit breaker with graceful degradation: fast classifier falls back to cached top-K if specialist is down | ML Infra | +2 weeks | Done |
| Add specialist load monitoring alert: alert if specialist traffic drops to near-zero for > 5 minutes | ML Infra | +1 week | Done |
| Add pre-warming policy for specialist: maintain minimum 2 warm pods regardless of traffic | ML Platform | +2 weeks | Done |

---

## Interview Discussion Points

**What would you have done differently?**
Design OOD detection into the cascade from the start. The fast classifier should not route inputs based solely on confidence score — it should also check an OOD signal (e.g., Mahalanobis distance from training distribution centroids, or an input density model). High-OOD inputs always go to the specialist regardless of confidence score. Also: validate confidence calibration on held-out OOD examples before launch, not just on the in-distribution test set.

**How would you prevent cascade failures in multi-model systems?**
Four patterns: (1) **graceful degradation at every stage** — each model should have a fallback (cached results, simplified model, or "I don't know" response), (2) **load shedding** — specialist should have a request queue with a max size; overflow is handled by the fast classifier, not ignored, (3) **steady-state pre-warming** — specialist maintains minimum pod count regardless of traffic, (4) **circuit breaker with timeout** — if specialist is down, fast classifier serves best-effort predictions rather than failing. All of these must be tested in chaos engineering exercises.

**What monitoring gaps does this reveal?**
Two critical gaps: (1) **model health monitoring beyond latency/errors** — specialist load drop to near-zero is a signal that the routing logic is broken, not just a good sign; unusual traffic distribution should trigger investigation, (2) **confidence calibration monitoring** — ECE (Expected Calibration Error) should be computed regularly on production data, not just at training time. A well-calibrated model in training can be miscalibrated in production as distribution shifts.

**What is Expected Calibration Error (ECE) and how do you fix miscalibration?**
ECE = Σ (|B_m|/n) * |acc(B_m) - conf(B_m)| where predictions are grouped into confidence bins, and we measure how much the mean confidence deviates from actual accuracy per bin. Well-calibrated: confidence=0.9 means 90% actual accuracy. Fix miscalibration: (1) temperature scaling — divide logits by scalar T (T>1 softens overconfident predictions); (2) Platt scaling — fit a logistic regression on confidence vs. accuracy; (3) isotonic regression — non-parametric calibration. Apply calibration on a held-out calibration set, never on the training set.
