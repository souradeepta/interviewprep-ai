# Production Readiness

## TL;DR
Checklist before shipping model: monitoring in place, fallback model exists, latency SLA met, accuracy validated on holdout set, alerting configured, runbook for incidents.

## Core Intuition
"Model looks good" ≠ "ready to ship". Production readiness = model works + system robust to failures.

## How It Works

**Production readiness checklist:**
- [ ] Accuracy validated on holdout set (test accuracy ≥ threshold)
- [ ] Monitoring in place (latency, errors, predictions logged)
- [ ] Fallback model exists (previous version, rule-based)
- [ ] SLA met (latency p99 < requirement, uptime 99.9%)
- [ ] Alerting configured (latency spike, error rate, drift)
- [ ] Runbook written (how to debug, how to rollback)
- [ ] Load testing done (can it handle peak traffic?)
- [ ] Security review (no data leaks, sanitized inputs)

## Key Properties / Trade-offs
- Time: readiness check takes days (testing, review, deployment prep)
- Safety: worth the time investment (prevents production incidents)

## Detailed Trade-off Analysis

| Readiness Level | Checklist Items | Time to Deploy | Incident Risk | Cost of Incident |
|-----------------|-----------------|----------------|---------------|-----------------|
| None (ad-hoc) | 0-2 items | 1 day | 50% (high) | $100K+ |
| Basic | 4-5 items | 3-5 days | 20% (medium) | $20K |
| Standard | 7-8 items (all) | 1-2 weeks | 5% (low) | $5K |
| Enterprise | 8 items + SRE review | 2-3 weeks | 1% (very low) | $500 |

**Decision:** Startup MVP → basic. Series A+ → standard. Regulated/critical → enterprise.

---

## Production Failure Scenarios

**Failure 1: Load Test Skipped to Meet a Deadline**
- **Symptom:** Model crashes at launch under real traffic (3× the load seen in development). The entire product is down for 90 minutes during the highest-traffic window of the week.
- **Root cause:** Load testing was deprioritized to save a single day. The model was tested at 50 QPS in development but received 150 QPS at launch.
- **Detection:** After the fact, the p99 latency graph shows a vertical cliff at the exact moment traffic exceeded 2× dev load. Pre-launch: mandatory load test at 2× expected peak before each deployment.
- **Fix:** Add load testing as a hard gate in the deployment checklist. The deployment is blocked until the model sustains p99 latency below SLA at 2× expected peak traffic for 10 minutes continuously.

**Failure 2: Rollback Procedure Untested**
- **Symptom:** A critical model bug requires rollback during a live incident. The documented rollback procedure takes 4 hours to execute instead of the expected 15 minutes. Incident duration triples.
- **Root cause:** The rollback runbook was written at initial deployment 18 months ago and never tested. Several steps reference infrastructure that was reorganized, and the model registry API changed.
- **Detection:** Quarterly rollback drill: simulate a production incident, execute the rollback, measure wall-clock time. Gate production access on drill completion.
- **Fix:** Test the rollback procedure quarterly. Measure actual time. If actual time exceeds target (15 minutes) by more than 2×, update the runbook and re-test. The rollback SLA must be part of the deployment contract.

**Failure 3: No Fallback Logic for Model Endpoint Failures**
- **Symptom:** Model endpoint becomes unavailable (OOM error spikes, pod crashes). The entire product is broken because all code paths flow through the model with no alternative.
- **Root cause:** The engineering team assumed the model would be highly available and did not implement a fallback path. There is no rule-based system, no cached-prediction layer, and no graceful degradation mode.
- **Detection:** Chaos engineering: terminate the model endpoint container and observe product behavior. If the product is immediately broken, the fallback is absent.
- **Fix:** Before any launch, implement and test at least one fallback path: cached predictions for the most common inputs, a lightweight rule-based heuristic, or a "degraded mode" that returns a reasonable default. The fallback must be exercised in production at least quarterly to verify it still works.

**Failure 4: Output Schema Mismatch at Deployment**
- **Symptom:** New model version deployed. Client error rate spikes to 40% within two minutes of deployment. Rollback takes 15 minutes; ~300K requests fail.
- **Root cause:** The new model version changed its output schema (added a field, changed a field name) without updating the API contract. Client code expected the old schema.
- **Detection:** Contract tests that compare output schema between model versions automatically block deployment if the schema changes are backward-incompatible.
- **Fix:** Implement schema versioning for model outputs. Any breaking schema change requires a new API version and client migration, not a silent model swap. Automated contract tests run in the deployment pipeline and fail the deployment if schema compatibility is broken.

---

## Implementation Guidance

**Wrong:** Ship model when "accuracy is good". Hope monitoring catches issues.
**Right:** Complete full checklist. Deploy with canary. Monitor 1 week before declaring success.

**Wrong:** Readiness check as checklist, skip items if pressed for time.
**Right:** Readiness check as gate. All items required, no exceptions. Time the checklist properly.

---

## Interview Q&A

**Q1: The product team wants to ship in one week, but your readiness checklist takes two. What do you cut?**
A: Prioritize by failure probability × failure cost. Never cut: load testing (high-probability failure, high cost), fallback model existence (makes complete outages possible), and monitoring setup (makes all other failures invisible). Cuttable: full SRE review can become a peer review by a senior engineer. Runbook documentation can be deferred to the stabilization week (write it while monitoring the canary, not before). Load testing threshold can be reduced from 2× peak to 1.5× peak with explicit sign-off. The result is a week-one release with a 10% incident risk instead of a week-two release with a 1% risk — document that trade-off and get explicit product approval.

**Q2: Load test shows the model handles 100 req/sec and your peak is 80. Is it ready?**
A: Borderline. 100/80 = 1.25× peak headroom is adequate for brief traffic spikes but not for sustained bursts or slow traffic growth. The more important question is p99 latency at 100 req/sec, not just throughput. If p99 at 80 req/sec is 90ms and your SLA is 100ms, you have 10ms of headroom — any spike to 100 req/sec will breach SLA. The right standard is: sustain 2× peak (160 req/sec) with p99 < SLA for 10 continuous minutes. If that fails, the model is not ready.

**Q3: The fallback model is six months old and 5% less accurate than the current model. Is it a valid fallback?**
A: Yes. A 5% worse model serving 100% of requests is always better than no model serving 0% of requests during an outage. The fallback's job is to prevent complete degradation, not to be competitive. Once the fallback is active, begin recovery procedures immediately (debug the new model, deploy when root cause is understood). The fallback should not be active for more than 24 hours before you escalate. And after every fallback activation, update the runbook with what triggered it.

**Q4: When would you NOT do a full production readiness check?**
A: For an internal tool with no SLA, used by fewer than 10 engineers, where an outage has near-zero business cost. For a model update that only changes a non-functional parameter (description text, logging label) with no code or weight changes. For a model in a development environment that is explicitly labeled "not for production use." In regulated industries (payments, healthcare), a full readiness check is mandatory regardless of these conditions — there is no exception.

**Q5: How do you automate production readiness checks in a CI/CD pipeline?**
A: Four automated gates: (1) Accuracy gate — run the test suite on the held-out dataset and fail the deployment if accuracy is below the threshold. (2) Latency gate — run a synthetic load test (using k6 or Locust) at 2× expected peak and fail if p99 exceeds SLA. (3) Schema contract gate — compare the new model's output schema against the registered consumer contracts and fail on breaking changes. (4) Monitoring gate — verify that the monitoring dashboards and alert rules for the new model version are present in the monitoring system before the deployment is allowed to proceed. Items like runbook documentation and security review are automated as checklist validation (requires sign-off from designated reviewers).

**Q6: A model is deployed with canary rollout (10% of traffic). After one hour, accuracy looks good. Safe to ramp to 100%?**
A: Not yet. One hour is sufficient to detect crash-level failures but not subtle accuracy degradation, seasonal effects, or rare cohort issues. The minimum canary duration for a meaningful signal depends on traffic volume: at 10% of traffic with 1K requests/minute, you have 6K requests in one hour — enough for statistical significance on aggregate metrics but not on rare cohorts. A more robust standard is to run the canary until each important cohort (by geography, user type, device type) has at least 500 requests — this often takes 6-24 hours. Only after per-cohort metrics are stable do you ramp to 100%.

**Q7: What monitoring must be in place before production rollout, not after?**
A: Three categories that are non-negotiable pre-rollout: (1) Availability monitoring — is the model endpoint responding? (health check on `/predict` every 30 seconds). (2) Error rate monitoring — is the error rate above baseline? (alert if error rate > 1% for 5 consecutive minutes). (3) Prediction distribution monitoring — is the model producing outputs in the expected range? (alert if the fraction of predictions in the expected range drops below 95%). These three cover the most common and fastest-moving failure modes. Drift monitoring, business metric correlation, and fairness monitoring can be set up in the first week after launch but should not block rollout.

**Q8: What does a good runbook include, and how often should it be tested?**
A: A good runbook includes: (1) The list of symptoms that trigger the runbook (not "if something is wrong" but specific alert names and metric thresholds). (2) The decision tree for diagnosis: follow these steps in order to identify whether the problem is model accuracy, infrastructure, or data quality. (3) Rollback instructions with exact commands and expected completion time. (4) Escalation contacts if the runbook doesn't resolve the incident. (5) Links to dashboards, logs, and model registry entries. The runbook should be tested quarterly via tabletop exercise (team walks through a simulated incident) or live drill (on-call executes rollback in staging). If the tested resolution time is more than 2× the target, update the runbook before the next deployment.

---

## Cost & Resource Analysis

**Readiness check investment:**
- Testing (load, integration, stress): 3 days = $3K engineer time
- Monitoring setup: 2 days = $2K
- Documentation (runbook): 2 days = $2K
- Reviews (security, design): 2 days = $2K
- **Total: ~$9K**

**Cost of production incidents (if skipped):**
- Incident response: 2-4 hours = $2K
- Service downtime: $100K-1M depending on criticality
- Reputation: hard to quantify, but significant
- **Total: $100K+ per incident**

**ROI:** Readiness investment $9K prevents incidents worth $100K+. Break-even: 1 prevented incident per year.

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Load testing infrastructure | $0.10/hr | 40 hr/release | $4 per release |
| Staging environment (always-on) | $2/hr | 720 hr | $1,440 |
| Chaos engineering tooling | $500/mo | 1 subscription | $500 |
| Release engineer time | $200/hr | 4 hr/release | $800 per release |
| **Total (2 releases/month)** | | | **~$3,548/month** |

At $3,548/month, the staging environment dominates (41%) followed by release engineer time (45% for 2 releases). The load-testing compute itself is almost free ($8/month for two releases). The key insight: the cost of production readiness is mostly people, not infrastructure. Reducing engineer time per release (via automation, standardized checklists, CI/CD gates) is the main lever. A well-automated pipeline can cut release engineer time from 4 hours to 1 hour, reducing the total to ~$1,948/month. One prevented production incident (average cost $100K) pays for 2.5 years of this readiness investment.

---

## Monitoring & Observability

**Pre-deployment metrics to track:** Checklist completion (% items done), test coverage (% code paths tested), load test pass rate, latency/accuracy at various load levels

**Post-deployment metrics:** Time to identify issue (alert latency), time to resolution (MTTR), incident frequency, fallback activation rate, runbook effectiveness (time to resolution)

## Common Mistakes / Gotchas
- Skipping checklist: "it works locally" → ships broken
- Incomplete monitoring: can't debug issues in production
- No fallback: model fails → complete outage
- False confidence: passed checklist but underlying issue exists

## Best Practices
- **Automated checks:** CI/CD pipeline verifies checklist items
- **Manual review:** domain expert signs off before prod
- **Gradual rollout:** start with canary, monitor, then full
- **Post-deployment:** monitor for 1 week before declaring success

## Code Example
```python
def production_readiness_check():
    checks = {
        "test_accuracy": test_accuracy >= 0.90,
        "latency_p99": latency_p99 < 100,  # ms
        "uptime": uptime >= 0.999,
        "monitoring": monitoring_enabled,
        "fallback_exists": fallback_model is not None,
        "alerts_configured": len(alerts) > 0
    }
    
    all_pass = all(checks.values())
    if not all_pass:
        print("Production readiness check FAILED")
        for check, result in checks.items():
            print(f"  {check}: {'✓' if result else '✗'}")
    
    return all_pass
```

## Interview Quick-Reference
| Checklist Item | Min Requirement | Block Deployment? |
|----------------|-----------------|-------------------|
| Accuracy on holdout | >= defined threshold | Yes |
| p99 latency at 2x peak | < SLA | Yes |
| Fallback model exists | Any fallback path | Yes |
| Monitoring active | Availability + error rate | Yes |
| Rollback tested | Completed within 15 min | Yes |
| Runbook written | Covers top 3 failure modes | Recommended |
| Schema contract tests | No breaking changes | Yes |

## Related Topics
- [Monitoring](16-monitoring-and-observability.md)
- [Deployment Safety](11-blue-green-deployment.md)

## Resources
- [ML System Design Checklist](https://github.com/stanford-cs329s/ml-systems-design-course)
