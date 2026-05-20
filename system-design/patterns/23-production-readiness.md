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

**Scenario 1: Skipped load testing, model crashes at peak traffic**
- Load testing skipped to save 1 day. Model handles 100 req/sec offline, crashes at 500 req/sec in production.
- Cascading failure: depends on model, entire system down.
- Fix: Load testing mandatory. Test at 2x peak traffic. If fails, optimize before shipping.

**Scenario 2: Monitoring not in place until day 3**
- Model ships. Works fine day 1-2. Day 3, accuracy drops silently (no monitoring). Discovered when customer complains.
- Fix: Monitoring must be in place BEFORE production rollout. Canary phase validates monitoring.

**Scenario 3: No fallback model**
- Model fails (OOM, bug). No fallback. System completely broken. Recovery 1+ hour.
- Fix: Fallback required (previous model, rule-based, baseline). Can switch in <5 minutes.

**Scenario 4: Runbook incomplete**
- Model fails in production. On-call doesn't know how to debug (where are logs? what metrics matter?).
- Incident takes 3 hours to resolve (should be 15 min with good runbook).
- Fix: Runbook written before deployment. Tested during on-call training. Updated post-incident.

---

## Implementation Guidance

**Wrong:** Ship model when "accuracy is good". Hope monitoring catches issues.
**Right:** Complete full checklist. Deploy with canary. Monitor 1 week before declaring success.

**Wrong:** Readiness check as checklist, skip items if pressed for time.
**Right:** Readiness check as gate. All items required, no exceptions. Time the checklist properly.

---

## Sophisticated Interview Q&A

**Q1: Model ready for production in 2 weeks. Can readiness check finish in 1 week?**
A: Depends on parallelization. (1) Testing (latency, load) in parallel: 5 days. (2) Monitoring setup: 2 days. (3) Documentation: 2 days. (4) Reviews: 2 days. Total: 1 week achievable if planned early. But if sequential, takes 1-2 weeks. Plan ahead.

**Q2: Load test shows model handles 100 req/sec, peak is 80. Ready?**
A: Not quite. (1) 100 req/sec = 1.25x peak. Acceptable for temporary spikes, not sustained. (2) Better: handle 2x peak = 160 req/sec. (3) Also check: tail latency (p99) at 100 req/sec. If p99 OK, maybe acceptable. (4) Plan auto-scaling for seasonal peaks.

**Q3: Fallback model is 6 months old, accuracy 5% worse. Use it?**
A: Yes. Better to serve 5% worse accuracy than no model at all (complete outage). Fallback is safety net, not ideal. Once on fallback, start recovery (debug new model, deploy when ready).

**Q4: Readiness check takes 2 weeks, product wants to ship in 1 week. What do you cut?**
A: Prioritize by risk: (1) Don't cut monitoring, load testing, fallback (highest risk if missing). (2) Can defer runbook documentation (write during stabilization week). (3) Can cut SRE review (do peer review instead). (4) Can do simpler load test (test to 1.5x peak instead of 2x). Trade-off: ship in 1 week with 10% incident risk vs 2 weeks with 1% risk.

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

## Interview Q&A
**Q: All tests pass, accuracy 95%. Ready to ship?**
A: No. Checklist: latency SLA? Monitoring? Fallback? Alerting? Runbook? Load tested? If not, complete checklist first.

**Q: Timeline: 2 weeks to ship model. Spend 1 week on readiness check?**
A: Yes. 1 week prevents 1-month incident. Readiness check investment pays off instantly if issue caught early.

## Interview Quick-Reference
| Item | Status | Action |
|------|--------|--------|
| Accuracy | 95% | ✓ Pass |
| Latency | 150ms SLA | ✓ Pass |
| Monitoring | Configured | ✓ Pass |
| Fallback | Exists | ✓ Pass |

## Related Topics
- [Monitoring](16-monitoring-and-observability.md)
- [Deployment Safety](11-blue-green-deployment.md)

## Resources
- [ML System Design Checklist](https://github.com/stanford-cs329s/ml-systems-design-course)
