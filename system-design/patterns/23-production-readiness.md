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
