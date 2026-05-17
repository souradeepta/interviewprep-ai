# Production Readiness

## TL;DR
Checklist before deploying ML to production: versioning, monitoring, fallbacks, SLA, disaster recovery, documentation, testing.

## Core Intuition
ML experiments work, but production is hard. Checklist prevents fires.

## How It Works
**Pre-deployment checklist:**
- Model versioning: can identify, rollback
- Monitoring: latency, accuracy, drift alerts
- Fallback: if model fails, have backup
- SLA defined: latency, availability targets
- Testing: load test, edge cases
- Documentation: team knows what it does
- Disaster recovery: plan if system down

**Post-deployment:**
- Monitor continuously
- Respond to alerts
- Retraining schedule
- Regular audits

## Common Mistakes / Gotchas
- **Ship and forget:** no monitoring → issues go undetected
- **No fallback:** service fails → bad user experience
- **Poor documentation:** knowledge leaves with engineer

## Interview Quick-Reference
**Production ready?** Versioning, monitoring, fallbacks, SLA, testing, docs, disaster recovery.

## Related Topics
- [Monitoring & Observability](monitoring-and-observability.md)
- [Disaster Recovery](disaster-recovery.md)

## Resources
- [ML Systems in Production](https://www.oreilly.com/library/view/machine-learning-systems-design/9781492072929/)
