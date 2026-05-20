# Disaster Recovery

## TL;DR
Plan for failures: model server down (fallback model), data pipeline broken (use cached data), entire region down (multi-region setup). RTO (recovery time), RPO (data loss tolerance).

## Core Intuition
Disaster = model server down. Users see errors. RTO: how long before users see results again? RPO: how much data lost? Plan accordingly.

## How It Works

**Disaster recovery levels:**

1. **Service degradation (RTO=minutes):**
   - Model server down → use fallback rule-based model
   - Users see worse predictions, but service available

2. **Failover (RTO=seconds):**
   - Active-passive: hot standby server takes over
   - Multi-region: traffic routed to healthy region

3. **Full recovery (RPO=hours):**
   - Restore from backup
   - Replay logs to recover recent data

| Scenario | RTO | RPO | Cost |
|----------|-----|-----|------|
| Fallback rule-based | 0 | 0 | Low |
| Multi-region | 1s | <1min | High |
| Backup restore | 1h | 1h | Medium |

## Key Properties / Trade-offs
- Cost vs RTO: faster recovery costs more
- Complexity vs RTO: simple fallback vs complex multi-region
- Testing: DR untested won't work when needed

## Detailed Trade-off Analysis

| Recovery Strategy | RTO | RPO | Cost | Complexity | Tested Frequency |
|------------------|-----|-----|------|-----------|------------------|
| Fallback model | 0 | 0 | Low | Low | Daily |
| Multi-AZ (same region) | 10s | 1min | Medium | Medium | Monthly |
| Multi-region active-passive | 1-5min | 5min | High | High | Quarterly |
| Multi-region active-active | <1s | <1s | Very high | Very high | Weekly |
| Backup restore | 1-4h | 1h | Low | Medium | Quarterly |

**Decision:** Startup → fallback + backup. Series A → multi-AZ. Unicorn/critical → multi-region active-active.

---

## Production Failure Scenarios

**Scenario 1: Model server fails, no fallback**
- Neural net server crashes. No fallback. Users see errors. Complete outage 2+ hours.
- Prevention: Fallback rule-based model (RTO=0). Test weekly.

**Scenario 2: Disaster recovery plan untested**
- DR doc says "failover to region B". During actual disaster, region B setup missing (deferred).
- RTO = infinite (never recovery).
- Prevention: Monthly mock disaster drills. Measure actual RTO (not planned). Update plan based on reality.

**Scenario 3: Data loss due to backup too old**
- Backup 30 days old. Data between backup and failure lost (recent model training data, predictions).
- Prevention: Daily backups. Hourly snapshots for critical data. RPO <1 hour target.

**Scenario 4: Multi-region failover fails**
- Primary region fails. Failover to region B. DNS not updated. Region B overloaded. Cascading failure.
- Prevention: Health checks detect primary failure <10s. LB switches instantly (not DNS). Region B pre-warmed.

---

## Implementation Guidance

**Wrong:** Have DR plan but never test it. Hope it works.
**Right:** Monthly mock disaster. Measure actual RTO/RPO. Update plan. Automate failover (don't manual).

**Wrong:** Fallback too simplistic (always deny loans, always reject requests).
**Right:** Fallback rule-based, but intelligent. Maintain service quality even degraded.

---

## Sophisticated Interview Q&A

**Q1: RTO goal 5 minutes, but primary failure recovery takes 10 min. How achieve?**
A: (1) Fallback rule-based <10 sec (RTO=0 for users, just degraded service). (2) Restore primary in parallel (10 min). (3) Once restored, switch back from fallback. (4) Users never see outage (just degradation). (5) This is acceptable for most use cases.

**Q2: Multi-region setup. Region A fails. Users routed to Region B. But Region B capacity 50%, suddenly 100%. Overload?**
A: Yes, risk. Solutions: (1) Pre-warm region B to handle 100% (doubles cost). (2) Auto-scale region B on failure (takes time, risky). (3) Degrade service gracefully (rate-limit, queue). (4) Accept short outage for overflow. Most companies choose: accept 1-5 min degradation while scaling.

**Q3: RPO 0 (no data loss). How?**
A: (1) Multi-region active-active (write to both, <1s RPO). (2) Synchronous replication (slow, ~100ms latency). (3) Cost: 2x infra + complexity. (4) Or: accept RPO=1min (reasonable for most). Only industries like finance need RPO=0.

**Q4: Monthly DR drill. Always passes. But actual disaster failed. Why?**
A: (1) Drill assumptions unrealistic (didn't test peak traffic). (2) Manual steps not practiced (forgotten). (3) Partial failure (one component OK, others not). (4) Solution: (a) Chaos engineering (kill random components). (b) Test all failure modes. (c) Automate (no manual steps). (d) Test during peak traffic.

---

## Cost & Resource Analysis

**Fallback model:** Development 1-2 weeks = $5-10K. Maintenance <$1K/year.
**Multi-AZ:** 1.5x infra cost ($1500/month for $1000 baseline). + 1-2 engineer weeks setup.
**Multi-region:** 2-3x infra cost. + 4-8 engineer weeks for replication, failover. Ongoing: 1 FTE ops.
**Active-active multi-region:** 3-4x infra. + 12-16 engineer weeks. Very complex.

**Cost of downtime:** $10K/min for critical service. 1 hour downtime = $600K loss.
**ROI:** DR investment $50-500K prevents $600K+ incident. Break-even: 1 incident per year.

---

## Monitoring & Observability

**Key metrics:** RTO (actual recovery time tested monthly), RPO (actual data loss measured), backup freshness (age of latest backup), failover success rate, primary/secondary health status, replication lag (multi-region), DNS propagation time

**Alerts:** Primary failure detected, fallback activated, backup older than SLA, replication lag exceeds threshold, failover fails, region unreachable, data corruption detected

## Common Mistakes / Gotchas
- No fallback: model down → complete outage
- Untested DR: "DR plan exists" but never tested → fails during disaster
- Single point of failure: one database, one region
- Data loss: backups too infrequent

## Best Practices
- **Fallback always:** rule-based model when neural net fails
- **Multi-region:** servers in 2+ regions, data replicated
- **Regular backup:** daily snapshots, 30-day retention
- **DR testing:** monthly mock disaster, measure RTO/RPO
- **Documentation:** clear runbook for each failure scenario

## Code Example
```python
class DisasterRecovery:
    def predict_with_fallback(self, features):
        try:
            # Primary model
            prediction = primary_model.predict(features)
            return prediction
        except Exception as e:
            logging.error(f"Primary model failed: {e}")
            
            # Fallback rule-based
            if features['income'] > $50K:
                return "approve"
            else:
                return "deny"
```

## Interview Q&A
**Q: Model server fails during peak traffic. 50K users affected. RTO goal: 5 min. How?**
A: (1) Fallback rule-based model (RTO=0). (2) Health checks detect failure <10s. (3) Route traffic to fallback <30s. (4) Restore server <5min. Combined: users see degraded service, not outage.

**Q: RPO: how much data loss acceptable?**
A: Business decision. For user preferences: 1 hour loss acceptable (small impact). For financial transactions: 0 loss (regulatory requirement). Design accordingly.

## Interview Quick-Reference
| Scenario | Solution | RTO |
|----------|----------|-----|
| Model server down | Fallback model | 0 |
| Region down | Multi-region failover | 1s |
| Data corruption | Restore from backup | 1h |

## Related Topics
- [Production Readiness](23-production-readiness.md)
- [Monitoring](16-monitoring-and-observability.md)

## Resources
- [Disaster Recovery Planning](https://en.wikipedia.org/wiki/Disaster_recovery)
