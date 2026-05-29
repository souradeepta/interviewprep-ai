# Disaster Recovery

## TL;DR
Plan for failures: model server down (fallback model), data pipeline broken (use cached data), entire region down (multi-region setup). RTO (recovery time), RPO (data loss tolerance).

## Core Intuition
Disaster = model server down. Users see errors. RTO: how long before users see results again? RPO: how much data lost? Plan accordingly.

## How It Works

**Disaster recovery levels:**

1. **Service degradation (RTO=minutes):**
   - Model server down -> use fallback rule-based model
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

**Decision:** Startup -> fallback + backup. Series A -> multi-AZ. Unicorn/critical -> multi-region active-active.

---

## Production Failure Scenarios

**1. Untested Restore Procedure**
- **Symptom:** During a live incident, database restore takes 6 hours instead of the 1 hour documented in the runbook; recovery SLA is missed.
- **Root Cause:** Restore runbook written at system launch 18 months ago; never tested under production conditions; data volume has since grown 5x and the restore time estimate was never updated.
- **Detection:** Quarterly restore drill on a non-production copy of the database; measure actual restore time vs documented RTO; alert if actual > 1.5x documented.
- **Fix:** Monthly restore drill for all critical systems; gate production certification on a successful restore test; update the runbook after every drill with measured timing.

**2. RPO Violation Due to Replication Lag**
- **Symptom:** Database failover is triggered and the replica is activated; post-incident analysis shows 4 hours of transaction data was lost despite an RPO target of 1 hour.
- **Root Cause:** Asynchronous replication was used; replica lag had grown to 4 hours due to a network partition that went undetected; no alerting on replication lag.
- **Detection:** Continuously monitor replication lag as an SLI; alert at lag > 0.5 x RPO target (i.e., alert at 30 minutes for a 1-hour RPO); page on-call if lag exceeds RPO target.
- **Fix:** For critical databases, switch to synchronous replication (accept 5-20ms latency overhead); for less critical data, add lag monitoring with automated failover suppression when lag exceeds RPO (surface the tradeoff explicitly rather than silently failing the RPO).

**3. Cold Standby Model Stale**
- **Symptom:** Failover activates the cold standby system, which serves predictions from a 6-month-old model version; users receive significantly degraded quality before the issue is discovered.
- **Root Cause:** Cold standby infrastructure is deployed but not updated when the primary model is updated; there is no automated sync between primary and standby model artifacts.
- **Detection:** Version comparison between primary and standby model artifact on every primary deployment; alert if standby version is more than 2 deployments behind the primary.
- **Fix:** Automate standby model sync as a step in the primary deployment pipeline; gate primary deployment completion on successful standby sync; include model version as a health check metric exposed by the standby serving endpoint.

**4. DR Runbook Incomplete**
- **Symptom:** On-call engineer successfully restores the database and application layer during a recovery, but model serving remains down because the ML serving stack was not covered in the DR runbook.
- **Root Cause:** DR runbook written by the infrastructure team covers database and application servers but not the ML-specific components (model server, feature store, embedding cache).
- **Detection:** Annual tabletop exercise with ML, infrastructure, and product teams; walk through the runbook step by step and identify gaps.
- **Fix:** DR runbook must enumerate every system component including ML serving stack, feature store, model registry, and inference cache; assign an ML engineer as a required participant in all DR drills; runbook is incomplete if any live service is not covered.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Multi-region standby (warm) | 100% of primary cost | $10K primary | $10,000 |
| Multi-region standby (cold) | 10% of primary cost | $10K primary | $1,000 |
| Backup storage (S3 cross-region) | $0.023/GB | 10TB | $230 |
| DR drill engineering time | $200/hr | 8 hr/quarter | $533 amortized |
| **Total (warm standby)** | | | **~$10,763/month** |
| **Total (cold standby)** | | | **~$1,763/month** |

The dominant DR cost is standby infrastructure, and the choice between warm and cold standby drives a 6x cost difference ($10,763 vs $1,763/month in this example). Warm standby justifies its cost when the RTO requirement is under 5 minutes and downtime has a measurable revenue impact (e.g., $10K/minute for a payments system means a 5-minute outage costs $50K -- more than the monthly standby cost). Cold standby is appropriate when the RTO requirement is measured in hours and the system's value does not warrant the warm standby premium. DR drill time ($533/month amortized) is a fixed cost regardless of standby strategy and is non-negotiable: an untested DR plan is not a plan.

---

## Implementation Guidance

**Wrong:** Have DR plan but never test it. Hope it works.
**Right:** Monthly mock disaster. Measure actual RTO/RPO. Update plan. Automate failover (don't manual).

**Wrong:** Fallback too simplistic (always deny loans, always reject requests).
**Right:** Fallback rule-based, but intelligent. Maintain service quality even degraded.

---

## Interview Q&A

**Q: Your RTO goal is 5 minutes, but the primary failure recovery takes 10 minutes. How do you achieve the SLA?**
A: Use a two-layer recovery strategy: (1) activate the fallback rule-based model immediately on primary failure detection (RTO = 0 for users; they see degraded quality but not an error); (2) restore the primary model server in parallel (10 minutes). Once restored, drain the fallback and switch back. Users experience degraded service for 10 minutes but zero outage. This is how most production ML systems handle model failures -- a fast, imperfect fallback plus a slower full recovery running in parallel. The key is that "degraded service" must be meaningful: a fallback that always returns the same response is not useful.

**Q: Multi-region failover is triggered and Region B is suddenly receiving 100% of traffic at double its normal load. How do you handle the capacity surge?**
A: Three options: (1) pre-warm Region B to handle 100% load at all times (simplest, doubles infra cost); (2) auto-scale Region B on failover trigger -- this requires Region B's autoscaling to respond within 2-3 minutes, which introduces a window of degraded service; (3) graceful degradation during the scale-up window -- implement request queuing, client-side retry with backoff, and rate limiting to protect Region B while it scales. Most production systems choose option 3 combined with partial pre-warming (Region B runs at 60% capacity baseline, reducing the scale-up gap). Accept 1-5 minutes of degraded service while scaling; document this in the SLA as an acceptable degraded-mode window.

**Q: How do you achieve RPO = 0 (zero data loss) for a critical financial system?**
A: Zero RPO requires synchronous replication: every write to the primary is confirmed only after it has been committed to at least one replica. This means: (1) write latency increases by the round-trip time to the replica (50-100ms for cross-region synchronous replication); (2) primary failure cannot progress faster than the replication commit path. In practice, true RPO = 0 is implemented with multi-region active-active databases (CockroachDB, Google Spanner, AWS Aurora Global) that use consensus protocols (Raft, Paxos) to guarantee all replicas agree before acknowledging a write. Cost: 2-3x single-region infrastructure plus 50-100ms additional write latency. Only finance, healthcare critical systems, and similar regulated domains justify this cost; most ML systems can accept RPO = 1 minute.

**Q: Your quarterly DR drill always passes in staging but the last actual disaster took 3x longer than documented. Why, and how do you fix it?**
A: Four common causes of drill-to-reality gaps: (1) drill uses a smaller data volume than production -- restore a 100GB staging database in 20 minutes, but production is 2TB (40-80 minutes); (2) drill doesn't test peak traffic -- failover under idle conditions succeeds, but under 10x peak load the standby is overwhelmed; (3) manual steps atrophy -- the runbook says "restart service X" but the person who wrote it left; the current oncall has never done it; (4) drift -- infrastructure changed since the runbook was written. Fixes: test with production-scale data; run drills during business hours under real traffic; automate every manual step; require runbook review after each infrastructure change.

**Q: How do you design a disaster recovery strategy for an ML system specifically, beyond generic infrastructure DR?**
A: ML systems have DR requirements that generic infrastructure DR misses: (1) model version recovery -- the DR plan must specify which model version activates on failover and how to validate its behavior before serving traffic; (2) feature store recovery -- if real-time features are unavailable, the fallback model must work with batch features only (or default features); (3) training pipeline recovery -- if the training pipeline is interrupted, the DR plan must include: resume from the last checkpoint, retrain from scratch, or serve the last good model indefinitely; (4) data pipeline recovery -- stale feature data has a time-to-corruption for model quality; the DR plan must specify the maximum tolerable feature staleness. Document each ML-specific component in the runbook with its RTO and the degraded-mode behavior when it is unavailable.

**Q: What early warning signs should trigger DR preparation before a disaster actually occurs?**
A: Six signals worth monitoring as leading indicators: (1) replication lag trending upward (approaching RPO threshold); (2) disk utilization on primary database above 80% (increases restore time and risk of write failures); (3) error rate on primary ML serving above 0.1% (may indicate instability before full failure); (4) memory or CPU utilization on model servers above 85% (headroom insufficient to absorb traffic spike during partial failover); (5) health check latency on secondary region increasing (secondary may be degraded before it is needed for failover); (6) backup completion time exceeding its SLA (backup job itself is slow, indicating backup infrastructure issues). Alert on all six; escalate to DR preparation when two or more are triggered simultaneously.

**Q: How do you test that your automated failover actually works before a real incident?**
A: Chaos engineering is the gold standard. Netflix's Chaos Monkey (randomly kills production instances) is the extreme version. A practical approach for ML systems: (1) monthly planned failover drill -- route 100% of traffic to Region B in a maintenance window, verify latency and accuracy, then switch back; (2) quarterly random-time failover drill -- kill the primary model server without advance notice during business hours; measure time-to-fallback-activation and time-to-recovery; (3) continuous canary -- keep a small amount of traffic (1-5%) routed through the secondary system at all times to verify it stays warm and functional. The continuous canary is the most valuable: it catches standby configuration drift before you need the standby in an emergency.

**Q: Your ML system's DR plan says RTO = 5 minutes, but you have never measured the actual failover time. How do you validate the claim?**
A: Measure it. Steps: (1) schedule a maintenance window during off-peak hours; (2) simulate the failure by stopping the primary model server (not just health-check blocking -- actually stop the service); (3) start a timer from the moment of failure to the moment the first successful user request is served from the failover path; (4) repeat the measurement 3 times across different days and traffic patterns to get a distribution, not a single data point; (5) compare P95 recovery time (not just average) against the 5-minute RTO; P95 is what determines whether you meet the SLA 95% of the time. Document the measurement methodology and results in the DR runbook. If P95 exceeds the SLA, the SLA is aspirational, not guaranteed -- update either the RTO claim or the infrastructure.

---

## Monitoring & Observability

**Key metrics:** RTO (actual recovery time tested monthly), RPO (actual data loss measured), backup freshness (age of latest backup), failover success rate, primary/secondary health status, replication lag (multi-region), DNS propagation time

**Alerts:** Primary failure detected, fallback activated, backup older than SLA, replication lag exceeds threshold, failover fails, region unreachable, data corruption detected

## Common Mistakes / Gotchas
- No fallback: model down -> complete outage
- Untested DR: "DR plan exists" but never tested -> fails during disaster
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

            # Fallback rule-based model (always available, no external dependencies)
            if features['income'] > 50000:
                return "approve"
            else:
                return "deny"
```

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
