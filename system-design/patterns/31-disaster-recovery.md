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
