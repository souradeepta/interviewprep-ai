# Disaster recovery

## TL;DR
Core ML system design pattern for production.

## Core Intuition
[Intuitive explanation]

## How It Works
[Technical details]

## Key Properties / Trade-offs
- Property 1
- Property 2

## Common Mistakes / Gotchas
- Mistake 1
- Mistake 2

## Best Practices
- Define RPO (Recovery Point Objective) and RTO (Recovery Time Objective) before designing recovery
- Test recovery procedures quarterly — untested backups often fail
- Store backups in a different geographic region from primary
- Automate failover — manual failover during incidents is slow and error-prone
- Practice chaos engineering to validate recovery before disasters happen
- Document step-by-step recovery runbooks with owner assignments
- Keep shadow warm standby systems for critical ML inference endpoints

## Interview Q&A

**Q: What is the RTO and RPO for ML systems and how do they differ from standard software?**
A: RTO (Recovery Time Objective): how long the system can be unavailable. For real-time ML serving: minutes. For batch inference: hours. RPO (Recovery Point Objective): how much data/state can be lost. For ML systems: typically measured in terms of model version (can we recover to the last good model) and training data (can we recover the last training dataset). ML-specific RPO consideration: if the model was retrained 3 times since the last backup, recovering the last good model version is more important than recovering the last training run.

**Q: How do you design an ML system to automatically failover when the primary model fails?**
A: Implement a fallback hierarchy: (1) primary model (latest deployed); (2) previous model version (always keep the N-1 version warm); (3) rule-based fallback (simple heuristic that handles the most common cases); (4) default response (safe fallback: "unable to process at this time"). Implement health checks that trigger automatic failover: if the primary model returns error rate >5% for 60 seconds, automatically switch traffic to the previous model version. Test failover regularly in staging—circuit breakers that aren't tested often fail silently.

**Q: What backup strategy is appropriate for ML model artifacts?**
A: Treat model artifacts like critical infrastructure: daily snapshots, retained for 30 days. Keep: model weights (or compressed serialized format), model architecture definition, preprocessing pipeline, evaluation results, and model card documentation. Cross-region replication for disaster recovery (primary failure should not cause data loss). Test recovery: monthly drill where you restore a model from backup and verify it produces correct predictions. The backup is worthless if you can't restore from it in a timely manner.

**Q: How do you handle ML system recovery when the underlying data pipeline is corrupted?**
A: A corrupted data pipeline can silently produce wrong features, causing model predictions to degrade without obvious errors. Recovery steps: (1) identify the last known good data state (use data versioning); (2) halt all model retraining that uses the corrupted pipeline; (3) assess impact: which models were trained on corrupted data? (4) roll back affected models to the last version trained on good data; (5) fix the pipeline; (6) validate fixed pipeline output matches pre-corruption baselines; (7) retrain affected models. This recovery can take days if data versioning is inadequate.

**Q: What is the difference between a model rollback and a model recovery and when do you use each?**
A: Rollback: intentional reversion to a previous known-good model version—used when a newly deployed model has degraded performance. Fast (minutes), no data loss, preserves operational continuity. Recovery: restoring a model after catastrophic failure (data corruption, lost model artifacts, infrastructure failure)—used when the current model can't serve at all. May take hours, requires backup restoration. Design for rollback first (it's more common) by keeping N-1 model versions always warm and ready to serve. Recovery is the fallback when rollback isn't possible.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
