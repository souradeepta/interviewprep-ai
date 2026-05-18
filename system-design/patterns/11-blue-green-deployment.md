# Blue-Green Deployment

## TL;DR
Two identical environments (blue, green). Traffic on blue. Deploy to green, test, switch traffic to green. Instant rollback: switch back to blue.

## Core Intuition
Two parallel systems. Keep old one ready, switch to new, easy rollback.

## How It Works
```
Blue (current): production, getting all traffic
Green (new): deploy new model, run tests, no real traffic

When ready:
  Switch load balancer: blue → green
  Now green is live, blue is backup

If issues:
  Switch back: green → blue (instant)
```

## Key Properties / Trade-offs
- Rollback: instant (switch LB)
- Cost: run 2 environments (2x cost)
- Downtime: minimal (switch is <1s)
- Testing: can test on green before switching

## Common Mistakes / Gotchas
- **Double cost:** full redundancy is expensive
- **Data sync:** if data changes during switch, blue-green diverge. Minimize DB changes during switch.
- **Not testing green:** deploy without testing → issues after switch

## Interview Q&A

**Q: What are the infrastructure requirements for blue-green deployment of ML models?**
A: Requires: capacity to run two full production environments simultaneously (2x cost during deployment), a load balancer that can switch traffic instantaneously, automated health checking for the green environment before traffic switch, and a rollback mechanism that can switch back in <5 minutes. For ML specifically: both environments need access to the same feature store and inference infrastructure, model artifacts must be pre-loaded in the green environment before traffic switch, and both environments must produce identical predictions for the same inputs (to validate correctness before switch).

**Q: How long should you maintain the blue environment after switching traffic to green?**
A: Maintain blue for: the time needed to detect slow-burn failures (P99 latency degradation, accuracy drift) that don't appear immediately. Minimum: 24 hours after the switch, longer for models where business impact takes time to manifest (recommendation models: 48-72 hours to see session depth changes). During this window, keep blue warm (don't scale down) so rollback is instantaneous, not another deployment. Delete blue after: deployment is considered stable AND rollback window has passed.

**Q: How do you validate the green environment before switching traffic?**
A: Automated checks: run the full model evaluation suite on the green endpoint, replay a sample of recent production requests and compare predictions to blue (should match for the same inputs unless the model deliberately changed), run performance benchmarks (latency/throughput within 10% of blue), verify all dependencies are reachable. Manual validation: have the team test the green endpoint on realistic inputs. Gate the traffic switch on all automated checks passing—never switch manually without automated validation.

**Q: When is blue-green deployment inappropriate for ML models?**
A: Inappropriate when: 2x infrastructure cost is prohibitive, the new model requires different infrastructure (different GPU type, different serving framework) making side-by-side impossible, or the model change is intentionally backward-incompatible (different input schema). Alternatives: canary deployment (route small % of traffic to new model, not 0/100% switch), shadow mode (run new model but ignore its predictions), or rolling deployment (replace replicas one by one with brief overlap period). Choose based on your risk tolerance and infrastructure constraints.

**Q: How do you handle database schema changes in conjunction with blue-green ML deployments?**
A: Database changes that affect feature computation or model metadata must be backward-compatible with both blue and green. Expand-contract pattern: (1) expand—add new column/table without removing old, both blue and green work; (2) cut over—switch traffic to green; (3) contract—remove old column/table after blue is decommissioned. Never do a one-step database migration that breaks the currently-live blue environment. Feature store schema changes require the same careful coordination.

## Interview Quick-Reference
**Blue-green?** Two envs, instant switch, instant rollback. More expensive, safer.

## Related Topics
- [Canary Deployment](12-canary-deployment.md) — gradual alternative
- [Production Readiness](23-production-readiness.md)

## Resources
- [Blue-Green Deployment](https://martinfowler.com/bliki/BlueGreenDeployment.html)
