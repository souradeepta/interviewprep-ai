# ML Governance

## TL;DR
Govern ML systems: model ownership, model registry, approval workflow (staging → prod), audit trail (who changed what), rollback procedure. Required for: compliance, safety, accountability.

## Core Intuition
ML system = code + data + model. Governance = versioning all three, audit trail, approval workflow.

## How It Works

**ML governance components:**

1. **Model ownership:** who is responsible for this model?
2. **Approval workflow:** 
   - Scientist trains model
   - Reviewers approve
   - Deploy to staging
   - Stakeholder signs off
   - Deploy to prod
3. **Audit trail:** log all changes (who, when, why)
4. **Rollback:** ability to revert to previous model

| Phase | Owner | Action |
|-------|-------|--------|
| Training | Data scientist | Train model |
| Review | Tech lead | Code review |
| Staging | QA | Test model |
| Approval | Business owner | Approve for prod |
| Prod | ML engineer | Deploy & monitor |

## Key Properties / Trade-offs
- Process overhead: governance slows deployment
- Safety: prevents bad models reaching prod
- Accountability: clear who is responsible

## Detailed Trade-off Analysis

| Governance Level | Approval Steps | Deployment Time | Safety Risk | Compliance |
|-----------------|----------------|-----------------|-------------|-----------|
| None (ad-hoc) | 0 | <1 hour | 50% (high) | None |
| Basic (1 approval) | 1 (tech lead) | 4-8 hours | 20% | Basic |
| Standard (2 approvals) | 2 (tech + business) | 1 day | 5% | Good |
| Strict (3+ approvals) | 3+ (tech + business + legal) | 2-3 days | 1% | Excellent |

**Decision:** Startup MVP → basic (1 approver). Series A → standard (2). Regulated (healthcare, finance) → strict (3+).

---

## Production Failure Scenarios

**Scenario 1: Bad model deployed without approval**
- Scientist trains model, deploys directly to prod (skipped staging/approval). Model breaks in production.
- Takes 4 hours to rollback (no rollback procedure).
- Prevention: Enforce approval workflow. No direct prod access without approval.

**Scenario 2: Approval process so slow, engineers bypass it**
- Approval takes 3 days. Engineers frustrated, start deploying via backdoor scripts.
- Compliance now broken (no audit trail).
- Prevention: Streamline approval (target: <4 hours). Automate checks (pre-approvals if tests pass).

**Scenario 3: Model in prod, audit log doesn't show who deployed**
- Issue occurs. Can't trace who deployed and when. Slow investigation.
- Prevention: Mandatory audit log (who, what, when, why). Can't skip logging.

**Scenario 4: Rollback procedure doesn't work**
- Bad model in prod. Try to rollback. Procedure outdated, fails. 2+ hour outage.
- Prevention: Rollback tested weekly. Documented procedure. <5 minute recovery time target.

---

## Implementation Guidance

**Wrong:** Approval process so strict (5 stakeholders) that deployment takes 2 weeks. Engineers bypass it.
**Right:** Balance safety and speed. 2 approvals (tech + business), target <4 hours. Automate checks (tests, fairness, monitoring).

**Wrong:** Audit log is CSV email (manual, incomplete).
**Right:** Audit log is immutable database. All model changes logged automatically. Cannot be deleted.

---

## Sophisticated Interview Q&A

**Q1: Model deployment blocked by approval. Business pressure to ship ASAP. What do you do?**
A: (1) Escalate: is business pressure due to approval slowness or legitimate urgency? (2) If slowness: streamline approval (parallel approvals, auto-pass if tests OK). (3) If urgent: use canary (deploy to 1% users) without full approval, monitor 1 hour, then full rollout. (4) Never skip approval, but can make it faster.

**Q2: Approval from 5 stakeholders (tech, product, legal, finance, business). Realistic?**
A: Depends on stakes. (1) High-stakes (healthcare): yes, worth 2-3 day delay. (2) Low-stakes (recommendations): no, too slow. (3) Compromise: (a) tech + business fast-track (4 hours), (b) legal/finance review in parallel (not sequential), (c) pre-approvals for minor changes.

**Q3: Rollback procedure: how quickly can you revert to previous model?**
A: Target: <5 minutes (config change to point to old model). Implementation: (1) Blue-green deployment (instant switch). (2) Or: load balancer switch (seconds). (3) Test weekly: can you actually do it? (4) Have runbook. (5) Automated rollback if error rate >1%.

**Q4: Audit trail: what should it capture?**
A: (1) WHO: email of deployer. (2) WHAT: model version, data version, code commit. (3) WHEN: timestamp. (4) WHERE: staging or prod. (5) WHY: reason for deployment (bug fix, improvement, etc.). (6) APPROVAL: who approved. (7) RESULT: success or failure. (8) Cannot be edited/deleted (immutable log).

---

## Cost & Resource Analysis

**Governance infrastructure:** MLflow Model Registry, approval workflow tool: $500-2K/month.
**Approval overhead:** 2 approvers × 30 min per deployment = 1 hour per deployment. For 10 deployments/month = 10 hours = $1K/month.
**Compliance team (regulated domains):** 1 person audit/monitor = $100K+/year.

**Cost of bad deployments (without governance):** $100K-1M per incident (downtime, reputation, fix time).
**ROI:** Governance $2-100K/year. Prevents 1 incident/year. Break-even easily justified.

---

## Monitoring & Observability

**Key metrics:** Deployment approval time (SLA: <4 hours), audit log completeness (100% logged), rollback time (SLA: <5 min), governance compliance (% approved before prod), incident root cause (% traced back to deployment), model lifecycle tracking (training → staging → prod)

**Alerts:** Approval SLA breached, unapproved deployment detected, audit log gap (missing entries), rollback failed, model deployed without staging test

## Common Mistakes / Gotchas
- No approval process: anyone can deploy (chaos)
- No audit trail: can't debug who changed what
- No rollback: bad model → stuck
- Too much process: bottleneck, slow deployment

## Best Practices
- **Clear workflow:** staging → prod (don't skip staging)
- **Signature required:** business owner signs off before prod
- **Audit logging:** every model change logged
- **Rollback SOP:** procedure for emergency rollback
- **Regular audits:** monthly review of all models in prod

## Code Example
```python
class MLGovernance:
    def request_deployment(self, model_name, version):
        # Create approval request
        request = {
            "model": model_name,
            "version": version,
            "requester": "alice@company.com",
            "timestamp": datetime.now(),
            "status": "pending_review"
        }
        self.db.save_request(request)
        
        # Notify reviewers
        self.send_approval_email(request)
    
    def approve_deployment(self, request_id, approver):
        request = self.db.get_request(request_id)
        request["approver"] = approver
        request["status"] = "approved"
        self.db.save_request(request)
```

## Interview Q&A
**Q: Bad model deployed, caused issue. How prevent?**
A: ML governance. (1) Staging environment (test before prod). (2) Approval workflow (manager signs off). (3) Audit trail (know who deployed what). (4) Monitoring (catch issue in staging, not prod). (5) Rollback (revert quickly).

**Q: How many levels of approval?**
A: Depends on stakes. Low-stakes (recommendations): tech lead. High-stakes (healthcare): tech lead + business owner + legal. Balance: safety vs speed.

## Interview Quick-Reference
| Component | Purpose |
|-----------|---------|
| Ownership | Accountability |
| Approval | Gate-keeping |
| Audit trail | Compliance |
| Rollback | Safety |

## Related Topics
- [Model Registry](04-model-registry.md)
- [Production Readiness](23-production-readiness.md)

## Resources
- [ML Governance Framework](https://www.mckinsey.com/)
