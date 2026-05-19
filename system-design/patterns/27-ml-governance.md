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
