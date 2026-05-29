# ML Governance

## TL;DR
Govern ML systems: model ownership, model registry, approval workflow (staging -> prod), audit trail (who changed what), rollback procedure. Required for: compliance, safety, accountability.

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

**Decision:** Startup MVP -> basic (1 approver). Series A -> standard (2). Regulated (healthcare, finance) -> strict (3+).

---

## Production Failure Scenarios

**1. Model Deployed Without Approval**
- **Symptom:** Compliance audit finds model in production without model card or review board sign-off; deployment has no entry in the approval log.
- **Root Cause:** Deployment pipeline did not enforce a governance gate; scientist had direct prod write access.
- **Detection:** Weekly deployment audit -- flag any model in the registry with status != "approved" that is currently serving traffic.
- **Fix:** CI/CD pipeline gate: deployment blocked unless model card is present and status = "approved" in the registry; revoke direct prod access for all non-ops roles.

**2. Model Card Outdated Post-Retraining**
- **Symptom:** Model card describes v1 behavior; v3 is deployed in production -- compliance gap discovered during external audit.
- **Root Cause:** Model card is a static document not versioned alongside the model artifact in the registry.
- **Detection:** Link model card version to model registry version; alert on version mismatch between serving model and its associated card.
- **Fix:** Regenerate model card template automatically on every training run; require human review and sign-off for material changes (new training data, changed objective, updated thresholds).

**3. Shadow Mode Skipped**
- **Symptom:** New model causes a 5% regression in a downstream revenue metric discovered in production 48 hours after deployment.
- **Root Cause:** Rushed deployment timeline; shadow testing period bypassed under business pressure.
- **Detection:** Retrospective checklist -- was shadow mode run for at least 1 week for all models touching revenue metrics?
- **Fix:** Mandatory 1-week shadow period for all models touching revenue or safety metrics; automate shadow traffic routing in the serving layer so it requires an explicit override to skip.

**4. Governance Theater**
- **Symptom:** Model cards filled with boilerplate ("model is fair and accurate") without quantitative substance; audit finds no actual bias measurements.
- **Root Cause:** Checklist-driven compliance without genuine understanding of what each field requires.
- **Detection:** Governance team reads a random 10% of cards quarterly; flag cards with <50 words in the "limitations" section or missing numeric bias metrics.
- **Fix:** Require quantitative bias metrics (demographic parity, equalized odds), specific failure modes with reproduction steps, and version history in every model card; template rejects submission if required numeric fields are empty.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| MLflow model registry | $200/mo | 1 hosted instance | $200 |
| Model review board time | $200/hr | 4 hr/review x 5 reviews | $4,000 |
| Shadow infrastructure | $2/hr | 168 hr/week (1 shadow) | $336 |
| Governance tooling (custom) | $300/hr | 80 hr build (amortized) | $250 |
| **Total** | | | **~$4,786/month** |

ML governance infrastructure is relatively cheap -- the dominant cost is human review time, not tooling. At $4,786/month for five model deployments, the per-deployment governance cost is roughly $957. This is almost always justified: the cost of a single bad model deployment (customer impact, rollback engineering time, compliance investigation) typically runs $50K-500K. For regulated domains (healthcare, finance), add a compliance engineer at ~$200/hr for 10-20 hours/month, bringing the total to ~$6-8K/month -- still well below the cost of one incident. The ROI calculation favors investing in automated gates (shadow routing, automated model cards) over manual review as deployment frequency increases.

---

## Implementation Guidance

**Wrong:** Approval process so strict (5 stakeholders) that deployment takes 2 weeks. Engineers bypass it.
**Right:** Balance safety and speed. 2 approvals (tech + business), target <4 hours. Automate checks (tests, fairness, monitoring).

**Wrong:** Audit log is CSV email (manual, incomplete).
**Right:** Audit log is immutable database. All model changes logged automatically. Cannot be deleted.

---

## Interview Q&A

**Q: A model deployment is blocked by the approval workflow, and the business is pressuring you to ship immediately. What do you do?**
A: First, distinguish between approval slowness and genuine urgency. If the process is slow, streamline it: run approvals in parallel rather than sequentially, and auto-approve routine retraining (same architecture, same data, metric improved) with a shorter review window. If it is genuinely urgent, use a canary deployment to 1% of traffic without full approval, monitor error rate and business metrics for 1 hour, then proceed to full rollout only if clean. Never skip the audit trail -- even expedited deployments must be logged with the justification.

**Q: How would you design a model card that is actually useful, not just a compliance checkbox?**
A: A useful model card answers four questions a reviewer can check: (1) What does the model do and what does it not do? (stated limitations section, at least 100 words); (2) Where does it fail? (specific failure modes with reproduction steps); (3) Is it fair? (quantitative bias metrics across demographic groups, not just "tested for fairness"); (4) What changed from the last version? (version history with a summary of each change). Gate submission on all four sections being populated with non-boilerplate content. The governance team should sample 10% of cards quarterly and score them 1-5 on substantiveness.

**Q: A bad model was deployed to production last week. How do you investigate who is responsible and prevent recurrence?**
A: Use the audit trail: (1) query the immutable deployment log for the model version, deployer identity, approver identity, and timestamp; (2) check whether all required gates were passed (staging test results, model card status, shadow mode duration); (3) if gates were bypassed, trace whether this was an authorized emergency override or an unauthorized action. For prevention: if gates were bypassed legitimately, review whether the emergency override process has appropriate safeguards; if bypass was unauthorized, revoke direct prod access and add an alert for any deployment without a corresponding approval record.

**Q: How quickly should you be able to roll back a model in production?**
A: Target is under 5 minutes for traffic rerouting -- this is achievable with a load balancer switch or a config change pointing to the previous model artifact, without redeployment. The rollback procedure must be tested monthly; if the procedure has never been exercised in production, you do not know your actual RTO. For regulated domains, document the rollback procedure in the DR runbook and include it in quarterly tabletop exercises. Automated rollback (triggered by error rate > 1% for 5 minutes) removes human latency from the critical path.

**Q: What should an ML model audit trail capture to be compliance-grade?**
A: Seven required fields: (1) WHO -- email of deployer and each approver; (2) WHAT -- model version, training data version hash, code commit hash; (3) WHEN -- timestamp of each state transition (submitted, approved, deployed, rolled back); (4) WHERE -- staging or production, specific serving endpoints; (5) WHY -- reason for deployment (performance improvement, bug fix, regulatory requirement); (6) RESULT -- deployment outcome (success, failure, rollback); (7) APPROVAL chain -- each approver's identity and timestamp. The log must be immutable (append-only, no deletes) and retained for the regulatory retention period (often 7 years in finance).

**Q: Your organization ships 20 models per month. The current 2-day approval process is creating a backlog. How do you scale governance without sacrificing safety?**
A: Tier the approval process by risk, not by model. Low-risk changes (same model architecture, >1% metric improvement, same training data distribution) can use an expedited 4-hour automated review: metrics gate + bias check + shadow mode for 24 hours. Medium-risk changes (new features, different training data) require tech lead review in 1 business day. High-risk changes (new model family, safety-critical domains, major architecture changes) keep the full 2-day review. This typically lets 60-70% of deployments use the expedited track, cutting the average approval time to under 1 day while preserving rigor for high-risk changes.

**Q: A model is serving predictions but its model card shows it was trained on data from 18 months ago. Is this a problem?**
A: It depends on the model's use case and data distribution stability. Ask: (1) Has the input data distribution shifted? (statistical tests: KL divergence, population stability index); (2) Have the business rules or labels changed? (e.g., fraud taxonomy updated, customer segmentation redefined); (3) Is model performance declining? (compare current live metric vs baseline from 18 months ago). If any are yes, trigger retraining. If all are no, document the drift analysis result in the model card and set a maximum staleness policy (e.g., retrain every 6 months regardless). Staleness itself is not the problem -- undocumented staleness without drift monitoring is.

**Q: How do you handle governance for a model that makes decisions with significant human impact, such as credit scoring or medical diagnosis?**
A: High-impact models require three additional governance layers beyond standard: (1) Explainability requirement -- every prediction must have a human-interpretable explanation (SHAP, LIME, or rule-based fallback); (2) Human-in-the-loop policy -- decisions above a certain impact threshold require human review before execution, not just logging; (3) Adverse action documentation -- if a model denies credit or flags a medical condition, the rationale must be documentable in terms a non-technical person can understand, per ECOA and similar regulations. These requirements should be in the model card and enforced at the serving layer, not left to the consuming application.

---

## Monitoring & Observability

**Key metrics:** Deployment approval time (SLA: <4 hours), audit log completeness (100% logged), rollback time (SLA: <5 min), governance compliance (% approved before prod), incident root cause (% traced back to deployment), model lifecycle tracking (training -> staging -> prod)

**Alerts:** Approval SLA breached, unapproved deployment detected, audit log gap (missing entries), rollback failed, model deployed without staging test

## Common Mistakes / Gotchas
- No approval process: anyone can deploy (chaos)
- No audit trail: can't debug who changed what
- No rollback: bad model -> stuck
- Too much process: bottleneck, slow deployment

## Best Practices
- **Clear workflow:** staging -> prod (don't skip staging)
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
