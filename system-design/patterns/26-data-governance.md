# Data Governance

## TL;DR
Manage data: ownership (who owns dataset?), lineage (where does data come from?), retention (how long keep?), access (who can use?). Critical for: compliance (GDPR, CCPA), debugging (data version = code version).

## Core Intuition
Data = asset. Asset needs governance: owner, version, lineage, retention, access control.

## How It Works

**Governance components:**

1. **Ownership:** who is responsible for data quality?
2. **Lineage:** where does data come from? (database -> ETL -> feature store)
3. **Versioning:** track data changes over time (DVC)
4. **Retention:** keep data for X days, then delete (privacy)
5. **Access control:** who can read/write this data?

| Component | Example |
|-----------|---------|
| Owner | Alice (data engineer) |
| Lineage | raw_logs -> clean -> features |
| Version | dataset_v1.4 (commit hash) |
| Retention | 90 days |
| Access | team_ml, team_analytics |

## Key Properties / Trade-offs
- Overhead: governance adds process overhead
- Compliance: mandatory for regulated domains
- Debugging: lineage crucial for reproducing issues

## Detailed Trade-off Analysis

| Governance Level | Setup Time | Compliance | Data Quality | Debugging | Cost |
|-----------------|-----------|-----------|--------------|-----------|------|
| None (ad-hoc) | 0 days | Not compliant | Poor | Impossible | $0 |
| Basic (Excel) | 2 days | Weak | 50% | Hard | $500/mo |
| Intermediate (tools) | 1 week | Good | 80% | Medium | $2K/mo |
| Full (automation) | 2 weeks | Compliant | 95%+ | Easy | $5K/mo |

**Decision:** Startup MVP -> basic. Series A -> intermediate. Regulated (healthcare, finance) -> full. GDPR required -> full.

---

## Production Failure Scenarios

**1. PII Leaked Into Training Data**
- **Symptom:** GDPR audit finds patient names in model training features.
- **Root Cause:** No automated PII scan in ingestion pipeline; data ingested as-is.
- **Detection:** Run Presidio + regex scan on all new data; quarantine rows with >0.9 PII confidence score.
- **Fix:** Block ingestion until PII removed or pseudonymized; retrain affected model after data remediation; document in compliance log.

**2. Schema Change Without Notification**
- **Symptom:** Downstream feature store breaks silently when upstream table adds nullable column; stale predictions served for days.
- **Root Cause:** No schema change registry or consumer notification system.
- **Detection:** Great Expectations schema test fails on next pipeline run; pipeline emits alert to data-quality channel.
- **Fix:** Implement schema registry with consumer subscription; breaking changes require 2-week notice plus migration guide before merge.

**3. Data Catalog Stale**
- **Symptom:** Analyst queries wrong (stale) table for 3 months, producing incorrect business metrics that fed into board reports.
- **Root Cause:** Catalog updated manually by data owners, not synced with actual table metadata.
- **Detection:** Automated catalog sync vs actual table metadata (weekly diff); flag entries where last_verified > 14 days.
- **Fix:** Auto-sync catalog from table metadata on every schema change; add "last verified" timestamp visible to all consumers.

**4. Access Control Too Permissive**
- **Symptom:** Contractor gains read access to customer PII table via inherited role; discovered only during quarterly audit.
- **Root Cause:** Role inheritance not audited for PII table access; role granted generically to all contractors.
- **Detection:** Quarterly access audit -- flag any non-employee with PII access; automated script compares role membership vs PII table ACLs.
- **Fix:** PII tables require explicit, individual-level access grants; no inheritance from generic roles; access expires automatically after 90 days.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Data catalog (Collibra/Alation) | $5,000/mo | 1 instance | $5,000 |
| PII scanning (Presidio on Lambda) | $0.0002/row | 10M rows/day | $60 |
| Data lineage tooling | $2,000/mo | 1 instance | $2,000 |
| Compliance engineer time | $200/hr | 20 hr/mo | $4,000 |
| **Total** | | | **~$11,060/month** |

Full governance infrastructure runs approximately $11K/month -- the largest single cost is the catalog platform license, which is justified in regulated domains (healthcare, finance) where compliance audit failures carry $500K-10M fines. For earlier-stage companies, the open-source stack (OpenMetadata + DVC + manual PII scanning) cuts this to $2-3K/month, with the tradeoff of higher engineer maintenance burden. The cost of poor governance in a single incident (debugging data provenance for a production failure or handling a GDPR deletion request manually) typically exceeds one month of tooling cost.

---

## Implementation Guidance

**Wrong:** Build data pipeline, assume quality is fine. Hope lineage is remembered.
**Right:** Assign owner per dataset. Document lineage explicitly. Version schema. Automate retention. Monitor quality.

**Wrong:** Governance as documentation checklist (manual, incomplete).
**Right:** Governance as automation (lineage tracked, versioning built-in, retention enforced, alerts on anomalies).

---

## Interview Q&A

**Q: A 10-stage data pipeline has unclear lineage. How do you instrument it without disrupting the existing pipeline?**
A: Instrument at boundaries, not inside stages. Add lineage metadata (input dataset version, stage name, output dataset version, timestamp) as a sidecar to each stage's output write call. Use Apache Atlas or OpenMetadata to auto-ingest this metadata via listeners on your message queue or data catalog. Start with the most critical path (raw -> features) and expand. The key deliverable is answering "given model v1.4, which raw input rows contributed to it?" -- trace backward through the DAG using the lineage graph.

**Q: GDPR user deletion request received. The data is in active tables, cold backups, and a feature store. How do you ensure complete deletion?**
A: Deletion requires three passes: (1) active databases -- run automated deletion script keyed on user_id across all known tables; (2) backups -- either restore+delete+re-backup (expensive) or use a "deletion tombstone" approach where restored backups suppress tombstoned user IDs; (3) feature stores -- purge the user's row and invalidate any cached features derived from their data. Log each deletion step with timestamp and executor for the audit trail. The full process should complete within 30 days (GDPR requirement); if your pipeline requires more than 14 days to identify all copies, your lineage is incomplete.

**Q: Who owns a dataset created by Team A but heavily used by Team B? There is a conflict when Team A wants to change the schema.**
A: Creator owns quality; consumer owns SLA. Team A is responsible for data correctness and schema stability; Team B is responsible for communicating usage requirements before changes. Resolve conflicts by formalizing a data contract: Team A commits to schema stability for 60 days after any change notice; Team B commits to updating consumers within that window. Escalate to a data governance board only when the contract is violated -- this keeps the board out of routine decisions.

**Q: Governance overhead is slowing down the ML team's iteration speed. What is the minimum viable governance stack?**
A: Four non-negotiable components, everything else can wait: (1) ownership -- one named person per dataset, no orphan datasets; (2) schema versioning -- prevents silent downstream breaks; (3) retention policy -- legal requirement, enforced automatically; (4) access audit logs -- needed for security investigations. Defer: full lineage automation, complex quality dashboards, automated PII scanning for non-regulated domains. Set these up manually first, automate when the manual overhead exceeds 4 hours/week.

**Q: The data catalog shows a dataset was last accessed 2 years ago. How do you decide whether to delete it?**
A: Check three things: (1) lineage -- does any live pipeline depend on it downstream? Even rare batch jobs may reference it; (2) compliance -- is it under a legal hold or within a required retention window? (3) value -- is it a point-in-time snapshot that cannot be recreated (e.g., pre-GDPR user profiles)? If all three are clear, archive to cold storage (Glacier) rather than delete outright; set a final deletion date 90 days out and notify stakeholders. Deletion is irreversible; archival is not.

**Q: A schema change is planned that will break three downstream consumers. How do you coordinate the migration?**
A: Use a two-version overlap strategy: (1) deploy the new schema as an additive change alongside the old schema (dual-write to both); (2) notify all three consumers with a migration guide and a 2-week window; (3) after all consumers have migrated, remove the old schema in a second deploy. Track consumer migration status in the schema registry. Never remove a field without confirming zero read traffic to it (monitor usage metrics for the deprecated column).

**Q: Your governance tool flags 3% of rows as potential PII. The data team says it is all false positives. How do you handle this?**
A: Don't unilaterally dismiss the flag. Steps: (1) sample 100 flagged rows manually and classify as true/false positive; (2) if false positive rate is high (>80%), tune the detection threshold or add a domain-specific allowlist; (3) document the decision and the sampling methodology; (4) if even 10% are true positives, quarantine those rows and remediate before ingestion. The cost of a false negative (real PII in training data) is far higher than the cost of a false positive (row dropped from training).

**Q: What early warning signs indicate that data governance is failing in a production ML system?**
A: Four signals: (1) engineers regularly ask "where does this feature come from?" -- lineage is undocumented or stale; (2) schema changes cause surprise pipeline failures -- no consumer notification process; (3) data deletion requests take more than 5 business days -- PII map is incomplete; (4) storage costs grow quarter-over-quarter without new data sources -- retention policies not enforced. Any one of these is a governance debt indicator; all four together mean a compliance incident is likely within 12 months.

---

## Monitoring & Observability

**Key metrics:** Lineage completeness (% of datasets with documented lineage), schema compatibility (% compatible downstream), data freshness (how recent?), retention policy compliance (% deleted on schedule), access audit trail completeness

**Alerts:** Lineage documentation missing (>7 days stale), schema version mismatch detected, retention deadline missed, unauthorized access detected, data quality anomaly (missing %, outliers %), ownership unassigned

## Common Mistakes / Gotchas
- No ownership: "nobody" owns data -> nobody fixes quality issues
- Forgotten lineage: can't debug, don't know where data came from
- No versioning: changed data schema -> broke downstream pipelines
- No retention: data accumulates -> storage explodes

## Best Practices
- **Clear owner:** assign one person per dataset
- **Document lineage:** create data DAG (directed acyclic graph)
- **Version datasets:** use DVC or similar
- **Access logs:** audit who accessed what data
- **Retention policy:** auto-delete data after N days (privacy)

## Code Example
```python
import dvc.api

# Register dataset ownership
dataset_metadata = {
    "name": "customer_profiles",
    "owner": "alice@company.com",
    "lineage": ["raw_logs", "clean_logs", "features"],
    "version": "v1.4",
    "retention_days": 90,
    "access": ["team_ml", "team_analytics"]
}

# Track with DVC
dvc.api.log_dataset(dataset_metadata)
```

## Interview Quick-Reference
| Component | Purpose |
|-----------|---------|
| Owner | Accountability |
| Lineage | Debugging |
| Version | Reproducibility |
| Retention | Privacy |
| Access | Security |

## Related Topics
- [Data Pipelines](02-data-pipelines.md)
- [ML Governance](27-ml-governance.md)

## Resources
- [Data Governance Best Practices](https://www.gartner.com/)
