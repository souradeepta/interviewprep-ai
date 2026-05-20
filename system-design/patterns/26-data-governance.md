# Data Governance

## TL;DR
Manage data: ownership (who owns dataset?), lineage (where does data come from?), retention (how long keep?), access (who can use?). Critical for: compliance (GDPR, CCPA), debugging (data version = code version).

## Core Intuition
Data = asset. Asset needs governance: owner, version, lineage, retention, access control.

## How It Works

**Governance components:**

1. **Ownership:** who is responsible for data quality?
2. **Lineage:** where does data come from? (database → ETL → feature store)
3. **Versioning:** track data changes over time (DVC)
4. **Retention:** keep data for X days, then delete (privacy)
5. **Access control:** who can read/write this data?

| Component | Example |
|-----------|---------|
| Owner | Alice (data engineer) |
| Lineage | raw_logs → clean → features |
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

**Decision:** Startup MVP → basic. Series A → intermediate. Regulated (healthcare, finance) → full. GDPR required → full.

---

## Production Failure Scenarios

**Scenario 1: Schema change broke downstream**
- Data team changed schema (added field). ML pipeline expects old schema. Pipeline fails silently, stale predictions served.
- Root cause: no schema versioning. No lineage documentation.
- Prevention: Schema version in data metadata. Downstream checks version, fails loudly if incompatible.

**Scenario 2: Compliance audit: where is user X's data?**
- GDPR deletion request. Can't find where user data is stored. Data scattered across 10 databases/backups.
- Takes 2 months to comply (should be 2 weeks).
- Prevention: Data lineage documented. Ownership assigned. Automated deletion scripts.

**Scenario 3: Data quality issue, root cause unknown**
- Model accuracy drops. Is it data quality issue (bad labels, missing values) or model drift? Can't trace data origin.
- Prevention: Data lineage + quality metrics. Monitor data at each pipeline stage (missing %, null %, outliers %).

**Scenario 4: Storage costs exploded**
- Data retained forever. No retention policy. Old data accumulated $1M storage cost.
- Prevention: Retention policy enforced. Auto-delete after N days. Archive hot → cold after M days.

---

## Implementation Guidance

**Wrong:** Build data pipeline, assume quality is fine. Hope lineage is remembered.
**Right:** Assign owner per dataset. Document lineage explicitly. Version schema. Automate retention. Monitor quality.

**Wrong:** Governance as documentation checklist (manual, incomplete).
**Right:** Governance as automation (lineage tracked, versioning built-in, retention enforced, alerts on anomalies).

---

## Sophisticated Interview Q&A

**Q1: Complex data pipeline (10 stages). How track lineage?**
A: (1) Create data DAG (directed acyclic graph). Each stage = node, inputs = edges. (2) Use tools: Apache Atlas, OpenMetadata, DataHub. (3) Auto-generate lineage from code (ETL logs). (4) Validate: can you trace user data from raw input to final output? (5) Alert on lineage breaks (pipeline stage failed).

**Q2: GDPR: user deleted. But data in backups. Compliant?**
A: (1) Not fully. GDPR requires "right to be forgotten"—must delete from all copies (active + backups). (2) Backup strategy: either don't back up PII, or delete PII from backups. (3) Retention policy: auto-delete backups after 30 days. (4) Audit trail: log all deletions.

**Q3: Data ownership: who owns dataset created by team A, used by team B?**
A: (1) Creator = owner by default (team A responsible for quality). (2) Consumer (team B) owns SLA (if data late, team B's problem). (3) Conflicts: escalate to data governance board. (4) Documents: explicit ownership agreement between teams.

**Q4: Data governance overhead too high. What's minimum viable?**
A: (1) Owner + lineage (critical). (2) Schema versioning (prevents downstream breaks). (3) Retention policy (legal requirement). (4) Access logs (auditing). (~2 weeks setup). (5) Defer: full lineage automation, complex quality checks. Start minimal, grow as needed.

---

## Cost & Resource Analysis

**Basic governance (Excel + manual processes):** 1 person × 0.5 FTE = $50K/year.
**Intermediate (DataHub/Collibra):** 2 people × 0.5 FTE + $2K/month tools = $150K/year.
**Full automation (DataHub + DVC + retention automation):** 2-3 people + $5K/month = $250K/year.
**Cost of poor governance:** Compliance fines $500K-10M+, debugging time $50K/incident, storage explosion $100K+.

**ROI:** Governance investment $150-250K/year. Prevents 1-2 incidents/year worth $500K+. Break-even easily justified.

---

## Monitoring & Observability

**Key metrics:** Lineage completeness (% of datasets with documented lineage), schema compatibility (% compatible downstream), data freshness (how recent?), retention policy compliance (% deleted on schedule), access audit trail completeness

**Alerts:** Lineage documentation missing (>7 days stale), schema version mismatch detected, retention deadline missed, unauthorized access detected, data quality anomaly (missing %, outliers %), ownership unassigned

## Common Mistakes / Gotchas
- No ownership: "nobody" owns data → nobody fixes quality issues
- Forgotten lineage: can't debug, don't know where data came from
- No versioning: changed data schema → broke downstream pipelines
- No retention: data accumulates → storage explodes

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

## Interview Q&A
**Q: Data pipeline changed schema. Downstream broke. How prevent?**
A: Data governance. Schema version stored with data. Downstream checks version before using. If version mismatch, alert before breakage.

**Q: GDPR: user requests data deletion. How to comply?**
A: Data governance. Know where user data stored (lineage). Delete from: databases, backups, archives. Audit trail shows what was deleted. Retention policy prevents accumulation.

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
