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
