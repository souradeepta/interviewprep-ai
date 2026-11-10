# ML Governance, Lineage, and Auditability

**TL;DR:** Governed ML artifacts need traceable data, code, features, configuration, evaluation, ownership and approval metadata so teams can reproduce, review and retire decisions responsibly.

## Prerequisites

[Model versioning](06-model-versioning.md), [reproducibility](07-reproducibility.md), and [model registry](15-model-registry.md).

## Learning Outcomes

- Define the lineage fields needed to trace a model to code, data and feature versions.
- Validate required provenance before approving an artifact.
- Explain immutable metadata, checksums, approval records and audit retention.
- Identify privacy and access boundaries in lineage records.

## Core Intuition

A model file is like a shipped component without a bill of materials. If a prediction is challenged months later, the artifact alone cannot answer which data, transformation, configuration and evaluation produced it, who approved it, or whether the deployed service still matches that record. Lineage is the chain of custody for a model. Governance makes that chain reviewable and assigns accountable owners without assuming that metadata alone guarantees a fair or safe system.

## How It Works

Create an immutable release record that ties together model identifier and version, source-code revision, dependency/container digest, data snapshot or manifest, feature schema, training configuration, random seeds, evaluation report, artifact checksum, owner, reviewer and approval time. Record intended use, limitations, monitoring plan and retirement criteria where appropriate. The required fields should reflect the model's risk and deployment environment; a low-risk offline experiment may not need the same controls as a regulated decision system.

Lineage has both forward and backward directions. Backward lineage answers “what produced this artifact?” by traversing from model to training run, data and code. Forward lineage answers “where is this model used?” so teams can assess the impact of a data deletion, vulnerability, or model defect. Link registry entries to serving deployments and consuming products. Preserve version identifiers rather than mutable labels such as `latest`; a moving pointer is convenient for routing but weak evidence in an audit.

The implementation validates a minimum metadata record and calculates a stable SHA-256 fingerprint over canonical JSON. Canonical key ordering means equivalent dictionaries produce the same digest. The digest detects accidental metadata changes when compared with a trusted stored fingerprint; it does not prove that the data or claim is truthful, nor does an unhashed artifact become immutable by declaration. Store the signed or access-controlled record separately from the artifact and verify it during promotion and deployment.

Approval is a workflow with separation of duties where warranted. A model author should not silently approve their own high-risk release. Approval evidence should show the evaluation criteria, test outcomes, unresolved limitations and reviewer identity. Policy checks can block missing provenance fields, failed safety evaluation or an expired approval. Exceptions need a reason, scope, owner and expiry date. Governance should be proportional and automated where rules are crisp, while leaving human judgment for context that cannot be reduced to a checklist.

Auditability also depends on data stewardship. Record data source, collection purpose, consent or legal basis where applicable, retention classification and transformation lineage. Do not copy sensitive row-level data into registry metadata. Use stable opaque identifiers or references with access controls. If an upstream source is deleted or corrected, identify derived datasets and models through lineage and apply the relevant retention policy. Keep evaluation reports and approval records for the period required by policy, but do not retain source data indefinitely merely because a model exists.

Operationally, define a completeness gate before registry promotion. For example, require all nine fields used by this exercise and a reviewer distinct from the model owner; production policy can add risk tier, privacy and fairness review. The gate should fail closed for missing core provenance. If a field is genuinely not applicable, encode a documented reason rather than an empty string. Periodically sample deployed services and verify that their model digest, feature schema and approval record still match the registry.

Lineage schemas evolve. Version the schema itself and provide explicit migrations so an older record can still be read without silently changing its meaning. Do not rewrite historical provenance to fit a new field definition; preserve the original values and add a migration record. When an artifact is copied between registries or regions, verify its digest and maintain the chain of custody. Auditors should be able to determine which identity performed a promotion and which automated checks ran, not merely that a status field says “approved.”

Governance also requires lifecycle ownership. Each deployed model should have an accountable owner, review date, intended-use boundary, monitoring contact and retirement trigger. A model without an owner can remain live after its source data or product purpose has changed. Inventory scans should compare registry records to actual serving deployments and surface orphaned endpoints. Retirement should revoke deployment credentials, preserve required audit evidence, delete data according to policy and update downstream consumers with a migration path.

## Trade-offs

| Governance choice | Benefit | Cost or boundary |
|---|---|---|
| Immutable versioned records | Reliable historical reconstruction | Requires storage lifecycle and migration planning |
| Central registry | Consistent promotion gates | Can become a bottleneck or single point of failure |
| Required metadata schema | Auditable completeness | Overly broad schemas encourage meaningless filler |
| Separation of duties | Independent high-risk review | Adds release time and reviewer workload |
| Artifact fingerprint | Detects content change | Does not establish artifact correctness or trustworthy provenance |

## Implementation with Python

Run `python3 mlops/implementations/22-governance-lineage.py`. It checks required provenance fields and fingerprints canonical JSON. The operational decision is to reject promotion when any required field is absent; the required schema should be versioned and risk-adjusted.

```python
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("lineage", "mlops/implementations/22-governance-lineage.py")
lineage = module_from_spec(spec); spec.loader.exec_module(lineage)
record = {field: field for field in lineage.REQUIRED_FIELDS}
assert lineage.validate_lineage(record)["valid"]
```

## Failure Modes

- Mutable `latest` pointer replaces provenance: store immutable version IDs and resolve deployment digests.
- Metadata looks complete but is false: verify references, checksums and approvals against controlled sources.
- Sensitive data leaks into audit fields: store protected references and redact row-level values.
- Approval becomes a checkbox: require evidence, distinct reviewer when needed and scoped exceptions with expiry.

## Interview Quick Reference

| Question | Expected answer | Follow-up |
|---|---|---|
| What do you need to reproduce a model release? | Code, data snapshot, features, configuration, dependencies, evaluation, artifact digest and approvals. | Debug a model whose data snapshot was overwritten. |
| What does an artifact hash prove? | Only that bytes match a trusted digest; it does not prove quality or provenance truth. | Design verification at model promotion. |
| How do you govern a high-risk exception? | Document scope, evidence, accountable approver and expiry, then monitor it. | Design separation of duties without blocking urgent recovery. |

## Related Concepts

See [model registry](15-model-registry.md), [reproducibility](07-reproducibility.md), [model versioning](06-model-versioning.md), and [MLOps interview practice](../interview-prep/mlops-interview-questions.md#reliability-governance-and-cost).

## References

[NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) and [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/).
