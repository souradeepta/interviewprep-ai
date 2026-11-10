# ML Security and Privacy Controls

**TL;DR:** Protect ML systems with least-privilege access, data minimization, retention limits and auditable controls across training, evaluation, serving and incident response.

## Prerequisites

[Data pipelines](01-data-pipelines.md), [model registry](15-model-registry.md), and basic identity/access-control concepts.

## Learning Outcomes

- Apply deny-by-default authorization to model and data operations.
- Identify sensitive content in logs, feature records and model artifacts.
- Select a retention action with explicit age and legal-hold inputs.
- Explain privacy threat boundaries such as prompt leakage, memorization and overly broad access.

## Core Intuition

An ML system is not protected just because its model endpoint requires a password. Data flows through ingestion, feature storage, notebooks, training workers, registries, logs, evaluation dashboards and incident tickets. Each copy creates a place where a sensitive record or credential may leak. Security is the collection of controls that limit who can do what to which resource, while privacy governs what data is collected, used, exposed and retained. Both need to cover the full lifecycle.

## How It Works

Start with a threat model and data inventory. Identify assets (raw training data, feature values, labels, model weights, prompts, credentials, predictions), actors (users, services, operators, vendors) and trust boundaries. Consider unauthorized access, malicious training data, model extraction, prompt injection, memorization, membership inference, data poisoning and supply-chain compromise. Rank each threat by likelihood and impact, then select controls and owners. A model's risk depends on its use and data; the same technical artifact can require different protections in a prototype and a high-impact production decision.

Authorization should be explicit and least-privilege. Grant a role a narrow action on a resource class: for example, on-call staff may read aggregate latency metrics but not raw customer records. Deny by default and log decisions. Separate duties for training, approval and production deployment when risk warrants it. Service identities should have short-lived credentials and scoped access; avoid long-lived secrets in notebooks, container layers or logs. Reassess grants when an employee changes role or a service is retired.

Data minimization reduces both privacy risk and breach blast radius. Collect only fields with a documented purpose, remove direct identifiers where not needed, and consider aggregation, pseudonymization or differential privacy for suitable analytics. Pseudonymized data may still be re-identifiable when joined with other sources, so treat it as sensitive. Restrict raw-data access and prefer aggregate metrics for operational monitoring. Redaction should be applied before logs leave the trust boundary; a later deletion from a dashboard does not remove copies in backups or exports.

Retention is an operational control, not a vague aspiration. Assign a retention duration to raw events, feature snapshots, logs and model artifacts based on purpose and policy. The example implementation deletes or expires records whose age exceeds the configured limit, while respecting a legal hold. A real system needs deletion propagation to replicas, caches and derived artifacts, plus an auditable record of completion. Legal hold should be scoped and reviewed; it is not an indefinite exception for every related dataset.

Security testing should include adversarial and misuse scenarios. Test whether an endpoint returns sensitive training examples under crafted prompts, whether a model can infer membership, whether uploaded data can poison features, and whether logs accidentally include secrets. Red-team prompts are one part of the control set; they do not replace authentication, rate limits, output filtering, audit and incident response. Monitor unusual query volume and model extraction patterns, with clear escalation thresholds that respect user privacy.

Privacy-preserving techniques have assumptions. Differential privacy can bound the contribution of an individual to a training result, but the guarantee depends on adjacency definition, clipping, noise, composition and implementation correctness. Encryption at rest and in transit protects particular boundaries but does not stop authorized misuse or data exposure in model outputs. Federated learning changes where raw data is held but introduces aggregation, poisoning and client security concerns. State what a control protects and what it does not prove.

Operationally, review a control matrix at release time: data classification, authorized roles, logging/redaction, retention, model-output risk, dependency provenance, monitoring and deletion path. For example, deny a production model promotion if high-sensitivity data lacks an approved purpose or if its logs are not redacted. The local code demonstrates allow/deny, field redaction and retention logic only; production controls require authenticated identities, secure storage, policy enforcement and audit logging.

## Trade-offs

| Control | Benefit | Limitation |
|---|---|---|
| Least privilege | Limits accidental or malicious access | Requires ownership and regular grant review |
| Redaction | Reduces sensitive log exposure | Can remove context needed for debugging |
| Short retention | Reduces stored-data risk | May limit incident investigation and reproduction |
| Differential privacy | Formal bound on individual influence under assumptions | Utility cost, privacy-budget composition and implementation complexity |
| Encryption | Protects data at selected storage/transport boundaries | Does not prevent authorized misuse or model memorization |

## Implementation with Python

Run `python3 mlops/implementations/23-security-privacy-controls.py`. The standard-library functions deny access unless a matching role/action/resource grant exists, redact known top-level sensitive fields, and compute a retention action with a legal-hold override. The policy threshold is the configured retention duration; field inventories and durations must come from the organization’s data policy.

```python
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("controls", "mlops/implementations/23-security-privacy-controls.py")
controls = module_from_spec(spec); spec.loader.exec_module(controls)
assert not controls.authorize(role="analyst", action="read_raw", resource_class="row", grants=set())["allowed"]
```

## Failure Modes

- Broad role grants expose raw data: deny by default and periodically review access against job need.
- Sensitive content reaches logs: redact at the boundary, scan outputs and test with synthetic secrets.
- Retention expires only in the primary store: track deletion through replicas, backups, caches and derived data.
- A privacy technique is overclaimed: document assumptions, threat model and residual risks; validate implementation.

## Interview Quick Reference

| Question | Expected answer | Follow-up |
|---|---|---|
| How would you let on-call debug a model without seeing customer records? | Grant aggregate observability, redact identifiers and provide an audited break-glass path if required. | Design a test that proves raw fields are absent from logs. |
| Does encryption prevent model memorization? | No; it protects storage/transport boundaries but not learned behavior or outputs. | Debug a prompt that reveals a training example. |
| How do you enforce data deletion? | Track lineage, retention and deletion acknowledgements across stores and derivatives. | Design deletion verification after a backfill. |

## Related Concepts

See [data versioning](04-data-versioning.md), [model governance](22-governance-lineage-auditability.md), [model monitoring](17-monitoring-alert-design.md), and [MLOps interview practice](../interview-prep/mlops-interview-questions.md#reliability-governance-and-cost).

## References

[NIST Privacy Framework](https://www.nist.gov/privacy-framework) and [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf).
