# Model registry

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
- Store model artifacts, metadata, and metrics together as a single versioned unit
- Implement promotion workflows: experiment → staging → production with approval gates
- Use tags for searchability (model type, task, dataset, team)
- Automate metric comparison before promotion — require improvement over current prod
- Integrate registry with CI/CD for automatic registration on successful training runs
- Store model cards and bias evaluation alongside model artifacts
- Keep registry access audited — log all reads, writes, and promotions

## Interview Q&A

**Q: What metadata should every model version in the registry include?**
A: Mandatory: training data version (or dataset hash), hyperparameters, evaluation metrics on test set, training code version (git commit), training infrastructure (GPU type, framework version), model size and inference latency benchmarks. Highly recommended: data lineage (what raw data was used), feature importance, behavioral tests results, fairness/bias metrics, and the business context (what problem this model solves). Without this metadata, debugging production issues or replicating experiments is extremely difficult.

**Q: How do you implement model promotion gates (dev to staging to production)?**
A: Each gate should have automated checks: dev to staging (unit tests pass, basic metrics above floor threshold, no regression vs. baseline), staging to production (full eval suite on held-out data, A/B test results meet statistical significance, latency/throughput benchmarks met, security scan clean). Some checks require human approval (model card review, compliance sign-off for regulated use cases). Never promote a model without automated gates—a broken model in production is more expensive than the time saved by skipping checks.

**Q: How do you handle rollback when a production model starts failing?**
A: Design for fast rollback from the start: keep the previous production model artifact registered and deployable in <5 minutes. Implement automated rollback triggers: if key metrics drop >X% in the first hour after deployment, automatically roll back. Have a manual rollback runbook that any on-call engineer can execute without ML expertise. After rollback, post-mortem to identify the root cause before attempting re-deployment. The ability to roll back quickly is more important than preventing all bad deployments.

**Q: What is the difference between a model registry and a model store?**
A: Model store: binary artifact storage (S3, GCS)—stores model files, serialized weights, ONNX files. Model registry: metadata management and lifecycle tracking—stores the version history, evaluation metrics, deployment status, and lineage of models. You need both: the registry knows about model versions and their metadata, and points to artifacts in the store. MLflow, Weights & Biases, and SageMaker Model Registry combine both; many teams use S3 + a custom metadata database separately.

**Q: How do you manage model registry access control in a multi-team environment?**
A: Implement role-based access: data scientists can register and evaluate models; ML engineers can promote to staging; deployment approvals require additional sign-off. Restrict production promotion to an automated CI/CD system (not individual humans). Maintain an audit log of all registry operations: who registered, promoted, or deprecated each model version. For regulated industries, the audit log is a compliance requirement, not just a best practice.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
