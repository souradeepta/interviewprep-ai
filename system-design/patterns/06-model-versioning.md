# Model versioning

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
- Tag every model artifact with a semantic version (major.minor.patch) tied to a git commit
- Store model metadata (training data hash, hyperparameters, metrics) alongside the artifact
- Use a model registry (MLflow, Weights & Biases) — don't just store files in S3 with dates
- Never overwrite model versions — treat them as immutable artifacts
- Automate promotion criteria (must pass eval thresholds before promotion to staging/prod)
- Keep model lineage — which dataset and code version produced this model
- Test model versions in shadow mode before promoting to production

## Interview Q&A

**Q: What constitutes a new model version vs. a model update?**
A: Treat as a new version: different architecture, different training data, different hyperparameters that change model behavior, schema changes to inputs or outputs. Treat as a minor update: bug fixes in serving code, infrastructure changes that don't affect predictions, documentation updates. The test: would two versions produce different predictions for the same input? If yes, it's a new version that requires evaluation and promotion gates. If no, it's an infrastructure change with different deployment procedures.

**Q: How do you version models when training data changes frequently?**
A: Version the training data as part of the model version: store a hash of the dataset or a reference to a specific data snapshot. This allows you to answer: "what data was this model trained on?" and "if we retrain on the same data, do we get the same model?" Without data versioning, model versions are not reproducible—you can't debug regressions by comparing model versions. DVC (Data Version Control) or Delta Lake are standard tools for data versioning in ML workflows.

**Q: What is the shadow registry pattern and when do you use it?**
A: Shadow registry: a model version that receives a copy of production traffic (shadow mode) without affecting real responses. Use it for: evaluating a candidate model on production data distribution before promotion, comparing predictions between versions without risk, and validating that a new version handles edge cases in production. Shadow mode requires: infrastructure to duplicate requests, logging to compare shadow vs. production predictions, and a comparison framework to identify meaningful differences.

**Q: How do you maintain multiple model versions in production simultaneously?**
A: Multiple production versions are needed for: A/B testing, gradual rollout, customer-specific model versions, and multi-tenant deployments. Implementation: version the serving endpoint (model_id or version_id as a request parameter), route requests to the appropriate version at the load balancer, maintain separate scaling policies per version. Monitor each version independently—a new version may fail on a specific segment of traffic that the aggregate monitoring misses. Have a sunset policy: versions older than N months should be deprecated.

**Q: How do you handle the model registry when models are trained with different frameworks?**
A: Use a framework-agnostic serialization format: ONNX (covers most frameworks), pickle for Python-specific models, SavedModel for TensorFlow. Store both the native format (for retraining) and a serving-optimized format (ONNX/TensorRT). Include the framework version in the model metadata—models trained with PyTorch 1.x may not load with PyTorch 2.x. Design your serving infrastructure to support multiple model formats or standardize on one (ONNX is the most portable choice).

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
