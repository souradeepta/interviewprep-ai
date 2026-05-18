# Data governance

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
- Classify all data assets by sensitivity (PII, confidential, public) before building pipelines
- Document data lineage from source through transformations to model features
- Implement access controls at the data layer, not just application layer
- Audit all data access for compliance — log who accessed what and when
- Define data retention policies and automate deletion
- Version datasets like code — immutable snapshots with metadata
- Conduct regular data quality audits and alert on schema drift

## Interview Q&A

**Q: What data governance processes are needed specifically for ML training data?**
A: ML-specific governance requirements beyond standard data governance: consent documentation for personal data used in training, purpose limitation documentation (data collected for X used to train model Y—is that in scope?), lineage tracking from raw data to model artifacts, training data retention policies that align with model lifecycle, and re-use policies when sharing training data across teams. Additionally: documentation of which demographic groups are represented in training data (affects bias and fairness documentation requirements).

**Q: How do you implement data access controls for ML training pipelines?**
A: Principle of least privilege: the ML training job should only access the specific data needed for that training run. Implement: IAM roles scoped to specific S3 buckets/prefixes, time-limited access tokens for training jobs, row-level security in databases (the training pipeline only sees the rows it's authorized for). Audit: log all data access during training and store with the model metadata. Prevent: training pipelines from having write access to source data, production models from having direct access to training data after deployment.

**Q: What is the right approach to managing PII in ML training data?**
A: Assess necessity: does the model actually need PII to work? Anonymize or pseudonymize if not. If PII is needed: document the legal basis, implement data minimization (only the PII fields needed), set retention limits (delete PII from training data after model deployment + x months), implement purpose binding (this dataset can only be used for model Y), and encrypt PII at rest. For model outputs: ensure the model can't reproduce training PII (test for memorization), implement output filtering for PII patterns.

**Q: How do you handle data governance for externally sourced datasets?**
A: Review the license terms carefully: can you use this data for commercial models? Is attribution required? Are there restrictions on model distribution? Common gotchas: web-scraped data may have conflicting copyright status, some open datasets prohibit commercial use, GDPR-covered data has special requirements. Track all external dataset licenses in your data catalog. Before using a new external dataset, require legal review. Avoid the "interesting dataset" trap: don't collect datasets without a clear use case and governance plan.

**Q: What does a data governance review process look like before training a new model?**
A: Checklist: (1) data provenance review—where does this data come from, is there consent/license? (2) PII assessment—what personal data is included, what's the legal basis for use? (3) bias assessment—which demographic groups are represented, is the sample representative? (4) retention plan—how long will training data be stored? (5) access control review—who can access this training data? (6) purpose documentation—what is this model for, is this data appropriate for that purpose? Require sign-off from data governance, legal, and privacy teams before training.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
