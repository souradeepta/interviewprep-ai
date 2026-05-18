# Ml governance

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
- Document model cards for every production model (intended use, limitations, metrics by subgroup)
- Require approval gates before production deployment — human sign-off for high-risk models
- Log model predictions in production for monitoring and audit
- Track model performance by demographic subgroups to detect disparate impact
- Set model expiration dates — trigger retraining reviews periodically
- Maintain rollback procedures with tested SLA
- Version control all configuration files, not just model weights

## Interview Q&A

**Q: What is model risk management and when is it required?**
A: Model risk management (MRM) is the process of identifying, assessing, and mitigating risks from model errors. Required by: US banking regulators (SR 11-7 guidance for banks), insurance regulators, and increasingly for high-risk AI under EU AI Act. MRM includes: model documentation (model purpose, assumptions, limitations), validation (independent review of model performance), ongoing monitoring (performance degradation detection), and governance (approval process for model deployment and changes). Companies outside financial services often implement MRM voluntarily for high-stakes models.

**Q: How do you structure model documentation for an ML governance audit?**
A: Required documentation: model card (purpose, capabilities, limitations), training data description (source, size, preprocessing, known biases), model architecture description, performance metrics (on multiple slices, not just overall), validation methodology (how was the model tested?), change log (what changed from previous version), known issues and mitigations, and intended use vs. out-of-scope use. Store documentation with model artifacts in the registry, version alongside the model. An audit should be able to reproduce key performance claims from the documentation alone.

**Q: What approval workflows are needed for high-risk ML model deployments?**
A: Approval chain for high-risk models: ML team sign-off (technical validation), data governance review (data usage compliance), security review (vulnerability assessment), legal/compliance review (regulatory requirements), business owner sign-off (accepts business risk), and for regulated industries, independent model validator sign-off. Codify approvals as blocking gates in the CI/CD pipeline—a model shouldn't deploy without all required approvals recorded. Track approval status in the model registry.

**Q: How do you handle a model that has been in production but was never properly validated?**
A: Don't immediately shut down the model—that would disrupt business operations. Instead: (1) document current model version and performance as the baseline; (2) run retrospective validation against historical data and compare to the undocumented baseline; (3) if validation reveals significant issues, implement monitoring and mitigations while working toward a proper replacement; (4) establish a governance timeline for bringing the model into compliance. Prioritize: high-risk models (consequential decisions) over low-risk (internal analytics).

**Q: What is the incident response process when an ML model causes a governance violation?**
A: Immediate: assess whether to take the model offline or implement emergency mitigations (e.g., filter outputs, escalate to human review). Document: what happened, when it was detected, who was affected, what the model did. Root cause analysis: was this a known limitation, a data issue, a model failure, or a deployment error? Remediation: implement technical fix, update governance documentation, improve monitoring to prevent recurrence. Reporting: depending on severity and regulation, may require notifying affected individuals, regulators, or senior management. Establish an ML-specific incident response playbook.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
