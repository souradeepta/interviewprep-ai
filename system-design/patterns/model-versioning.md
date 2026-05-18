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

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
