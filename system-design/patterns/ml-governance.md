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

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
