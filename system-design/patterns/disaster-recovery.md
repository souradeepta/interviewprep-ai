# Disaster recovery

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
- Define RPO (Recovery Point Objective) and RTO (Recovery Time Objective) before designing recovery
- Test recovery procedures quarterly — untested backups often fail
- Store backups in a different geographic region from primary
- Automate failover — manual failover during incidents is slow and error-prone
- Practice chaos engineering to validate recovery before disasters happen
- Document step-by-step recovery runbooks with owner assignments
- Keep shadow warm standby systems for critical ML inference endpoints

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
