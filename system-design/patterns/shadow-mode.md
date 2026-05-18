# Shadow mode

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
- Route 100% of production traffic to shadow model — don't subsample if data volume allows
- Compare distributions of predictions, not just aggregate metrics
- Set a fixed time window for shadow evaluation (1-2 weeks) before deciding
- Measure shadow model latency independently — it must meet production SLA
- Log shadow model outputs with the same schema as production
- Run shadow evaluation for full business cycles (weekdays + weekends, seasonality)
- Define promotion criteria upfront, not after seeing shadow results

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
