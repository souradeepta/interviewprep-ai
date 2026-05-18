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

## Interview Q&A

**Q: What is shadow mode and how does it differ from canary deployment?**
A: Shadow mode: the new model receives a copy of all production traffic and generates predictions, but those predictions are discarded—production responses still come from the current model. Canary: a fraction of users receive responses from the new model. Shadow mode has zero risk to users but requires duplicate infrastructure cost. Use shadow mode for: validating new models on production data distribution before any user exposure, comparing predictions between model versions, and testing infrastructure changes. Use canary after shadow mode validates basic correctness and quality.

**Q: How do you handle the infrastructure cost of running shadow mode at scale?**
A: Shadow mode doubles inference cost. Mitigate: sample traffic (run shadow on 10-20% of requests, not 100%), use shadow in off-peak hours for CPU/memory-bound models, share GPU capacity during low-utilization periods. For very expensive models (large LLMs), shadow mode may be cost-prohibitive—use synthetic traffic replay instead (collect real requests, replay them asynchronously against the shadow model). Track shadow mode cost as a separate budget line and time-box shadow periods (1-2 weeks max).

**Q: What metrics should you compare between shadow and production during shadow mode?**
A: Prediction comparison: what fraction of inputs get different predictions? If >10% are different, investigate root cause before promoting. Quality proxy metrics: if you have labels for recent inputs, compare accuracy. Distribution metrics: are the shadow model's prediction score distributions similar to production? Performance: shadow model latency/throughput (must meet production SLAs before promotion). Silent failure detection: are there input types where the shadow model errors while production succeeds?

**Q: How do you use shadow mode to validate model changes that are intentionally different?**
A: When you expect the new model to produce different (better) outputs, shadow comparison isn't a pass/fail test—it's data collection. Collect: the distribution of differences (what types of inputs get different predictions?), human evaluations of a sample of differing predictions (which model's output is better?), and downstream metric impact simulation (if we had served the shadow model's outputs, what would engagement/conversion look like?). Shadow mode becomes a model evaluation pipeline, not just a regression detector.

**Q: What are the limitations of shadow mode for testing ML model correctness?**
A: Shadow mode can't test: user interactions that depend on model output (a recommendation model in shadow mode can't test whether users click the shadow model's recommendations), long-term behavioral effects (session quality requires users to actually experience the model), and models with side effects (if the model's output triggers other actions). For conversational AI, shadow mode is especially limited because responses depend on user reactions, creating fundamentally different conversations. In these cases, a canary with proper user assignment is required.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
