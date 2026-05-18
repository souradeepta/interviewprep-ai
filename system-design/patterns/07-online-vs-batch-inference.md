# Online vs Batch Inference

## TL;DR
Online: serve predictions per-request, real-time, high latency, per-request cost. Batch: precompute predictions offline, low latency retrieval, high throughput, cheaper. Choose based on SLA and use case.

## Core Intuition
For each user, compute immediately (online) or precomputed this morning (batch)?

## How It Works
**Online:** User → Request → Model → Predict → Response (100ms)
**Batch:** Precompute tonight, store, user queries cache (1ms lookup)

**When to use:**
- Online: personalization changes per request, user-specific context
- Batch: recommendations, scoring (can be daily), static predictions

## Key Properties / Trade-offs
| Aspect | Online | Batch |
|--------|--------|-------|
| Latency | 100-500ms | 1-10ms |
| Freshness | Real-time | Stale (lag) |
| Cost | High (per-request) | Low |
| Scalability | Limited (sync) | High |
| Personalization | Full | Limited |

## Common Mistakes / Gotchas
- **Batch for SLA <100ms:** can't meet latency. Need online.
- **Online for 1B users:** too expensive. Batch + cache.
- **No fallback:** online service down → no predictions. Have batch fallback.

## Interview Q&A

**Q: How do you decide whether to switch a batch inference system to online inference?**
A: Switch to online when: business logic requires real-time decisions (fraud detection, real-time recommendations), the latency between batch creation and consumption causes stale predictions, or user interactions require immediate responses. Stay with batch when: predictions can be precomputed (user-item scores computed nightly), input data isn't available in real-time (end-of-day transaction aggregations), or cost is a primary constraint (batch is 5-10x cheaper). Many systems use both: batch for expensive features, online for final scoring.

**Q: What is the optimal batch size for batch inference and how do you determine it?**
A: Batch size is a throughput vs. latency trade-off. GPU utilization peaks at batch sizes of 32-256 for most models. Test with increasing batch sizes and measure GPU utilization and throughput—the optimal batch size is where GPU utilization is >80% and latency is still acceptable. For time-constrained batch jobs, maximize batch size. For online pseudo-batch (grouping real-time requests), use dynamic batching with max-latency timeout (e.g., batch up to 32 requests or wait up to 10ms).

**Q: How do you handle stragglers in distributed batch inference?**
A: Stragglers (slow workers) in distributed batch jobs can delay the entire job. Mitigations: use speculative execution (re-run the slowest 5% of tasks on new workers), implement task timeouts (kill and retry tasks exceeding 3x median time), partition data to avoid skew (straggler is often due to uneven data distribution—sort by input complexity), and use heterogeneous hardware detection (don't assign large batches to slower machines). Monitor task completion time distribution, not just average—a bimodal distribution indicates stragglers.

**Q: What monitoring is needed for batch inference pipelines?**
A: Job-level: completion time trend (is the job getting slower?), success/failure rate, input data volume. Quality: distribution of prediction scores (compare against historical baseline), sample of predictions for manual review. Data quality: null rate in input features, count of inputs outside model's training distribution. Infrastructure: worker failures, memory usage, data read throughput. Alert on: job duration exceeding 2x historical average, prediction distribution shift >3σ, input data volume drop >20%.

**Q: How do you version and reproduce batch inference runs?**
A: For reproducibility, a batch inference run requires: model version, input data version (or snapshot), inference code version, inference configuration (batch size, preprocessing parameters). Store these provenance records in your model registry or a separate job tracking system. This enables: debugging (reproduce a specific run to trace a bad prediction), auditing (prove which model made which prediction), and regression testing (re-run a historical batch with a new model to compare).

## Interview Quick-Reference
**Online vs batch?** Online for personalized, real-time. Batch for static, precomputed. Hybrid common.

## Related Topics
- [Model Serving](05-model-serving.md) — online serving architecture
- [Inference Caching](08-inference-caching.md) — cache batch results

## Resources
- [Clipper: A Low-Latency Online Prediction Serving System](https://arxiv.org/abs/1611.08613)
