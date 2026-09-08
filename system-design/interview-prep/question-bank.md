# ML/AI System-Design Question Bank

Run one prompt for 45–60 minutes using the [framework](system-design-framework.md)
and scorecard. Clarify scale, latency, labels, privacy, cost, and success
before proposing models. Every prompt requires a baseline, data/split plan,
serving path, offline/online evaluation, monitoring, failure handling, and
trade-offs. The linked reference is remediation, not a prepared answer.

## Product ranking and prediction

1. Recommendation feed with cold-start users and items — [inference caching](../patterns/08-inference-caching.md).
2. Search retrieval and ranking for rare identifiers — [model serving](../patterns/05-model-serving.md).
3. Ads click prediction with delayed conversions — [A/B testing](../patterns/14-ab-testing.md).
4. Transaction fraud detection with asymmetric costs — [model debugging](../patterns/17-model-debugging.md).
5. Demand forecasting with promotions and missing labels — [data pipelines](../patterns/02-data-pipelines.md).

## Data, training, and serving platforms

6. Point-in-time feature platform for batch and online consumers — [feature store](../patterns/03-feature-store.md).
7. Training platform with reproducible datasets and rollback — [reproducibility](../patterns/21-reproducibility.md).
8. Model registry and promotion workflow across environments — [model registry](../patterns/04-model-registry.md).
9. Low-latency online inference with burst traffic — [online vs batch](../patterns/07-online-vs-batch-inference.md).
10. Batch scoring platform with freshness and backfill guarantees — [data pipelines](../patterns/02-data-pipelines.md).

## Evaluation, quality, and governance

11. Experiment platform with sample-ratio mismatch detection — [A/B testing](../patterns/14-ab-testing.md).
12. Delayed-label monitoring for a risk model — [monitoring](../patterns/16-monitoring-and-observability.md).
13. Data-quality gates for a feature pipeline — [data pipelines](../patterns/02-data-pipelines.md).
14. Drift detection and retraining policy — [drift detection](../patterns/15-drift-detection.md).
15. Fairness and privacy controls for a personalization model — [fairness metrics](../patterns/25-fairness-metrics.md).

## LLM and multimodal systems

16. RAG assistant with citations and tenant isolation — [production readiness](../patterns/23-production-readiness.md).
17. LLM serving with streaming, batching, and cost routing — [model serving](../patterns/05-model-serving.md).
18. Fine-tuning pipeline with data review and regression gates — [model versioning](../patterns/06-model-versioning.md).
19. Evaluation platform for quality, safety, cost, and latency — [monitoring](../patterns/16-monitoring-and-observability.md).
20. Multimodal search with image/text retrieval and reranking — [inference caching](../patterns/08-inference-caching.md).

## Agents and human workflows

21. Tool-using customer-support agent with permissioned actions — [production readiness](../patterns/23-production-readiness.md).
22. Human-in-the-loop claims workflow with approval and audit — [model governance](../patterns/27-ml-governance.md).
23. Multi-agent research workflow with bounded fan-out — [cost optimization](../patterns/22-cost-optimization.md).
24. Safety architecture for an agent that can modify records — [privacy-preserving ML](../patterns/28-privacy-preserving-ml.md).
25. Cost-aware routing across models and deterministic fallbacks — [online vs batch](../patterns/07-online-vs-batch-inference.md).

## Follow-ups to ask yourself

For each prompt, explain the baseline, label maturity, leakage prevention,
candidate/model boundary, rollback path, load shedding, quality slices, and the
trade-off you would revisit first if the budget were cut in half.
