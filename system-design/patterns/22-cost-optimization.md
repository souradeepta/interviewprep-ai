# Cost Optimization

## TL;DR
Reduce ML system costs: model size (quantization, pruning), serving (batch, caching), compute (right-size infrastructure), data (reduce volume). Common 3-5x reductions.

## Core Intuition
ML is expensive: training, serving, data storage. Optimize where money goes.

## How It Works
**Model costs:**
- Quantization: 4x smaller, 2-3x faster, ~1-2% accuracy loss
- Pruning: remove unimportant parameters
- Distillation: small student model mimics large teacher

**Serving costs:**
- Caching: avoid recomputation
- Batch inference: cheaper per prediction
- Right-size: don't over-provision GPUs

**Data costs:**
- Retention: don't keep indefinitely
- Compression: reduce storage
- Sampling: train on sample, not full data

## Common Mistakes / Gotchas
- **Optimizing wrong thing:** cheap model, expensive serving. Look holistically.
- **Premature optimization:** optimize before measuring. Profile first.
- **Quality sacrificed:** cut costs too much → broken models. Measure impact.

## Interview Q&A

**Q: What are the highest-ROI cost optimization strategies for ML inference workloads?**
A: In order of ROI: (1) right-sizing instances (25-40% savings: match GPU memory to model size), (2) model quantization INT8/INT4 (2-4x cost reduction with minimal quality loss), (3) auto-scaling (30-50% savings by eliminating idle capacity during off-peak), (4) spot/preemptible instances for batch inference (60-70% savings), (5) continuous batching (2-3x throughput improvement on the same hardware). Model distillation (smaller model) has highest long-term ROI but requires upfront training cost and quality validation.

**Q: How do you balance model quality and inference cost when making optimization trade-offs?**
A: Quantify the cost of quality degradation: A/B test a cheaper model (smaller, quantized, faster) and measure business metric impact. If the cheaper model costs 50% less and reduces business metric by 2%, that's a concrete trade-off decision for stakeholders, not a purely technical one. Some quality degradation is worth the cost savings; some isn't. Present the trade-off with data, don't make the decision unilaterally. Implement the optimization with a canary deployment and measure actual business impact before full rollout.

**Q: How do you identify overprovisioned ML infrastructure?**
A: Indicators of overprovisioning: GPU utilization consistently <30%, CPU utilization <20%, memory usage <50% of provisioned. Check: average vs. peak utilization (provision for peak, not average), request queue depth (if near zero, capacity is sufficient and may be excess), cost per prediction over time (should be decreasing as you optimize, not flat). Use cloud cost allocation tags to attribute GPU spend to specific models. A model that costs $10K/month with <30% GPU utilization is likely overprovisioned and worth investigating.

**Q: What is model cascading and how does it reduce inference cost?**
A: Model cascading: use a cheap, fast model for easy cases and a powerful, expensive model only for hard cases. Example: a small BERT model handles 80% of text classification requests with high confidence; uncertain cases (<70% confidence) are escalated to a larger model. Cost reduction: 80% of requests use the cheap model, reducing average cost by 70-80% with minimal quality impact (the expensive model handles the cases it's actually needed for). Design the cascade threshold by measuring the accuracy distribution of the cheap model on a validation set.

**Q: How do you estimate and forecast ML infrastructure costs?**
A: Build a cost model: cost = f(requests per day, average tokens/request, model size, GPU type, batch size efficiency). Track: actual cost per 1K predictions by model, cost trend over time, cost breakdown by component (inference vs. feature store vs. data pipeline). Forecast using request growth projections. Alert when: actual cost exceeds forecast by >20% (unexpected usage spike or efficiency regression), cost per prediction increases unexpectedly (optimization regression). Make cost a first-class metric alongside accuracy and latency.

## Interview Quick-Reference
**Cost optimization?** Model size, serving efficiency, right-size infrastructure, data reduction.

## Related Topics
- [Model Compression](../ml/concepts/model-compression.md)
- [Quantization](../llm/concepts/quantization.md)

## Resources
- [Cost-Effective ML](https://www.oreilly.com/library/view/cost-effective-machine-learning/9781491990797/)
