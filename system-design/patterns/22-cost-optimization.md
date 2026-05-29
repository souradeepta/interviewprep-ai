# Cost Optimization

## TL;DR
Reduce ML system cost without sacrificing quality. Techniques: model distillation (teacher → student, smaller), quantization (fp32 → int8), caching, cheaper models (GPT-3.5 vs GPT-4). Target: 30-50% cost reduction.

## Core Intuition
GPU costs $2.50/hour. Can you get similar accuracy with cheaper GPU ($0.50/hour)? Or smaller model? Save 80% cost.

## How It Works

**Cost levers:**

| Lever | Technique | Savings | Trade-off |
|-------|-----------|---------|-----------|
| Model size | Distillation | 50% | 2-3% accuracy drop |
| Precision | Quantization | 40% | Negligible |
| Hardware | Cheaper GPU | 70% | Slight latency increase |
| Inference | Batching, caching | 50% | None |
| Model | Simpler model (XGBoost vs NN) | 80% | More accuracy drop |

## Key Properties / Trade-offs
- Accuracy vs cost: can't eliminate cost without accuracy cost
- Latency vs cost: optimizing cost might increase latency
- Complexity: optimization increases code complexity

## Detailed Trade-off Analysis

| Optimization | Cost Savings | Accuracy Drop | Latency Impact | Complexity | Time to Deploy |
|--------------|--------------|---------------|----------------|-----------|-----------------|
| Distillation | 50% | 2-5% | -30% (faster) | Medium | 1-2 weeks |
| Quantization | 40% | <1% | -5% (slight slower) | Low | 3-5 days |
| Model pruning | 30% | 1-2% | -20% (faster) | Low | 3-5 days |
| Caching | 50% | 0% | -80% (cache hit) | Low | 2-3 days |
| Cheaper GPU | 70% | 0% | +20% (slower) | None | 1 day |
| Tiered models | 40% | 1% (complex cases) | Variable | High | 2-3 weeks |

**Decision:** Time-constrained → quantization. Accuracy-critical → caching. Mature product → distillation + tiering.

---

## Production Failure Scenarios

**Failure 1: Spot Preemption Mid-Training**
- **Symptom:** A 40-hour training run is lost at hour 39 with no checkpoint. The team must restart from scratch, costing $785 in compute and a full day of delay.
- **Root cause:** Spot instance preempted during the final epoch. No checkpointing was configured because it "seemed unnecessary for a near-finished run."
- **Detection:** CloudWatch or equivalent provides a 2-minute spot interruption notice. Hook into this event and trigger an immediate checkpoint save.
- **Fix:** Checkpoint every 30 minutes regardless of estimated time remaining. Implement a SIGTERM handler that saves a final checkpoint before the instance is terminated. Never run spot training without a checkpoint-on-interrupt path.

**Failure 2: Development Cluster Left Running**
- **Symptom:** $10K/month wasted on idle GPU instances that engineers forgot to shut down after experiments.
- **Root cause:** No auto-shutdown policy for development clusters. Engineers focus on their experiment and forget to terminate.
- **Detection:** Billing anomaly alert: alert if daily GPU spend on non-production accounts exceeds $500. Tag all dev clusters distinctly from production.
- **Fix:** Auto-shutdown after 1 hour of idle CPU (< 5% utilization). Require explicit renewal to extend beyond the idle threshold. Use Infrastructure-as-Code (Terraform) so clusters are ephemeral and must be explicitly recreated.

**Failure 3: Wrong Instance Type Selected**
- **Symptom:** Training takes 10× the expected time at 3× the expected cost. The team suspects a code regression but the real cause is running a GPU workload on a CPU instance.
- **Root cause:** The instance configuration was copied from a preprocessing job (CPU) and not updated for the training job (GPU). GPU utilization shows 0% — the model is running on CPU.
- **Detection:** Compare actual TFLOPS to theoretical GPU TFLOPS. If GPU utilization is under 10% for a neural network training job, the workload is not running on GPU.
- **Fix:** Maintain a benchmark table of instance type vs workload type with minimum GPU utilization thresholds. Require approval and a utilization justification for any instance costing over $5/hr.

**Failure 4: Cross-Region Data Transfer Cost Spike**
- **Symptom:** An unexpected $5K bill attributed to data transfer — a cost category that the team rarely monitors.
- **Root cause:** Training cluster was deployed in us-east-1 but training data lives in us-west-2. Every epoch reads the full dataset across regions at $0.08/GB.
- **Detection:** Tag and monitor data transfer costs separately from compute and storage. Alert on daily transfer costs exceeding $200 on a single workload.
- **Fix:** Co-locate training and data storage in the same region. Use S3 Transfer Acceleration only for genuinely unavoidable cross-region access. When migrating a dataset, pay the one-time transfer cost once and store in the training region permanently.

**Failure 5: Over-Aggressive Quantization Breaks Business Metric**
- **Symptom:** Post-quantization accuracy drop reported as 1% on the test set — but the business reports a 12% drop in conversion rate over the following two weeks.
- **Root cause:** The test set overrepresents easy cases. The 1% accuracy drop concentrates in the margin cases (confidence 0.45-0.55) that drive the most conversions.
- **Detection:** Evaluate accuracy on stratified slices: high-confidence (>0.8), medium-confidence (0.5-0.8), and low-confidence (<0.5). The quantization impact is almost always worst in the medium-confidence band.
- **Fix:** Set an accuracy floor on medium-confidence predictions before approving any quantization. The business metric impact should be simulated on a holdout with realistic production traffic proportions before deployment.

---

## Implementation Guidance

**Wrong:** Optimize for cost without measuring business impact. Cut accuracy 5%, lose revenue.
**Right:** Define accuracy floor. Measure before-after on business metrics (revenue, churn, NPS). Only optimize if impact is acceptable.

**Wrong:** Optimize everything at once (distillation + quantization + pruning). Confounded results.
**Right:** Optimize one lever at a time. Measure. Then add next. Isolate impact of each.

---

## Interview Q&A

**Q1: You need to cut ML inference cost by 60% (from $100K to $40K/month). How do you approach it?**
A: Start by profiling before optimizing — determine whether the cost is compute, storage, or networking. For compute-dominated costs: quantization (int8) saves ~40%, and caching repeated requests saves ~50% of cached-query costs without accuracy impact. Combined with moving to spot or preemptible instances (60-70% hardware savings), reaching 60% overall is achievable. Validate each step independently: apply quantization, measure accuracy on business metrics, confirm savings, then apply the next lever. Never combine all optimizations simultaneously — you lose the ability to attribute regressions.

**Q2: Distillation reduces accuracy by 2%. Is it acceptable?**
A: Context-dependent. A 2% test-set drop almost always maps to a larger business metric impact (3-5%) because test sets oversample easy cases. Before deciding, simulate the accuracy drop on the medium-confidence population specifically — those are the predictions where distillation errors concentrate. If the business metric impact is under 0.1% (e.g., click-through rate), the cost savings likely justify the loss. If the impact is measurable and the model is used for high-stakes decisions (credit, medical), 2% is not acceptable.

**Q3: A cheaper GPU class is available at $0.50/hr vs your current $2.50/hr, but latency doubles. Is it worth it?**
A: Calculate the total savings: $2/hr × 730 hr/month × number of instances = $1,460/month per instance. Then calculate the business cost of the latency change. If your SLA is 100ms and the new latency is 200ms, you breach SLA — the cost is contract penalties plus customer churn. If your SLA is 500ms and the new latency is 300ms, you are well within bounds and the switch is straightforward. The decision hinges entirely on the SLA, not the raw latency number.

**Q4: How do you route requests in a tiered model system to distinguish easy from hard cases?**
A: Three routing strategies: (1) Confidence-based — if the cheap model's softmax confidence is below 0.6, route to the expensive model. Works for classification but not regression. (2) Feature-based — pre-identify hard cohorts (new users, rare categories, out-of-distribution inputs) and route them directly to the expensive model, bypassing the cheap model entirely. (3) Ensemble router — a small binary classifier (100-parameter neural network) trained to predict whether the cheap model will be wrong, using the cheap model's output as one of its inputs. The router adds ~2ms but saves the 50ms expensive-model call on 90% of traffic.

**Q5: When would you NOT optimize cost, even under budget pressure?**
A: Three situations: (1) The model is in a critical safety path (autonomous driving, medical diagnosis) where any accuracy drop creates unacceptable risk. (2) The optimization would take longer to implement than the budget period justifies — a $2K/month savings requiring 4 weeks of engineering time (at $10K/week) never breaks even. (3) The system is about to be replaced — optimizing infrastructure that will be deprecated in three months is pure waste. Before any optimization, estimate the payback period: implementation cost / monthly savings. If payback > 12 months, reconsider.

**Q6: What breaks at 10× inference scale — from 1M to 10M predictions/day?**
A: Three things break: (1) The cache hit rate drops as the request distribution becomes more long-tail — 10× more unique queries means each query is requested less often, so the cache covers less traffic. Revisit cache key design and TTL at 10× scale. (2) The quantized model's throughput ceiling is hit — a single int8 GPU instance handles a fixed number of requests per second. At 10×, you need either horizontal scaling (more instances) or a more aggressive optimization like batching with dynamic shapes. (3) Logging costs scale linearly with predictions — at 10× volume, prediction logging can become a larger cost than the model itself. Implement sampling-based logging (log 1% of predictions) rather than exhaustive logging.

**Q7: An engineer suggests switching from a fine-tuned BERT model to GPT-4o for all queries to simplify the codebase. The inference cost will go from $5K to $180K per month. How do you evaluate this?**
A: This is a classic accuracy-cost-maintainability triangle. First, benchmark GPT-4o on your task — it may be significantly more accurate, which has a revenue-side benefit. Second, model the total cost of ownership for BERT: $5K inference + $15K/year engineer time for maintenance + $3K/quarter retraining = ~$8K/month total. Compare to GPT-4o's $180K with near-zero maintenance. Third, consider the hybrid: use GPT-4o for the 5% of hard or high-value cases and keep BERT for the other 95%. This typically captures 80% of the accuracy gain at 20% of the cost increase.

**Q8: Cost per prediction is increasing month-over-month despite stable traffic. How do you debug this?**
A: Compare the cost-per-prediction ratio across time periods. If cost is rising faster than traffic, the problem is either: (1) the model has grown (a new model version is larger or slower), (2) caching efficiency has dropped (cache keys changed, TTL reduced, or distribution shifted), (3) instance types were upgraded without budget approval, or (4) a dependency cost (embedding API, vector DB query) increased. Tag costs by component — compute, storage, external APIs — to isolate the driver. A cost dashboard that shows cost/prediction (not just total cost) per component is the prerequisite for this kind of debugging.

---

## Cost & Resource Analysis

**Cost breakdown (typical 100K predictions/day, GPU inference):**
- Compute: $5K/month (GPU hours)
- Storage: $500/month (model artifacts, logs)
- Ops: $2K/month (monitoring, scaling)
- **Total: $7.5K/month baseline**

**Optimization ROI:**
- Quantization: $3K savings, $1K implementation = 3x ROI
- Caching (30% cache hit rate): $2.5K savings, $500 implementation = 5x ROI
- Distillation: $2.5K savings, $5K implementation = 0.5x ROI (break-even in 2 months)
- Tiered models: $3K savings, $10K implementation = 0.3x ROI (6-month payback)

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| On-demand A100 (p4d.24xlarge) | $32.77/hr | 100 hr/month | $3,277 |
| Spot A100 (p4d.24xlarge, 80% of runs) | $9.83/hr | 80 hr/month | $786 |
| S3 storage (model checkpoints) | $0.023/GB | 500GB | $11.50 |
| Data transfer (same region) | $0 | — | $0 |
| **Total (mixed on-demand + spot)** | | | **~$1,085/month** |
| **Savings vs all on-demand** | | | **~$2,203/month** |

The largest single cost lever in GPU training is spot vs on-demand pricing: running 80% of training jobs on spot at ~$9.83/hr instead of $32.77/hr saves roughly $2,200/month for a team running 100 hours of A100 compute. The risk is preemption (roughly 5-15% interruption rate for A100 spot in peak hours), which requires checkpointing every 30 minutes. S3 checkpoint storage at $11.50/month is a negligible insurance premium against losing a multi-hour training run. Data transfer is zero when training and data are co-located in the same AWS region — cross-region transfers can easily add $500-5,000/month at scale and should be treated as a hard architectural constraint.

---

## Monitoring & Observability

**Key metrics:** Cost per prediction, accuracy per cohort, cache hit rate, latency by model tier, GPU utilization, model size, inference throughput

**Alerts:** Cost per prediction increases >10%, accuracy drops after optimization, cache hit rate drops <20%, GPU OOM errors, latency breaches SLA

## Common Mistakes / Gotchas
- Optimizing wrong thing: optimize latency when bottleneck is storage
- Too aggressive: distill so much model is useless
- Not measuring: "optimized cost" but didn't actually measure savings

## Best Practices
- **Baseline cost:** establish current cost per prediction
- **Target:** reduce by 30%, measure if achieved
- **Prioritize:** optimize most expensive component first
- **Validate:** measure accuracy after each optimization, stop if quality degrades
- **Portfolio:** use cheaper model for easy cases, expensive for hard

## Code Example
```python
# Model distillation
teacher_model = load_large_model()  # 500MB, 100ms latency
student_model = train_student(teacher_model, X)  # 50MB, 20ms latency

# Inference
output = student_model.predict(X)  # Faster, cheaper

# Quantization (fp32 → int8)
quantized_model = torch.quantization.quantize_dynamic(model)
# 4x smaller, similar accuracy
```

## Interview Quick-Reference
| Technique | Cost Savings | Accuracy Impact |
|-----------|---|---|
| Distillation | 50% | 2-3% drop |
| Quantization | 40% | <1% drop |
| Caching | 30% | 0% |
| Batch | 50% | 0% |
| Spot instances | 60-70% | 0% (requires checkpointing) |

## Related Topics
- [Model Serving](05-model-serving.md)
- [Request Batching](09-request-batching.md)

## Resources
- [Model Optimization Techniques](https://pytorch.org/tutorials/recipes/quantization.html)
