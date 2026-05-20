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

**Scenario 1: Over-aggressive quantization**
- Quantize to int8. Accuracy drops 15%. Business impact: $100K/month lost revenue.
- Company spent $5K optimization cost to lose $100K revenue.
- Prevention: Set accuracy floor before optimizing. Test on business metrics, not just test set accuracy.

**Scenario 2: Cache invalidation not tracked**
- Cache predictions, forget to invalidate on model update. Stale predictions served 2 weeks.
- Users complain about accuracy.
- Fix: Always invalidate cache on model deployment. Monitor cache age. Alert if >24h old.

**Scenario 3: Cheaper GPU runs out of memory**
- Switch to cheaper GPU to save $2K/month. Model doesn't fit, falls back to CPU (slower, more expensive).
- No cost savings, plus latency increases.
- Prevention: Test model on target hardware before deploying. Benchmark memory and latency.

**Scenario 4: Distilled model only optimized for common cases**
- Distill on common inputs. Rare edge cases have 50% accuracy drop.
- Model seems good on test set (99% common cases), fails on production (1% rare cases high error).
- Fix: Validate distillation on full distribution, including rare cases. Use stratified test set.

---

## Implementation Guidance

**Wrong:** Optimize for cost without measuring business impact. Cut accuracy 5%, lose revenue.
**Right:** Define accuracy floor. Measure before-after on business metrics (revenue, churn, NPS). Only optimize if impact is acceptable.

**Wrong:** Optimize everything at once (distillation + quantization + pruning). Confounded results.
**Right:** Optimize one lever at a time. Measure. Then add next. Isolate impact of each.

---

## Sophisticated Interview Q&A

**Q1: Need to cut cost 60% (from $100K to $40K/month). Approach?**
A: Prioritize by impact. (1) Identify cost drivers: inference vs training, compute vs storage. (2) Quantization (40% savings) + caching (50% savings on cached queries) + cheaper GPU (70% hardware cost). Combine: ~60% total possible. (3) Validate: measure accuracy on each step. (4) If not enough, consider model tiering (easy queries → cheap model, hard queries → expensive model).

**Q2: Distill model, accuracy drops 2%. Is 2% acceptable?**
A: Depends. (1) If accuracy 95% → 93%, acceptable for most uses. (2) If accuracy 70% → 68%, worse. (3) Measure business impact: does 2% accuracy drop cause revenue loss? If churn increases <0.1%, cost savings outweigh. (4) Consider: 2% on test set might be 5% on production distribution.

**Q3: Cheaper GPU available ($0.50 vs $2.50/hour). Latency doubles. Worth it?**
A: Cost benefit: $2/hour savings × 730 hours = $1460/month. Latency cost: SLA 100ms → 200ms. If SLA critical, not worth. If acceptable, yes. Trade-off: is $1.5K/month worth 2x latency increase? Depends on business criticality.

**Q4: Tiered model: hard cases detected how?**
A: (1) Confidence-based: if model confidence <0.6, route to expensive model. (2) Feature-based: hard cohorts pre-identified (rare classes, new users). (3) Ensemble: cheap model output + router model decides if needs expensive model. Routers add latency but save cost on simple cases (90% of traffic).

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

## Interview Q&A
**Q: GPU inference costs $10K/month. Budget $5K. How?**
A: (1) Quantization (int8) → 40% savings ($6K). (2) Smaller model via distillation → 30% savings ($4.2K). (3) Caching repeated requests → 10% savings ($3.8K). Combined: easily hit $5K.

**Q: Cost optimization: distill model, accuracy drops 5%. Worth it?**
A: Depends on use case. If accuracy >95%, 5% drop → still >90%, acceptable. If already at 70%, unacceptable. Measure business impact (does 5% accuracy drop hurt revenue?).

## Interview Quick-Reference
| Technique | Cost Savings | Accuracy Impact |
|-----------|---|---|
| Distillation | 50% | 2-3% drop |
| Quantization | 40% | <1% drop |
| Caching | 30% | 0% |
| Batch | 50% | 0% |

## Related Topics
- [Model Serving](05-model-serving.md)
- [Request Batching](09-request-batching.md)

## Resources
- [Model Optimization Techniques](https://pytorch.org/tutorials/recipes/quantization.html)
