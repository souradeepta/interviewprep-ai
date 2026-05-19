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
