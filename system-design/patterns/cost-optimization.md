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

## Interview Quick-Reference
**Cost optimization?** Model size, serving efficiency, right-size infrastructure, data reduction.

## Related Topics
- [Model Compression](../ml/concepts/model-compression.md)
- [Quantization](../llm/concepts/quantization.md)

## Resources
- [Cost-Effective ML](https://www.oreilly.com/library/view/cost-effective-machine-learning/9781491990797/)
