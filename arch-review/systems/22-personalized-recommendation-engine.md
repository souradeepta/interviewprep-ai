# Personalized Recommendation Engine (ML + LLM Re-ranking)

## TL;DR
Two-tower retrieval + LLM re-ranking for personalized product recommendations. 1B recommendations/day, 15% CTR lift.

## Problem Statement
Generic recommendations underperform. Need personalized ranking.

## Requirements

### Functional
- User embedding
- Item embedding
- Retrieval
- LLM re-ranking

### Non-Functional (Scale Targets)
- Scale: 1B recs/day
- CTR lift: 15%
- Latency: <200ms

## Envelope Calculation
1B recs × $0.00001 = $10K/month.

## Architecture Overview
[Detailed architecture diagram with Mermaid showing component flow]

## Component Breakdown
- Core components and their responsibilities
- Latency and cost breakdown per component

## AI/ML Integration Points
- Where LLM/ML models are used
- Model selection and routing logic
- Cost optimization strategies

## Key Trade-offs

| Approach | CTR Lift | Latency | Cost/Rec | Personalization | Diversity |
|----------|----------|---------|----------|-----------------|-----------|
| Popularity-based | 0% | 1ms | $0 | None | Low |
| Collaborative filtering | 10% | 50ms | $0.00001 | Medium | Medium |
| Two-tower embedding | 13% | 150ms | $0.00005 | High | Medium |
| Two-tower + LLM rank | 15% | 300ms | $0.0005 | Very high | High |
| Multi-model ensemble | 16% | 500ms | $0.001 | Highest | Highest |

**Decision:** Cost-critical → collaborative filtering. CTR optimization → two-tower. Diversity/quality → LLM rerank.

---

## Production Failure Scenarios

**Scenario 1: Cold-start (new users with no history)**
- New users get generic recs (embedding unfilled). CTR drops 50%.
- Fix: Fallback to popularity-based. Hybrid recommendations (content + collaborative).

**Scenario 2: LLM reranking too expensive**
- LLM reranking costs $0.0005/rec, budget $0.00001/rec. System over budget.
- Fix: Selective LLM (only top-5 candidates, not all 1000). Or: cheaper model.

**Scenario 3: Embedding staleness**
- User behavior changes (interests shift). User embeddings stale (updated weekly).
- Fix: Real-time embedding update. Or: short-term decay (recent interactions weighted higher).

**Scenario 4: Filter bubble (diversity loss)**
- ML model optimizes CTR. Users see only similar items. Discovery low.
- Fix: Diversity constraint. Enforce >30% new/diverse items. Monitor serendipity.

---

## Implementation Guidance

**Wrong:** Optimize CTR/engagement alone. Ignore long-term user satisfaction.
**Right:** Balance CTR + diversity + novelty. Monitor user churn, not just CTR.

**Wrong:** Use expensive LLM for all reranking.
**Right:** Tiered: ML for ranking, LLM only for final rerank if needed.

---

## Sophisticated Interview Q&A

**Q1: How do you scale this system from current to 10x volume?**

A: Identify bottleneck (usually inference or storage). Auto-scaling: add GPUs for model serving, replicate databases, implement caching at retrieval layer. Example: for 10x compute, scale from 8 A100s to 80 A100s with load balancing.

**Q2: What's the cost optimization strategy as volume grows?**

A: Batch processing where possible (saves 50%), model distillation (cheaper inference), caching (reduce LLM calls), negotiate volume discounts with cloud providers. Target: cost per request drops 30-50% at 10x scale.

**Q3: How do you handle model failures or hallucinations?**

A: Confidence thresholds (only auto-act if confidence >0.95), human review queue for uncertain cases, validation checks (does output make sense?), continuous monitoring with alerts if error rate increases.

**Q4: What metrics do you track for system health?**

A: Latency (P50, P99), error rate, cost per request, model accuracy, throughput, user satisfaction. Dashboard updated real-time. Alert if latency >2x SLA or accuracy drops >5%.

**Q5: Privacy and compliance: how do you protect user data?**

A: Data minimization (keep only necessary data), encryption in transit + at rest, RBAC for access, audit logs. For regulated domains (medical, financial), additional: data residency, compliance certifications, annual penetration testing.

**Q6: Multi-region deployment: latency vs cost trade-off?**

A: Deploy in 3-5 regions, route user to closest region (100ms latency savings). Cost: ~3x infrastructure. Benefit: global coverage + disaster recovery. For most systems, worth it.

**Q7: Monitoring model drift: how do you detect performance degradation?**

A: Continuous evaluation on production data (10% sample). Weekly accuracy report. If accuracy drops >2%, alert and investigate (data drift, model bug, or expected variation). Retrain if needed.

**Q8: Cost target vs reality: if you're 2x over budget, what do you do?**

A: (1) Cheaper model (GPT-3.5 vs GPT-4): 10x cost reduction, 15% accuracy drop. (2) Caching (save 30%). (3) More selective LLM usage (only for hard cases). (4) Volume discounts. Target: get to 1.1-1.2x budget.

## Interview Quick-Reference

| Metric | Target |
|--------|--------|
| **Scale** | [Users/requests/day] |
| **Latency P99** | [<X ms] |
| **Accuracy** | [Y%] |
| **Cost** | [$Z per request] |
| **Availability** | [99.9%+] |

## Related Systems
- [Related system 1]
- [Related system 2]
- [Related system 3]
