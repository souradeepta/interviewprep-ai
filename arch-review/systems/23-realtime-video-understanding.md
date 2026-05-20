# Real-time Video Understanding System

## TL;DR
Scene understanding + LLM narration for video. 10K videos/day, <5s processing.

## Problem Statement
Video platforms need instant scene understanding for recommendations + moderation.

## Requirements

### Functional
- Frame sampling
- Object detection
- Scene understanding
- LLM narration

### Non-Functional (Scale Targets)
- Volume: 10K videos/day
- Processing: <5s/video
- Languages: 10+

## Envelope Calculation
10K videos × 60 frames × $0.001 = $600/day.

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

| Approach | Processing Time | Accuracy | Cost/Video | Detail Level | Infrastructure |
|----------|--------|----------|-----------|----------|---------|
| Frame sampling (1fps) | <1s | 70% | $0.01 | Low | CPU |
| Dense sampling (5fps) | 3s | 85% | $0.05 | Medium | GPU |
| All frames (30fps) | 10s | 90% | $0.30 | High | GPU cluster |
| Hierarchical (5fps + detail) | 5s | 88% | $0.10 | High | GPU + CPU mix |

**Decision:** Real-time content → 5fps. Archive/batch → all frames. Speed critical → hierarchical.

---

## Production Failure Scenarios

**Scenario 1: GPU memory exceeded**
- Process all frames. Memory OOM. System crashes. 10K videos queued.
- Fix: Adaptive sampling (start 5fps, increase if needed). Memory budgeting.

**Scenario 2: LLM narration cost explosion**
- Narrate every frame (30fps). Cost $10/video instead of $0.30.
- Fix: Keyframe detection (narrate only scene changes, not every frame).

**Scenario 3: Multi-language inference lag**
- Narrate in 10 languages sequentially. 5s processing becomes 50s.
- Fix: Batch translate. Or: narrate English first, translate in background.

**Scenario 4: Quality vs latency conflict**
- User wants <5s. But 85% accuracy requires 3s processing. Need 2s for other steps.
- Fix: Streaming response (start narration at 2s, add detail over time).

---

## Implementation Guidance

**Wrong:** Process all frames (30fps). Never drop frames.
**Right:** Adaptive sampling. Start coarse (1fps), refine on demand.

**Wrong:** Narrate every frame in multiple languages.
**Right:** Narrate keyframes. Translate asynchronously.

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
