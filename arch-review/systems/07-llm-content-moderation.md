# Content Moderation Platform with LLM

## TL;DR
LLM-powered moderation for 100M daily user posts (text, images, video). Classifies: hate speech, violence, spam, adult content. 99.9% uptime, <500ms latency per item, <1% false positive rate.

## Problem Statement
Platform receives 100M posts/day. Manual moderation impossible. Need instant automated decisions + human appeal queue.

## Requirements

### Functional
- Multi-modal: text, image, video
- Classify 20+ violation types
- Confidence scoring
- Appeal workflow
- Custom org policies

### Non-Functional (Scale Targets)
- Throughput: 1.2M items/hour (100M/day)
- Latency: <500ms
- False positive: <1%
- Availability: 99.9%

## Envelope Calculation
100M posts/day → 1.2M items/hour → 333 QPS. Peak: 500 QPS (evening traffic). Cost: 100M × $0.001 (LLM cost) = $100K/month.

## High-Level Architecture
Input → Fast Path (regex patterns for obvious violations) → If uncertain, LLM Classification → Confidence Threshold → Auto-Decision or Human Queue → User Appeal.

## Component Breakdown
Pattern matcher (100+ regex rules), LLM classifier (GPT-4 for complex cases), vision model (CLIP for image), confidence scorer, appeal queue, analytics dashboard.

## AI/ML Integration Points
Ensemble: patterns + vision model + text LLM. Pick majority vote or use confidence weighting.

## Data Flow
User post → Extract text/images/video → Classify → Score confidence → Queue if uncertain → Human review → Action (remove/label/keep).

## Key Trade-offs
Speed vs accuracy: fast regex (500ms, 85% recall) + slow LLM (1s, 95% recall). Hybrid: regex first, LLM for uncertain cases.

## Interview Q&A

**Q1: Cost target $100K/month for 100M items. How to achieve?**

A: $100K/month = $0.03/item on infra. LLM cost: $0.001. Infra cost: $0.029. Use fast regex for 80% (no LLM), LLM for 20% (hard cases). Total: $0.0004/item.

**Q2: Appeal rate: 5% of decisions appealed. Human review cost?**

A: 5M appeals/month × $0.10 (human labor) = $500K cost. Better: ML oracle (disagreement) goes to harder human review. Only 1% make it past ML, costing $50K.

**Q3: False positive catastrophe: innocent post labeled as hate speech. How to minimize?**

A: Confidence threshold: only auto-remove if >0.95 confidence. Below threshold, queue for human. Also: user appeal immediate review. Target: <1 false positive per 10K decisions.

**Q4: Image moderation: what if user uploads blurred/obfuscated NSFW?**

A: CLIP model detects context even blurred. Semantic understanding catches intent. Fallback: if uncertain, request user clarification or queue for human review.

**Q5: Cultural context: what's hate speech in one culture may be normal in another. Handle?**

A: Org policy override: allow per-region/org configuration of what's banned. Example: politics banned in some orgs, allowed in others. LLM prompt includes context.

**Q6: Video moderation: 100K videos/day, avg 10 min each. Feasible?**

A: Sample frames (1/sec = 600 frames per video). Classify sample + audio transcript. Cost: 100K × 600 frames × $0.001 = $60K/month. Faster: sample 10 frames/video → $6K.

**Q7: Adversarial examples: users try to evade moderation (leetspeak h4t3). Detect?**

A: Normalize input: spell-correct, decode obfuscation. Use character-level encoding (not word-level). LLM handles semantics. Test with red-team adversarial examples monthly.

**Q8: SLA: 99.9% uptime on moderation system. Cost of downtime?**

A: 100M posts/day → 115K posts/hour during downtime. If 1% violate policy, 1.15K bad posts go live per hour of downtime. Brand damage + legal risk high. Worth multi-region redundancy.

## Interview Quick-Reference
| Metric | Value |
|--------|-------|
| **Throughput** | 100M posts/day, 333 QPS avg |
| **Latency** | <500ms per item |
| **False Positive** | <1% |
| **Cost** | $100K/month |
| **Confidence Threshold** | >0.95 for auto-remove |
| **Appeal Rate** | 5% of decisions |

## Related Systems
- 01-llm-customer-service.md
- 25-ai-observability.md
- 14-autonomous-data-analysis-agent.md
