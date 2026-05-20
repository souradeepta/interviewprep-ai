# LLM-Powered Enterprise Semantic Search

## Overview
An enterprise semantic search engine indexing 100K+ documents and handling 50K+ daily queries with <400ms latency and 85% relevance. Combines dense vector retrieval with LLM-generated summaries to enable instant document discovery without keyword limitations.

## Problem Statement
Traditional keyword search fails users: (1) semantic mismatch (user asks "how to reset password" but docs say "account recovery"), (2) vocabulary gap (domain-specific terms not indexed), (3) synonyms (multiple ways to describe same concept), (4) long tail (obscure topics poorly ranked), (5) answer fragmentation (answer split across multiple docs, user must read all). Economic impact: (1) users abandon search after 3 irrelevant results, (2) helpdesk calls increase (user can't find answer), (3) employee productivity lost (searching instead of working). Scale challenge: (1) 100K documents is too large for LLM context, (2) latency must be <400ms (user experience), (3) cost must be reasonable (50K queries/day). Solution: dense embeddings for semantic matching, re-ranking for relevance, LLM summaries for quick answers.

## Requirements

### Functional
- Semantic indexing
- LLM-generated summaries
- Cross-document synthesis
- Faceted search

### Non-Functional (Scale Targets)
- Latency <400ms
- Index 100K docs
- 85% relevance
- Cost <$0.50/query

## Envelope Calculation
50K queries/day × 500 tokens LLM = 25M tokens. Cost: $75/day.

## High-Level Architecture
Query → Embed → Vector search → LLM synthesis → Summary.

## Component Breakdown
Encoder, vector DB (Pinecone), LLM, re-ranker.

## AI/ML Integration Points
Dense passage retrieval + LLM synthesis.

## Data Flow
Query → Retrieve docs → Synthesize → Return.

## Key Trade-offs
Latency vs quality: smaller model (200ms) vs larger (400ms).

## Detailed Trade-off Analysis

| Approach | Latency | Accuracy | Cost/Query | Hallucination | Maintenance |
|----------|---------|----------|-----------|---------------|------------|
| Keyword search | 50ms | 60% | $0.001 | N/A | Low |
| Semantic (small model) | 200ms | 80% | $0.10 | 5% | Medium |
| Semantic (large model) | 400ms | 85% | $0.50 | 2% | High |
| Semantic + reranking | 350ms | 88% | $0.35 | 1% | High |
| Multi-hop retrieval | 600ms | 90% | $0.60 | 0.5% | Very high |

**Decision:** Speed critical → semantic small model. Accuracy critical → semantic large + reranking. Multi-doc synthesis → multi-hop.

---

## Production Failure Scenarios

**Scenario 1: Embedding drift, relevance drops**
- Embeddings indexed with E5-small. Later switch to E5-large. Cosine similarity scores incompatible.
- Relevance drops to 70%. Users complain.
- Fix: Re-index with new model. Version embeddings. Validate before switch.

**Scenario 2: Hallucination undetected**
- LLM synthesis makes up facts. Grounding check insufficient (uses synonyms, claims valid). Users trust wrong info.
- Fix: Strict grounding (only phrases from docs). Check source attribution. Flag if confidence <0.8.

**Scenario 3: Latency SLA breach at peak**
- 100 queries/min at peak. LLM queue builds up. Response time 800ms (SLA 400ms).
- Fix: Model quantization (200ms). Request batching. Caching common queries (80% cache hit possible).

**Scenario 4: Embedding index corrupted**
- Index out-of-sync with docs. Deleted docs still retrievable. New docs not indexed.
- Fix: Rebuild index weekly. Version snapshots. Validation checks (expected doc count matches).

---

## Implementation Guidance

**Wrong:** LLM synthesis without grounding. Trust LLM output.
**Right:** Strict grounding against retrieved docs. Remove claims not in source. Show source attribution.

**Wrong:** Optimize latency without measuring accuracy impact.
**Right:** Measure accuracy-latency frontier. Choose Pareto-optimal point based on SLA.

---

## Sophisticated Interview Q&A

**Q1: Latency <400ms but LLM takes 300ms. How?**

A: Parallel: while LLM generates, format docs. Stream-first-token at 200ms. User sees response starting at 200ms, completion at 400ms.

**Q2: 100K docs: indexing cost?**

A: Batch embed 1000 docs/batch on GPU: 10 batches × $5 = $50. Annual indexing: ~$2.5K (doc updates).

**Q3: Hallucination risk: LLM invents facts not in docs?**

A: Check generated answer against retrieved docs. If claim not in docs, remove. Strict grounding reduces hallucination <2%.

**Q4: Synonyms: 'salary' vs 'compensation'. Semantic match?**

A: Yes. Embeddings capture semantic similarity. E5 embedding model shows cosine similarity 0.85 for synonyms.

## Interview Quick-Reference
| Throughput | 50K queries/day |
| Latency | <400ms |
| Accuracy | 85% relevance |
| Cost | <$0.50/query |
| Hallucination | <2% |
| Supported docs | 100K |

## Related Systems
- 02-enterprise-rag-document-qa.md
- 03-llm-api-gateway.md
