# LLM-Powered Enterprise Semantic Search

## TL;DR
Semantic search over enterprise documents + LLM summaries. 100K docs indexed, 50K daily queries, <400ms latency, 85% relevance.

## Problem Statement
Keyword search misses semantic matches. Users need instant document retrieval + summarized answers.

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

## Interview Q&A

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
