# AI Legal Document Analysis & Contract Review

## TL;DR
LLM analyzes contracts/legal docs, extracts key clauses (payment terms, liability, IP), flags risks (unfair terms, missing clauses). 10K docs/month, 95% accuracy, <2 minute review time.

## Problem Statement
Legal review is expensive ($200-500/hour). Need instant first-pass analysis + flagged risks for lawyer review.

## Requirements

### Functional
- Extract key clauses
- Risk flagging
- Comparison with templates
- Compliance check
- Export summaries

### Non-Functional (Scale Targets)
- Accuracy: 95% clause extraction
- Latency: <2 minutes/doc
- Cost: <$10 per document
- Throughput: 500 docs/day

## Envelope Calculation
10K docs/month = 330 docs/day. Avg doc: 5KB = 5K tokens. LLM: 10K tokens input + 500 output = 105M tokens/month. Cost: $315 (GPT-4 @ $0.003/1K).

## High-Level Architecture
PDF → Text extraction → Section classification (Definitions, Payment, IP, etc.) → Clause extraction → Risk scoring → Comparison to template → Report generation.

## Component Breakdown
PDF parser, section classifier, clause extractor, risk scorer, template library, compliance checker.

## AI/ML Integration Points
Few-shot LLM prompting: 'Extract from this contract: [payment terms, IP ownership, liability cap]. Provide as JSON.'

## Data Flow
Upload doc → Parse → Classify sections → Extract clauses → Score risks → Flag for lawyer → Lawyer reviews + confirms.

## Key Trade-offs
Speed (basic extraction) vs depth (full analysis with comparison). Default: basic + flagged items for human review.

## Interview Q&A

**Q1: 95% accuracy goal: how do you measure accuracy?**

A: Gold standard: 100 contracts manually analyzed by lawyer. Compare LLM output to gold standard (F1 score). Target: 95% F1 on key clauses.

**Q2: Different contract types (NDA vs employment vs vendor). Different analysis needed?**

A: Yes. Classifier first identifies contract type. Load template + risk rules for that type. Example: NDA focuses on confidentiality, employment focuses on non-compete.

**Q3: Unfair terms: how do you identify risk?**

A: Rule-based: 'liability unlimited' → flag. 'termination for convenience' → flag. LLM: 'does this term seem unfair?' → risk score. Template comparison: deviation from standard.

**Q4: Cost $10/doc for 10K/month = $100K/month. Too expensive?**

A: Compare: lawyer review $300/doc × 10K = $3M/month. LLM saves 90% ($100K remaining for LLM + 10% human review). ROI: 30x.

**Q5: Outdated template library: contract law changes. How to stay current?**

A: Quarterly updates: lawyer reviews top 100 contract clauses, updates template. Monitor legal databases (LexisNexis, Westlaw) for new precedents.

**Q6: Multi-language contracts: English contract + German translation. Same analysis?**

A: Use multilingual LLM (GPT-4 handles all languages). Risk: translation errors may cause misreads. Recommend: lawyer reviews if not native language.

**Q7: Clause interaction: term A + term B together is risky, but separately fine. Detect?**

A: Advanced: pass full contract to LLM for holistic review, not just clause-by-clause. Longer context window (GPT-4 128K tokens). Cost increases 2x.

**Q8: Blind spot: what legitimate risks does LLM miss?**

A: Missing clauses are the main risk. Example: no 'force majeure' in pandemic-era contract. Template comparison catches this. Monthly red-team: test on contracts with known issues.

## Interview Quick-Reference
| Metric | Value |
|--------|-------|
| **Accuracy** | 95% clause extraction, F1 |
| **Latency** | <2 min per document |
| **Cost** | $10/document |
| **Throughput** | 330 docs/day, 10K/month |
| **Risk Scoring** | Template + rule-based + LLM |
| **Contract Types** | NDA, Employment, Vendor, M&A |

## Related Systems
- 02-enterprise-rag-document-qa.md
- 20-autonomous-db-query-agent.md
- 25-ai-observability.md
