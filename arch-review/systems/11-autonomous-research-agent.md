# Autonomous Multi-Step Research Agent

## TL;DR
Agent autonomously researches topics: searches web, reads docs, synthesizes findings. 1K research requests/day, <30 minute turnaround, 80% answer completeness.

## Problem Statement
Researchers waste days on literature search. Need automated research agent that finds + synthesizes.

## Requirements

### Functional
- Web search
- Document parsing
- Synthesis
- Citation tracking

### Non-Functional (Scale Targets)
- Turnaround <30 min
- 80% completeness
- Cost <$5/research

## Envelope Calculation
1K researches/day × $5 = $5K/day. Search API: 5K × 10 searches × $0.001 = $50/day.

## High-Level Architecture
Request → Planning → Search → Reading → Analysis → Synthesis → Report.

## Component Breakdown
Planner, searcher, reader, analyzer, synthesizer.

## AI/ML Integration Points
Multi-step LLM reasoning + tool use (search, read).

## Data Flow
Request → Plan steps → Execute → Report.

## Key Trade-offs
Depth vs breadth: deep (20 sources, 30min) vs broad (100 sources, 60min).

## Detailed Trade-off Analysis

| Approach | Turnaround | Sources | Completeness | Cost/Research | Hallucination | Accuracy |
|----------|-----------|---------|--------------|---------------|---------------|----------|
| Keyword search | 2 min | 5 | 30% | $0.10 | 10% | 50% |
| Single-agent broad | 30 min | 50 | 70% | $3.00 | 5% | 75% |
| Single-agent deep | 60 min | 20 | 80% | $5.00 | 3% | 85% |
| Multi-agent parallel | 30 min | 100 | 85% | $8.00 | 2% | 90% |
| Human-in-loop | 4 hours | 50 | 95% | $50.00 | <1% | 98% |

**Decision:** Speed critical → single-agent broad. Quality critical → multi-agent + verification. Regulatory → human-in-loop.

---

## Production Failure Scenarios

**Scenario 1: Agent hallucinates sources**
- Agent cites "Smith et al. 2024" but source doesn't exist. User follows up, discovers false.
- Trust destroyed.
- Fix: Verify citations programmatically. Only cite if actually read. Confidence scoring.

**Scenario 2: Agent goes in circles (infinite loops)**
- Agent searches, reads, searches again (same query). Never converges. Timeout after 30 min.
- Incomplete research.
- Fix: Deduplicate searches. Track visited URLs. Early termination if progress stalls.

**Scenario 3: Cost explosion from API calls**
- Agent makes 1000 API calls per research (search, parse, LLM). Cost $10/research instead of $5.
- Project unprofitable.
- Fix: Budget-aware planning. Limit searches (max 50). Batch LLM calls. Cache results.

**Scenario 4: Synthesis misses key finding**
- Agent searches 20 sources, reads 15, synthesizes from 5. Misses critical paper.
- Research incomplete.
- Fix: Source ranking. Prioritize high-impact sources. Multi-stage synthesis (broad → deep).

---

## Implementation Guidance

**Wrong:** Let agent search indefinitely. Trust it will complete.
**Right:** Set budgets (max searches, max time, max cost). Early termination on stalled progress.

**Wrong:** Synthesize from first N sources found.
**Right:** Rank sources by relevance. Prioritize high-impact. Verify citations.

---

## Sophisticated Interview Q&A

**Q1: Turnaround <30 min but 1K/day = 8.6 sec/research. Feasible?**

A: Parallel execution: 100 agents running simultaneously. 30 min / 100 = 18 sec per agent. Tight but feasible.

**Q2: Hallucination: agent cites nonexistent sources?**

A: Strict: only cite if found + read. Verify citations before report. Filter out low-confidence sources.

## Interview Quick-Reference
| Throughput | 1K requests/day |
| Turnaround | <30 minutes |
| Completeness | 80% |
| Cost | <$5/research |

## Related Systems
- 12-multi-agent-software-dev.md
- 14-autonomous-data-analysis-agent.md
