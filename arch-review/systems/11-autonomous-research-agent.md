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

## Interview Q&A

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
