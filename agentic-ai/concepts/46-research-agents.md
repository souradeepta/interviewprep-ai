# Research Agents
**TL;DR:** A research agent searches sources, synthesizes claims, and retains citations that let a reviewer verify each statement.

## Prerequisites
Agent loops, tool boundaries, and evaluation.

## Learning Outcomes
- Explain the main source led synthesis workflow and its state boundary.
- Compare an automated path with review or fallback using measured outcomes.
- Diagnose a representative failure using trace evidence.

## Core Intuition
Research Agents is best understood as a controlled workflow with explicit evidence and a bounded decision. The system should expose what it knows, what it is allowed to do, and when a person or deterministic checker must take over. This framing prevents a fluent answer from being confused with a verified outcome.

## How It Works
Separate source discovery from claim synthesis. Store publisher, retrieval time, URL, relevant passage, and source type; then link every material claim to supporting passages. Conflicting evidence should be represented as disagreement with dates and methods, not flattened into a single confident sentence. Search rank is not source quality, and generated citations must be checked for existence and support.
## Detailed Explanation

Research agents synthesize information: literature review, hypothesis generation, experiment design, result analysis. Core strengths: tireless search, pattern finding across papers, novelty detection. Challenges: hallucination (cite non-existent papers), opinion (mix of fact and interpretation), bias (selection bias in papers). Mechanisms: (1) literature search (semantic), (2) document understanding, (3) synthesis, (4) hypothesis generation. Best for: literature reviews, identifying research gaps, organizing findings, initial hypothesis generation (human refines).


## Core Intuition

Research assistant who reads hundreds of papers, finds patterns, suggests research directions. Speeds up literature review by 10x.


## Interview Q&A

**Q: How do you evaluate a research agent's output quality and identify hallucinations?**
A: Verify citations: follow each cited source and confirm the claim is actually supported. Check claim specificity: exact statistics (e.g., "34.7%") should be directly traceable to a source. Test with known ground truth: give the agent questions with known authoritative answers and measure accuracy. Use a second-pass verification agent: have a separate agent attempt to find counter-evidence for each claim. Track hallucination rate by domain—agents are more reliable for well-covered topics than niche or recent ones.

**Q: How do you design a research agent that handles conflicting sources?**
A: Detect conflict explicitly: compare claims across sources and flag contradictions. Present all perspectives when sources conflict: "Source A claims X, Source B claims Y. The disagreement may be due to [methodology difference/time period/definition variation]." Don't silently prefer one source—explain the basis for any prioritization (peer-reviewed > preprint, recent > older, primary > secondary). For empirical claims, prefer the source with the most rigorous methodology.

**Q: What is the appropriate scope for a research agent query and how do you prevent scope creep?**
A: Define scope at query time: topic, depth (surface overview vs. comprehensive), time range, geographic scope, source types. Implement token budgets to prevent runaway searches. Use a relevance filter: retrieved documents must exceed a minimum similarity threshold to the original query before being included. Review intermediate results with the user for multi-step research tasks rather than completing a 50-step research plan before showing results. Scope creep wastes compute and produces unfocused output.

**Q: How do you handle paywalled or inaccessible sources in research agent design?**
A: Index what you have access to—don't attempt to access paywalled content without subscription. Track which sources are available vs. not. When a highly relevant source is paywalled, note its existence and abstract (often public) but flag that full access is unavailable. For critical research, integrate with institutional library subscriptions or use APIs from publishers (Elsevier, Springer). Open-access repositories (arXiv, PubMed Central) are good primary sources for scientific research without access barriers.

**Q: What temporal limitations affect research agents and how do you communicate them?**
A: LLM parametric knowledge has a training cutoff—the agent may not know about recent developments. RAG mitigates this but depends on index freshness. Communicate: clearly indicate the publication date of all cited sources and the agent's knowledge cutoff. Recommend human review for fast-moving topics (current events, recent clinical trials, new regulations). Implement automatic dating of outputs: "This analysis is based on sources available as of [date]." For topics where recency matters, prioritize freshness in retrieval ranking.

**Q: How do you design a research agent for systematic review vs. exploratory research?**
A: Systematic review: predefined search protocol, exhaustive coverage of relevant literature, standardized quality assessment, explicit inclusion/exclusion criteria, documented methodology for reproducibility. Exploratory research: broader queries, iterative refinement based on findings, synthesis of emerging themes rather than exhaustive coverage, faster iteration. Design separate workflows: systematic review requires more structured search strategies (Boolean queries, database-specific syntax) and quality checklists; exploratory research benefits from semantic search and LLM synthesis.
Measure claim-level citation support, coverage of required subquestions, source diversity, freshness, and unsupported assertions. Human reviewers should sample claims rather than judging only fluency or citation count.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Deterministic baseline | Predictable behavior and low cost | Limited coverage of ambiguous cases |
| Model-assisted path | Handles flexible inputs | Requires evidence checks and evaluation |
| Human review | Resolves high-impact uncertainty | Capacity and response-time limits |
| Fallback or abstention | Bounds unsafe behavior | May defer useful work |

## Implementation
This runnable Python example shows a bounded policy gate relevant to the workflow. It models control logic only; it does not claim production quality.

```python
def claim_record(text, sources):
    if not sources:
        raise ValueError('claim needs source evidence')
    return {'claim': text, 'sources': tuple(sources)}

assert claim_record('The study reports an effect', ['paper-1'])['sources'] == ('paper-1',)
```

## Failure Modes
- Research Agents uses stale or incomplete input: log source and timestamp, validate freshness, and request a refresh before acting.
- Research Agents produces an unsupported result: require evidence and a domain-specific validator; route unresolved cases to review.
- Research Agents exceeds its resource or authority boundary: enforce explicit budgets and permissions in the controller, then report the stopped state.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| How do you evaluate a research agent's output quality and identify hallucinations? | Verify citations: follow each cited source and confirm the claim is actually supported. Check claim specificity: exact statistics (e.g., "34.7%") should be directly traceable to a source. Test with known ground truth: give the agent questions with known authoritative answers and measure accuracy. Use a second-pass verification agent: have a separate agent attempt to find counter-evidence for each claim. Track hallucination rate by domain—agents are more reliable for well-covered topics than niche or recent ones. |
| How do you design a research agent that handles conflicting sources? | Detect conflict explicitly: compare claims across sources and flag contradictions. Present all perspectives when sources conflict: "Source A claims X, Source B claims Y. The disagreement may be due to [methodology difference/time period/definition variation]." Don't silently prefer one source—explain the basis for any prioritization (peer-reviewed > preprint, recent > older, primary > secondary). For empirical claims, prefer the source with the most rigorous methodology. |
| What is the appropriate scope for a research agent query and how do you prevent scope creep? | Define scope at query time: topic, depth (surface overview vs. comprehensive), time range, geographic scope, source types. Implement token budgets to prevent runaway searches. Use a relevance filter: retrieved documents must exceed a minimum similarity threshold to the original query before being included. Review intermediate results with the user for multi-step research tasks rather than completing a 50-step research plan before showing results. Scope creep wastes compute and produces unfocused output. |

## Related Concepts
See [agent testing](28-agent-testing.md), [agent evaluation](29-agent-evals.md), and [agent monitoring](30-agent-monitoring.md).

## References
- [www.nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)
- [opentelemetry.io](https://opentelemetry.io/docs/)

## Further Topic Detail
## Detailed Explanation

Research agents synthesize information: literature review, hypothesis generation, experiment design, result analysis. Core strengths: tireless search, pattern finding across papers, novelty detection. Challenges: hallucination (cite non-existent papers), opinion (mix of fact and interpretation), bias (selection bias in papers). Mechanisms: (1) literature search (semantic), (2) document understanding, (3) synthesis, (4) hypothesis generation. Best for: literature reviews, identifying research gaps, organizing findings, initial hypothesis generation (human refines).


## Core Intuition

Research assistant who reads hundreds of papers, finds patterns, suggests research directions. Speeds up literature review by 10x.


## Interview Q&A

**Q: How do you evaluate a research agent's output quality and identify hallucinations?**
A: Verify citations: follow each cited source and confirm the claim is actually supported. Check claim specificity: exact statistics (e.g., "34.7%") should be directly traceable to a source. Test with known ground truth: give the agent questions with known authoritative answers and measure accuracy. Use a second-pass verification agent: have a separate agent attempt to find counter-evidence for each claim. Track hallucination rate by domain—agents are more reliable for well-covered topics than niche or recent ones.

**Q: How do you design a research agent that handles conflicting sources?**
A: Detect conflict explicitly: compare claims across sources and flag contradictions. Present all perspectives when sources conflict: "Source A claims X, Source B claims Y. The disagreement may be due to [methodology difference/time period/definition variation]." Don't silently prefer one source—explain the basis for any prioritization (peer-reviewed > preprint, recent > older, primary > secondary). For empirical claims, prefer the source with the most rigorous methodology.

**Q: What is the appropriate scope for a research agent query and how do you prevent scope creep?**
A: Define scope at query time: topic, depth (surface overview vs. comprehensive), time range, geographic scope, source types. Implement token budgets to prevent runaway searches. Use a relevance filter: retrieved documents must exceed a minimum similarity threshold to the original query before being included. Review intermediate results with the user for multi-step research tasks rather than completing a 50-step research plan before showing results. Scope creep wastes compute and produces unfocused output.

**Q: How do you handle paywalled or inaccessible sources in research agent design?**
A: Index what you have access to—don't attempt to access paywalled content without subscription. Track which sources are available vs. not. When a highly relevant source is paywalled, note its existence and abstract (often public) but flag that full access is unavailable. For critical research, integrate with institutional library subscriptions or use APIs from publishers (Elsevier, Springer). Open-access repositories (arXiv, PubMed Central) are good primary sources for scientific research without access barriers.

**Q: What temporal limitations affect research agents and how do you communicate them?**
A: LLM parametric knowledge has a training cutoff—the agent may not know about recent developments. RAG mitigates this but depends on index freshness. Communicate: clearly indicate the publication date of all cited sources and the agent's knowledge cutoff. Recommend human review for fast-moving topics (current events, recent clinical trials, new regulations). Implement automatic dating of outputs: "This analysis is based on sources available as of [date]." For topics where recency matters, prioritize freshness in retrieval ranking.

**Q: How do you design a research agent for systematic review vs. exploratory research?**
A: Systematic review: predefined search protocol, exhaustive coverage of relevant literature, standardized quality assessment, explicit inclusion/exclusion criteria, documented methodology for reproducibility. Exploratory research: broader queries, iterative refinement based on findings, synthesis of emerging themes rather than exhaustive coverage, faster iteration. Design separate workflows: systematic review requires more structured search strategies (Boolean queries, database-specific syntax) and quality checklists; exploratory research benefits from semantic search and LLM synthesis.
