# Research Agents

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


## Best Practices

1. Cite sources always
2. Distinguish fact vs interpretation
3. Include confidence levels
4. Enable expert review
5. Version control findings
6. Track methodology
7. Disclose limitations
8. Validate claims

## Code Examples

```python
class ResearchSynthesizer:
    def search_literature(self, topic, max_papers=50):
        papers = self._semantic_search(topic, max_papers)
        return papers  # {title, abstract, relevance, url}
    
    def extract_key_findings(self, papers):
        findings = []
        for paper in papers:
            key_claims = self._extract_claims(paper)
            findings.extend(key_claims)
        return findings
    
    def identify_gaps(self, findings):
        '''What's not covered?'''
        covered = set(f['topic'] for f in findings)
        potential_gaps = self._brainstorm_related(covered)
        return potential_gaps

class ResearchAgent:
    def literature_review(self, topic):
        papers = self.search_literature(topic)
        synthesis = self.synthesize(papers)
        gaps = self.identify_gaps(synthesis)
        return {'synthesis': synthesis, 'gaps': gaps, 'sources': papers}
```

## Related Concepts

- Retrieval-Augmented Generation, Knowledge Graphs, Human Collaboration
