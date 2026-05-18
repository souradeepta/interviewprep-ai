# Research Agents

## Detailed Explanation

Research agents synthesize information: literature review, hypothesis generation, experiment design, result analysis. Core strengths: tireless search, pattern finding across papers, novelty detection. Challenges: hallucination (cite non-existent papers), opinion (mix of fact and interpretation), bias (selection bias in papers). Mechanisms: (1) literature search (semantic), (2) document understanding, (3) synthesis, (4) hypothesis generation. Best for: literature reviews, identifying research gaps, organizing findings, initial hypothesis generation (human refines).

## Core Intuition

Research assistant who reads hundreds of papers, finds patterns, suggests research directions. Speeds up literature review by 10x.

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
