# Project 02: RAG Evaluation and Retrieval

**Timebox:** 8 hours. **Role path:** LLM/RAG engineer. **Prerequisites:** [RAG](../llm/concepts/18-rag.md), [semantic search](../llm/concepts/21-semantic-search.md), [retrieval systems](../llm/concepts/40-retrieval-systems.md), [LLM evaluation](../llm/concepts/32-evaluation.md).

## Problem brief

Build a local retrieval-and-answerability baseline over a small synthetic policy corpus. The system should return evidence IDs and abstain when the corpus does not support the query. Compare lexical retrieval with a transparent token-overlap baseline; do not claim that retrieved text proves a generated answer.

## Data contract

Supply 12–20 fictional policy passages and at least 10 queries with expected relevant passage IDs, including answerable, unanswerable, paraphrased, conflicting, and stale-policy cases. Keep all data in the repository/project folder, label document version and effective date, and use no API or external service.

## Deliverables

1. A deterministic local index and retrieval command.
2. Evaluation output for recall@k, reciprocal rank or another justified ranking metric, abstention behavior, and at least one failure slice.
3. A short report separating retrieval correctness from answer faithfulness and listing unsupported claims that the system must not make.
4. A five-minute interview debrief on chunking, evidence, and the evaluation set’s limits.

## Local success criterion

`python3 solution.py` exits zero, retrieves the known passage in at least 8/10 answerable fixed queries at k=3, abstains on every explicitly unanswerable query, and prints per-query evidence IDs plus aggregate metrics. The threshold is a project gate for this synthetic set, not a production quality claim.

## Evaluation rubric — 100 points

| Dimension | Points | Evidence |
|---|---:|---|
| Engineering and reproducibility | 20 | Local deterministic corpus/index, versioned passages, stable output. |
| Measurement and retrieval quality | 30 | Correct metric denominators, baseline, query slices, clear retrieval/answer distinction. |
| Failure handling | 20 | Unanswerable, conflicting, stale, empty, and paraphrased query cases. |
| Communication | 15 | Evidence-linked report and calibrated limitations. |
| Interview debrief | 15 | Defends k/chunking choice and proposes the next evaluation experiment. |

## Extension prompt

Add a simple reranker and measure whether it improves reciprocal rank without reducing answerable-query recall or causing unanswerable queries to appear supported.
