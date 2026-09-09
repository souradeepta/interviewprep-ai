# Retrieval Failure Remediation

Use this page after **Q11: “A RAG system retrieves irrelevant chunks.”** The
goal is to localize the failure before changing the embedding model or prompt.

## Contract and evidence

Define a query-level evaluation record before debugging:

| Field | Meaning |
|---|---|
| `query_id` | Stable identifier for the user question and corpus snapshot |
| `relevant_doc_ids` | Human- or label-derived set, with assessor agreement recorded |
| `retrieved_doc_ids` | Ordered IDs returned at each pipeline stage |
| `answer_support` | Whether the final answer is entailed by the cited evidence |
| `latency_ms` / `cost` | Retrieval, reranking, generation, and total budgets |

Do not use answer quality alone to diagnose retrieval: a strong generator can
hide a miss, and a weak generator can fail with perfect evidence.

## Triage sequence

1. **Reproduce against a frozen snapshot.** Pin query normalization, corpus
   version, chunker configuration, embedding model, filters, and random seeds.
2. **Check labels and identity.** Confirm the relevant document is present,
   tenant-visible, not expired, and mapped to the same canonical ID at index
   and retrieval time.
3. **Measure candidate recall.** Run lexical BM25, dense retrieval, and hybrid
   retrieval at a generous candidate `K`. If recall is low for all methods,
   inspect corpus coverage and query intent rather than reranking.
4. **Inspect chunk boundaries.** Record neighboring chunks, section titles,
   table rows, and parent-document IDs. A relevant document split across
   unrelated chunks is a chunking failure.
5. **Measure ranker movement.** Compare recall before and after reranking. A
   drop means the ranker, feature scaling, or truncation policy is suspect.
6. **Check context assembly.** Verify deduplication, token-budget truncation,
   source ordering, and metadata filters. Retrieved evidence discarded before
   generation is a context failure, not a retriever failure.

## Diagnostic matrix

| Observation | Likely boundary | Next test |
|---|---|---|
| Relevant ID absent at high candidate `K` | corpus, filters, query, or index | lexical-vs-dense recall and filter audit |
| Relevant ID present, ranked below cutoff | embedding or reranker | recall@K before/after rerank |
| Evidence present but answer unsupported | context assembly or generator | citation entailment with evidence-only prompt |
| Only rare IDs fail | lexical signal or tokenizer | hybrid retrieval and exact-match slice |
| Only long documents fail | chunking or truncation | parent reconstruction and token-budget audit |

## Metrics and release gates

Report recall@10/50, MRR or nDCG for ranking, answer support/faithfulness,
no-evidence refusal precision, p95 latency, and cost per answered query. Slice
by language, document age, query length, tenant, and identifier density. A
retriever change should not ship on aggregate recall alone: require no severe
slice regression, a fixed corpus snapshot, and a canary with rollback.

## Common wrong turns

- Increasing `K` indefinitely, which raises context cost while hiding a bad
  ranker or duplicate-heavy index.
- Treating an unjudged query as a negative example.
- Changing chunk size and embedding model simultaneously, making attribution
  impossible.
- Counting a cited document as support without checking whether it entails the
  claim.

## Related references

- [RAG concept](../concepts/18-rag.md)
- [Embeddings concept](../concepts/02-embeddings.md)
- [Evaluation exercises](../../ml/interview-prep/evaluation-experiment-questions.md)
