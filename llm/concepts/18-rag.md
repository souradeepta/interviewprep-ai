# RAG (Retrieval-Augmented Generation)
TL;DR: Retrieval-Augmented Generation retrieves evidence from an external corpus and places selected passages in the model context before generation, allowing answers to use updateable or private information without retraining the model.

## Prerequisites

Understand [embeddings](02-embeddings.md), [prompting](12-prompting.md), and basic language-model generation. This lesson focuses on the end-to-end retrieval and generation design; [semantic search](21-semantic-search.md) and [vector databases](20-vector-databases.md) cover retrieval components in more detail.

## Learning Outcomes

- Trace a RAG request from document ingestion through retrieval, context construction, and answer generation.
- Choose chunking, sparse, dense, or hybrid retrieval strategies using measurable quality and latency constraints.
- Distinguish retrieval failure from generation failure with separate evaluation sets and metrics.
- Design safeguards for stale documents, access control, unsupported answers, and prompt injection in retrieved text.

## Core Intuition

RAG is an open-book answering pattern. A language model supplies language generation, while a retrieval system selects pages from a controlled collection. The answer is generated after those pages are placed in context. This lets an application update a product manual or search private policies without changing model weights.

The open-book analogy has an important limit: finding a page does not make the answer true. Retrieval can return an irrelevant, stale, incomplete, or unauthorized passage, and the generator can ignore good evidence or invent a claim that the passages do not support. RAG can reduce dependence on parametric memory when evidence is retrieved, but it does not remove hallucination by construction. The system must measure retrieval and answer grounding separately.

RAG also differs from fine-tuning. Fine-tuning changes model parameters to teach behavior or a stable task distribution; RAG changes the request context with selected external information. A production system may use both: fine-tuning for output behavior and retrieval for current facts. Retrieval is especially useful when the corpus changes frequently, contains private data, or needs citations and access filtering.

## How It Works

An ingestion job parses source files, preserves metadata, and splits content into chunks. Each chunk receives an embedding or an inverted-index representation and is stored with its source, version, permissions, section, and position. Chunk boundaries are a quality decision: splitting a definition from its qualification can produce a misleading answer, while very large chunks dilute lexical or semantic relevance and consume context.

At query time, the system may rewrite or classify the query, then retrieves candidates. Sparse methods such as BM25 are strong for exact names, identifiers, and rare terms. Dense retrieval compares learned representations and can match paraphrases. Hybrid retrieval combines both signals. A cross-encoder or other reranker can score a smaller candidate set with the query and improve ordering at additional latency. Filtering by tenant, document permissions, language, or freshness must happen before evidence is exposed to the generator.

The generator receives the user query and a bounded, labeled evidence set. A useful prompt states that the evidence is untrusted reference material, requires the answer to distinguish supported facts from uncertainty, and includes source identifiers for citations. The application should support abstention when retrieval scores are weak or no passage answers the question. “Top-k” is a tuning parameter, not a quality guarantee; more passages can add recall while also adding noise and context cost.

Evaluate RAG as two linked systems. Retrieval tests use labeled query-document pairs and report recall@k, precision or nDCG, filtered-retrieval correctness, and latency. Generation tests use questions with known evidence and score answer correctness, citation correctness, and unsupported-claim rate. An oracle-context experiment, where the generator receives known relevant passages, helps separate retrieval failure from generation failure. A fresh-document test checks whether ingestion and indexing actually expose updates.

## Trade-offs

| Design | Strength | Cost | Failure boundary |
|---|---|---|---|
| Sparse retrieval (BM25) | Exact terms, simple indexing, interpretable matches | Vocabulary mismatch can hide relevant passages | Weak when the query paraphrases the source |
| Dense retrieval | Semantic matching across wording differences | Embedding model, vector index, and domain fit matter | Can miss identifiers or retrieve semantically related but wrong text |
| Hybrid retrieval | Combines lexical and semantic signals | More storage, scoring, and tuning | Score fusion can favor the wrong signal without evaluation |
| Reranking | Improves ordering among retrieved candidates | Extra model inference and tail latency | Cannot recover evidence absent from the candidate set |
| Fine-tuning without retrieval | No retrieval hop and stable learned behavior | Retraining for updates; private facts need careful data handling | Poor fit for fast-changing or source-cited knowledge |

Choose chunk size, overlap, `k`, reranking, and hybrid weights on a labeled development set. Measure p50 and p95 latency, token count, retrieval recall, answer quality, and unsupported-claim rate together. A design that improves answer accuracy by flooding the context may fail its latency or cost objective. A design with excellent retrieval recall may still need a generator or verifier that can use several passages correctly.

## Implementation

This dependency-light example uses token sets and cosine similarity to illustrate indexing, retrieval, and evidence construction. It is a transparent baseline, not a production embedding model, vector database, or language-model call.

```python
import math
import re

documents = [
    {"id": "reset", "text": "Reset a password from Settings, then choose Security."},
    {"id": "billing", "text": "Duplicate charges are reviewed by the billing team."},
    {"id": "export", "text": "Exports are available as CSV from the Reports page."},
]

def terms(text):
    return set(re.findall(r"[a-z]+", text.lower()))

def similarity(left, right):
    overlap = len(left & right)
    denominator = math.sqrt(len(left) * len(right))
    return overlap / denominator if denominator else 0.0

def retrieve(query, docs, k=1):
    query_terms = terms(query)
    ranked = sorted(
        ((similarity(query_terms, terms(doc["text"])), doc) for doc in docs),
        key=lambda item: (-item[0], item[1]["id"]),
    )
    return [doc for score, doc in ranked[:k] if score > 0]

hits = retrieve("Where do I reset my password?", documents)
assert hits and hits[0]["id"] == "reset"
assert retrieve("unknown topic", documents) == []

context = "\n".join(f"[{doc['id']}] {doc['text']}" for doc in hits)
prompt = f"Use only this evidence:\n{context}\nQuestion: Where do I reset my password?\nAnswer:"
assert "[reset]" in prompt
```

A production implementation should replace this baseline with a tested tokenizer and retrieval index, preserve source metadata, enforce authorization filters, and evaluate the generator with citations. The example's empty result is useful: an application can abstain or ask for clarification instead of treating a zero-score retrieval as evidence.

## Failure Modes

- **The relevant passage is not retrieved.** Recall@k is low for labeled queries. Inspect chunking and query language, compare BM25 and dense retrieval, add hybrid search or query rewriting, and measure whether the missing evidence exists in the index.
- **The right passage is retrieved but the answer is unsupported.** Oracle-context tests pass while end-to-end answers add claims. Inspect retrieved passages, prompt formatting, citations, and generated claims before attributing the issue to the generator; require evidence-linked statements, use a verifier or structured extraction where appropriate, and monitor unsupported-claim rate.
- **Chunking removes necessary context.** Definitions, table headers, or policy exceptions are separated from the retrieved text. Preserve section metadata, use parent-child or neighboring-chunk expansion, and test questions that require context across boundaries.
- **Documents are stale or duplicated.** Users receive superseded policy versions. Version documents, delete or deprecate old chunks, re-index on update, and include effective dates in retrieval filters and citations.
- **Unauthorized evidence crosses a tenant boundary.** A semantically relevant chunk is returned to the wrong user. Apply access-control filters before generation and test cross-tenant queries with synthetic restricted documents.
- **Retrieved text injects instructions.** A document tells the model to ignore the application task or reveal secrets. Delimit evidence as untrusted data, use instruction hierarchy and filtering, and test adversarial documents in the retrieval and generation pipeline.
- **Too much context increases cost and confusion.** Recall improves while answer quality or p95 latency worsens. Rerank, deduplicate, cap context by token budget, and compare quality against a smaller evidence set.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What is RAG? | Retrieve external evidence, place it in the model context, and generate an answer conditioned on that evidence. |
| How do you diagnose a bad RAG answer? | Check retrieval recall and permissions first, then run an oracle-context generation test to distinguish missing evidence from poor evidence use. |
| Dense or sparse retrieval? | Sparse is strong for exact terms; dense handles semantic paraphrases. Hybrid retrieval is a candidate when both patterns occur, but it must be measured. |
| How do you handle no answer in the corpus? | Use score thresholds or a trained abstention policy, instruct the model to say the evidence is insufficient, and test unanswerable questions. |
| Design follow-up: what do you monitor after launch? | Retrieval recall proxies, citation and unsupported-claim rates, stale-document incidents, access-control failures, token cost, and p95 latency. |

## Related Concepts

- [Embeddings](02-embeddings.md) — representations commonly used for dense retrieval.
- [Semantic Search](21-semantic-search.md) — query and document matching for the retrieval stage.
- [Vector Databases](20-vector-databases.md) — storage and nearest-neighbor search for vectors.
- [Prompting](12-prompting.md) — context and output instructions for generation.
- [Context Window](25-context-window.md) — limits and trade-offs for retrieved evidence.
- [RAG practice](../interview-prep/curriculum-practice.md#rag-retrieval-augmented-generation) — matching manifest-owned assessment.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) — primary RAG paper.
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — primary dense-retrieval study.

## Real-World Examples

A product-support assistant can retrieve the current version of troubleshooting articles and answer with article identifiers. The system should filter articles by product and customer entitlement, preserve effective dates, and abstain when no current article addresses the question. Evaluation can include known-answer tickets, unanswerable tickets, recently changed articles, and prompt-injection text embedded in a document. This workflow shows why retrieval quality, authorization, citation correctness, and response quality must be measured together.
