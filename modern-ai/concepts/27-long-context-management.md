# Long Context Management

**TL;DR:** Context management is a budget-allocation problem: preserve evidence useful to the query while respecting token, latency and attention limits.

## Prerequisites

[Advanced RAG patterns](05-advanced-rag-patterns.md) and basic tokenization.

## Learning Outcomes

- Compare truncation, retrieval and summarization against a fixed context budget.
- Explain why a larger context window does not guarantee effective use of every token.
- Choose a context strategy using relevance, coverage and provenance tests.

## Core Intuition

Imagine answering a question from a large binder when only a few pages can be kept open. Cutting off the end is simple but may discard the answer. Retrieving pages by relevance can preserve evidence but miss supporting context. Summarizing reduces volume but may erase a qualifier or exact number. The context strategy should therefore be evaluated against the actual task, not just how many tokens fit.

## How It Works

Let each candidate segment d_j have token cost c_j, relevance r_j to query q, and possibly diversity value. A context allocator chooses a set S with \(\sum_{j\in S}c_j\le B\), where B is the prompt budget after system instructions, user input and reserved generation tokens. Truncation follows source order. Retrieval ranks chunks by a relevance function. Summarization transforms selected content into a shorter representation and introduces a lossy channel. These methods can be composed, but each stage needs a measurable contract: recall@k, answer support, citation correctness, compression ratio, and latency.

Self-attention's dense pairwise work scales approximately quadratically with sequence length in the basic formulation, and key/value memory grows linearly with cached sequence length. Efficient attention kernels improve memory traffic, but they do not fix irrelevant context or lost-in-the-middle behavior. Position encoding extension also does not guarantee retrieval or reasoning quality at the new length.

Budgeting should begin by counting all prompt components with the tokenizer used by the serving model. A character count is not a safe token estimate: code, identifiers, non-English text and punctuation can tokenize very differently. Reserve space for the system policy, chat history, retrieved evidence, tool outputs and expected answer before admitting another document. If the answer cap is ignored, a prompt that fits at admission can still exceed the model's context limit during generation. Keep a margin for tokenizer differences and protocol framing.

Retrieval quality depends on chunk boundaries as well as ranking. Very short chunks lose definitions and references; very large chunks spend the budget on unrelated text. Overlap can preserve boundary context but causes duplicate evidence. A robust retrieval evaluation measures recall of labeled supporting spans and answer completeness, then checks source diversity and duplication. Include exact-match queries, paraphrases, rare names and questions whose answer requires joining evidence from multiple sections. Semantic similarity alone can rank a passage that discusses the same topic but does not support the claim.

Summarization has a different objective from retrieval. A summary can keep broad themes while dropping a date, negation or exception that determines the answer. For high-stakes facts, retain source citations and test each generated claim against the original passage. Hierarchical summarization introduces repeated lossy transformations; inspect whether a fact survives each level rather than validating only the final prose. Query-focused summaries can preserve relevant details better than general summaries, but depend on having the right query before compression.

Long-context evaluation should distinguish capacity from effective context use. Construct probes with the same evidence placed near the beginning, middle and end, and report accuracy by position. Add distractor passages that share vocabulary with the answer and measure whether the system cites the correct source. Test context lengths around realistic percentiles and maximums, not just one impressive long example. Track prompt tokens, prefill latency, cache bytes and output quality together so a strategy is not labeled efficient solely because it reduces one resource.

The allocator is also a policy boundary. System instructions and safety rules should not be displaced by low-confidence retrieval. A practical priority ordering can reserve mandatory policy tokens, allocate a bounded history window, select evidence by relevance and provenance, then reserve a generation allowance. When no evidence fits, return a clear insufficiency result rather than silently truncating the tail. For conversational systems, summarize old turns with speaker and temporal markers and keep the original transcript addressable when auditability matters.

## Trade-offs

| Strategy | Works well when | Failure boundary |
|---|---|---|
| Truncation | The tail is low value or order encodes priority | Relevant facts appear late |
| Retrieval | Evidence is separable and queryable | Ranking misses rare terms or cross-document dependencies |
| Summarization | Broad context must be compressed | Exact details, caveats or provenance must survive |
| Hybrid | Retrieval selects evidence and a summary organizes it | Errors compound across retrieval and compression |

Prefer a retrieval path when facts need provenance or change frequently. Prefer summarization when broad themes matter more than exact wording, after verifying that the summary retains task-critical evidence.

## Implementation with Python

Run `python3 modern-ai/implementations/27-long-context-management.py`. It compares source-order truncation, query-term retrieval and a deliberately simple extractive compression proxy on fixed sentence snippets. It is a teaching baseline, not a semantic search model.

```python
import runpy
allocate = runpy.run_path("modern-ai/implementations/27-long-context-management.py")["allocate"]
snippets = ["refunds take five days", "annual plan renews in June"]
selected = allocate(snippets, 1, "retrieve", ("renewal",))
assert "renewal" in selected[0]
```

## Failure Modes

| Failure | Diagnostic | Mitigation |
|---|---|---|
| Relevant evidence falls outside prefix | Recall of gold evidence versus token position | Retrieve by query and preserve source offsets |
| Retrieval returns duplicate chunks | Unique-source ratio and answer coverage | Add diversity constraints and deduplicate overlap |
| Summary changes a critical qualifier | Claim-level comparison to source | Keep citations and test numerals, negation and exceptions |

- Late evidence is truncated: recall falls with source position; reorder by priority or retrieve by query, then verify answer support.
- Retrieval returns duplicates: unique-source ratio and coverage stay low; diversify results and deduplicate overlapping chunks.
- Compression drops a qualifier: compare each claim with its cited source and test dates, negation and exceptions before using the summary.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| Why not always use the largest context window? | Cost, latency, memory and attention quality; ask how you would measure lost-in-middle. |
| When is summarization preferable to retrieval? | When broad evidence must be compacted and exact provenance is less central; follow up on factuality tests. |
| What belongs in the context budget? | Instructions, conversation, evidence and reserved output; ask how dynamic output reservation changes admission. |

## Related Concepts

See [KV cache optimization](29-kv-cache-optimization.md), [sparse attention](30-sparse-attention.md), [notebook 27](../notebooks/27-long-context-management.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#long-context-management).

## References

[Lost in the Middle](https://arxiv.org/abs/2307.03172) and [LongBench](https://arxiv.org/abs/2308.14508).
