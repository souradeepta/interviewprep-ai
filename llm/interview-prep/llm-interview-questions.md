# LLM Interview Questions

Each prompt is a 5–10 minute LLM engineering scenario. State assumptions,
constraints, implementation outline, evaluation plan, failure modes, and a
follow-up before reading the linked remediation page.

## Architecture and training

1. Design a next-token training data pipeline with deduplication and a held-out set.
2. Explain how tokenizer choice changes cost, context capacity, and multilingual quality.
3. Diagnose training loss falling while held-out loss rises.
4. Choose a model size and data mixture under a fixed compute budget.
5. Compare pretraining from scratch, continued pretraining, and instruction tuning.
6. Design a distributed training recovery plan after a worker failure.
7. Explain data contamination and construct a contamination-resistant evaluation split.
8. Diagnose catastrophic forgetting after domain adaptation.
9. Choose LoRA, full fine-tuning, or adapters for a small domain dataset.
10. Design a safe instruction-tuning dataset and reviewer agreement process.

## Retrieval and RAG

11. A RAG system retrieves irrelevant chunks: separate query, chunking, index, and ranker causes.
12. Choose chunk size, overlap, metadata filters, and parent-document reconstruction.
13. Evaluate embeddings and a reranker when recall@K improves but answer quality does not.
14. Design a hybrid lexical/vector retrieval system for rare identifiers.
15. Decide between fine-tuning and RAG for frequently changing policy documents.
16. Make citations faithful when the model combines multiple retrieved sources.
17. Set an offline-to-online RAG evaluation plan with attribution, latency, and cost.

## Alignment, safety, and structured output

18. Compare SFT, preference optimization, RLHF, and DPO for a support assistant.
19. Explain LLM-as-judge bias and design human calibration and judge-agreement checks.
20. Make JSON output reliable when schemas evolve and malformed output is costly.
21. Defend against prompt injection in retrieved documents and tool results.
22. Design refusal and safe-completion evaluation without rewarding over-refusal.
23. Turn production failures into a versioned regression set and release gate.

## Inference and operations

24. Route requests across models under quality, latency, and token-cost budgets.
25. Diagnose a latency regression by separating queueing, prefill, decode, and network time.
26. Explain KV caching, continuous batching, and when speculative decoding helps.
27. Choose quantization levels while protecting rare-token and long-context quality.
28. Design rate limits, retries, timeouts, and fallbacks for a model-serving tier.
29. Respond to a quality incident with no code deploy: scope, rollback, and root cause.
30. Design observability for prompt versions, retrieval traces, token use, safety, and user outcomes.

Evaluation follow-up: connect Q13, Q17, Q19, Q23, and Q30 to the
[cross-domain evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

## Interviewer follow-ups

For any prompt, ask what the candidate would measure, what would invalidate the
plan, how quality changes by slice, and how they would bound cost and latency.
Required scenarios include retrieval failure, chunking, reranking, judge
limitations, fine-tuning versus RAG, structured output, injection, routing, and
incident response.
