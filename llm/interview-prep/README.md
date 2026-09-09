# LLM Interview Practice

Prompt-first practice for LLM/ML engineering rounds. For each question, spend
3–7 minutes clarifying the role, scale, data, quality target, cost, latency,
and safety constraints before answering. The [question bank](llm-interview-questions.md)
has 30 numbered scenarios; the [coding set](llm-coding-exercises.md) has six
deterministic, API-free exercises.

| Round | Practice | Reference |
|---|---|---|
| Architecture/training | Q1–Q10 | [pretraining](../concepts/03-pretraining.md), [fine-tuning](../concepts/04-finetuning.md) |
| Retrieval/RAG | Q11–Q17 | [retrieval failure remediation](retrieval-failure-remediation.md), [RAG](../concepts/18-rag.md), [embeddings](../concepts/02-embeddings.md) |
| Alignment/structured output | Q18–Q23 | [LoRA](../concepts/08-lora.md), [evaluation](../concepts/32-evaluation.md) |
| Inference/operations | Q24–Q30 | [KV cache](../concepts/23-kv-cache.md), [quantization](../concepts/30-quantization.md) |

Every prompt uses this response contract: assumptions and clarifying questions;
design/solution outline; quality, cost, latency and safety trade-offs; failure
modes and one follow-up. Company-specific claims are intentionally excluded.
