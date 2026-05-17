# LLM Roadmap

## Who This Is For
Engineers who want to understand large language models from the ground up — tokenization through
production deployment — and be ready for LLM engineering interviews. Assumes you've completed
Phase 1–2 of the ML Roadmap or have equivalent ML foundations.

---

## Phase 1 — Foundations (Beginner)
**Goal:** Understand how LLMs work at a conceptual and architectural level. Prompt effectively.
**Estimated time:** 2–3 weeks at 10 hrs/week

- [ ] [Tokenization](../llm/concepts/tokenization.md)
- [ ] [Pretraining](../llm/concepts/pretraining.md)
- [ ] [Prompting](../llm/concepts/prompting.md)
- [ ] [Deep Learning — Attention Mechanism](../ml/concepts/deep-learning/attention-mechanism.md)
- [ ] [Deep Learning — Transformers](../ml/concepts/deep-learning/transformers.md)
- [ ] Implement: [Prompt Engineering](../llm/implementations/prompt-engineering.ipynb)
- [ ] Practice: [LLM Theory Questions](../llm/interview-prep/llm-theory-questions.md) — Q1–Q15

**Phase 1 exit check:**
- Can you explain how BPE tokenization works?
- Can you draw the transformer architecture from memory?
- Can you explain what next-token prediction is and why it produces capable models?

---

## Phase 2 — Core Depth (Intermediate)
**Goal:** Build RAG pipelines, understand fine-tuning, evaluate LLMs. Interview-ready for most LLM roles.
**Estimated time:** 2–3 weeks at 10 hrs/week

- [ ] [Embeddings](../llm/concepts/embeddings.md)
- [ ] [RAG](../llm/concepts/rag.md)
- [ ] [Fine-tuning](../llm/concepts/finetuning.md)
- [ ] [Evaluation](../llm/concepts/evaluation.md)
- [ ] [Context Window](../llm/concepts/context-window.md)
- [ ] Implement: [Build RAG Pipeline](../llm/implementations/build-rag-pipeline.ipynb)
- [ ] Implement: [Embeddings Search](../llm/implementations/embeddings-search.ipynb)
- [ ] Implement: [LLM Evals](../llm/implementations/llm-evals.ipynb)
- [ ] Implement: [Fine-tune LLM](../llm/implementations/finetune-llm.ipynb)
- [ ] Practice: [LLM Theory Questions](../llm/interview-prep/llm-theory-questions.md) — Q16–Q40
- [ ] Practice: [Prompting Questions](../llm/interview-prep/prompting-questions.md)

**Phase 2 exit check:**
- Can you build a RAG pipeline from scratch using only the OpenAI/Anthropic API and a vector DB?
- Can you explain the difference between SFT, RLHF, and DPO?
- Can you design an eval framework for a RAG system?

---

## Phase 3 — Advanced + Production (Advanced)
**Goal:** Inference optimization, LLM system design, production observability.
**Estimated time:** 2–3 weeks at 10 hrs/week

- [ ] [Quantization](../llm/concepts/quantization.md)
- [ ] [Inference Optimization](../llm/concepts/inference-optimization.md)
- [ ] [Multimodal](../llm/concepts/multimodal.md)
- [ ] [System Design — RAG System](../llm/system-design/rag-system-design.md)
- [ ] [System Design — LLM Serving](../llm/system-design/llm-serving-design.md)
- [ ] [System Design — Fine-tuning Pipeline](../llm/system-design/fine-tuning-pipeline.md)
- [ ] [System Design — LLM Observability](../llm/system-design/llm-observability.md)
- [ ] Implement: [Structured Output](../llm/implementations/structured-output.ipynb)
- [ ] Practice: [LLM System Design Questions](../llm/interview-prep/llm-system-design-questions.md)

**Phase 3 exit check:**
- Can you explain KV cache, speculative decoding, and continuous batching?
- Can you design a low-latency LLM serving system that handles 5k concurrent requests?
- Can you explain how to monitor an LLM in production for quality drift?

---

## Interview Readiness Checklist
- [ ] Can explain transformer architecture (attention, FFN, positional encoding) without notes
- [ ] Built a working RAG pipeline end-to-end
- [ ] Fine-tuned a model using LoRA (even a tiny one)
- [ ] Completed 30+ LLM theory questions in simulation format
- [ ] Completed at least one full LLM system design mock

---

## Suggested Weekly Schedule

| Week | Focus | Files |
|------|-------|-------|
| 1 | Tokenization + pretraining + transformer arch | tokenization.md, pretraining.md, transformers.md |
| 2 | Prompting + implement prompt engineering | prompting.md, prompt-engineering.ipynb |
| 3 | Embeddings + RAG + implement RAG pipeline | embeddings.md, rag.md, build-rag-pipeline.ipynb |
| 4 | Fine-tuning + evals | finetuning.md, evaluation.md, finetune-llm.ipynb, llm-evals.ipynb |
| 5 | Context window + quantization + inference opt | context-window.md, quantization.md, inference-optimization.md |
| 6 | LLM system design | rag-system-design.md, llm-serving-design.md, system design Qs |
