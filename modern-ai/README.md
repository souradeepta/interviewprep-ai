# Modern AI Engineering

Contemporary techniques and patterns that define how AI systems are built, optimized, and operated in 2024. This section covers 25 practical innovations (mostly 2023-2024) that bridge research models to production systems, including Anthropic's Model Context Protocol (MCP) ecosystem.

## Why This Section

The gap between "I trained a model" and "I deployed a production system" is huge. These 25 concepts represent the tools, techniques, and patterns that modern AI engineers use every day:
- How to test and evaluate LLMs systematically
- How to build persistent memory for agents
- How to optimize inference for cost and latency
- How to train models efficiently with the latest techniques
- How to make LLMs reason better at test time
- How to run LLM operations at scale

Every concept here is practical: each has sample implementations, real code, and production patterns.

## Contents

| Folder | What's Inside |
|--------|--------------|
| [concepts/](concepts/) | 25 concept files with detailed explanations, architecture trade-offs, interview Q&A, best practices |
| [notebooks/](notebooks/) | 25 Jupyter notebooks with working implementations (basic → advanced → real-world examples) |

## 25 Concepts by Category

### Evaluation & Testing (01-03)
How do you know if your LLM is actually good? These concepts cover systematic evaluation and adversarial testing.

- **01 — [LLM Evaluation Harness](concepts/01-llm-evaluation-harness.md)** — Frameworks like lm-eval-harness for systematic benchmarking
- **02 — [AI Red-Teaming](concepts/02-ai-red-teaming.md)** — Finding vulnerabilities: adversarial attacks, prompt injection defense, jailbreaks
- **03 — [Agentic Testing Harness](concepts/03-agentic-testing-harness.md)** — Testing agent workflows end-to-end: action validation, tool calling, safety

### Memory & Context (04-06)
Modern AI systems need to remember. These concepts cover memory architectures and how to efficiently work with large contexts.

- **04 — [Persistent AI Memory](concepts/04-persistent-ai-memory.md)** — Long-term memory types: episodic, semantic, procedural; implementations with Chromadb, Mem0
- **05 — [Advanced RAG Patterns](concepts/05-advanced-rag-patterns.md)** — Beyond basic RAG: HyDE, multi-hop retrieval, re-ranking, GraphRAG, iterative retrieval
- **06 — [Context Distillation](concepts/06-context-distillation.md)** — Compressing long contexts while preserving information for LLMs

### Training & Fine-tuning (07-10)
These techniques unlock new capabilities: training on better data, using AI for feedback, and combining multiple approaches.

- **07 — [Synthetic Data Generation](concepts/07-synthetic-data-generation.md)** — Self-instruct, Magpie, GPT-4 generated data; quality filtering and validation
- **08 — [Constitutional AI & RLAIF](concepts/08-constitutional-ai-rlaif.md)** — Using AI feedback instead of human feedback; constitutional principles
- **09 — [RAFT — Retrieval-Augmented Fine-tuning](concepts/09-raft-retrieval-augmented-finetuning.md)** — Combining RAG + fine-tuning: retrieve, then train on top answers
- **10 — [Model Merging](concepts/10-model-merging.md)** — Combining fine-tuned models: SLERP, DARE, Task Arithmetic, OLE

### Inference Optimization (11-14)
Production inference must be fast and cheap. These concepts are about making models run orders of magnitude faster.

- **11 — [Flash Attention](concepts/11-flash-attention.md)** — Memory-efficient attention kernels for long sequences; I/O aware algorithms
- **12 — [Test-Time Compute Scaling](concepts/12-test-time-compute-scaling.md)** — o1-style reasoning: allocate compute at inference time, not training time
- **13 — [LLM Serving Frameworks](concepts/13-llm-serving-frameworks.md)** — Production deployments: vLLM, TGI, SGLang, PagedAttention, batching
- **14 — [Structured Generation](concepts/14-structured-generation.md)** — Constrained outputs: grammar-guided, JSON-forced, regex-guided, LMQL

### Reasoning & Self-Improvement (15-17)
Models that can think better and improve themselves in production.

- **15 — [Reflexion & Self-Critique](concepts/15-reflexion-self-critique.md)** — SELF-RAG, reflexion loops, critique-then-revise, iterative refinement
- **16 — [Chain-of-Draft](concepts/16-chain-of-draft.md)** — Scratchpad reasoning, draft-then-refine, intermediate checkpoints
- **17 — [Continual Learning for LLMs](concepts/17-continual-learning-llms.md)** — Training on new data without forgetting; elastic weight consolidation, replay buffers

### Operations & Infrastructure (18-20)
Running LLMs in production means managing versions, costs, and models.

- **18 — [LLMOps](concepts/18-llmops.md)** — Full LLM lifecycle: versioning, experiment tracking, monitoring, rollback, A/B testing
- **19 — [AI Gateway & Routing](concepts/19-ai-gateway-routing.md)** — Multi-LLM orchestration: route by cost/latency/capability, fallbacks, failover
- **20 — [Multimodal Fine-tuning](concepts/20-multimodal-finetuning.md)** — CLIP-style, LLaVA-style: aligning vision and language, PEFT for multimodal

### MCP & Agentic SDK (21-25) — Latest 2024 Innovations
Anthropic's Model Context Protocol and modern agent frameworks are reshaping how AI systems interact with external resources.

- **21 — [MCP (Model Context Protocol)](concepts/21-mcp-model-context-protocol.md)** — Anthropic's standard protocol for connecting LLMs to resources, tools, and services in a unified way
- **22 — [MCP Memory](concepts/22-mcp-memory.md)** — Memory management and persistence via MCP for stateful agents and conversational systems
- **23 — [RAG Memory Store](concepts/23-rag-memory-store.md)** — Vector databases and retrieval systems for agent knowledge and memory integration
- **24 — [MCP Tunnel](concepts/24-mcp-tunnel.md)** — Networking and tunneling for secure MCP connections across boundaries and deployments
- **25 — [Agentic SDK](concepts/25-agentic-sdk.md)** — Modern frameworks and SDKs for building production-grade AI agents with reliability and observability

---

## How to Use This Section

**Pick a goal:**
- Building an evaluation pipeline? → Start with 01-03
- Adding memory to an agent? → Start with 04-06
- Optimizing training? → Start with 07-10
- Making inference faster? → Start with 11-14
- Making models reason better? → Start with 15-17
- Deploying to production? → Start with 18-20

**For each concept:**
1. Read the markdown file for theory, patterns, and best practices
2. Follow the notebook for hands-on implementation
3. Look at the interview Q&A to understand what hiring managers ask
4. Check the common pitfalls section to avoid real mistakes

---

## Key Patterns (from CLAUDE.md)

Every concept here follows these principles:

✅ **Code Over Theory:** Notebooks are 70% code, 30% explanation
✅ **Real Libraries:** All code uses actual imports (transformers, torch, peft, vllm, etc.) — no pseudo-code
✅ **Production Patterns:** Includes device management, batching, error handling, memory optimization
✅ **3-Level Progression:** Basic → Advanced → Real-World Examples
✅ **Interview Ready:** Q&A focuses on "when/why/how to debug" not memorization

---

## Related Sections

- **[AI Fundamentals](../ai/)** — Core ML theory these concepts build on
- **[LLM](../llm/)** — Deeper dive into transformer models and training
- **[Agentic AI](../agentic-ai/)** — Building multi-step AI systems
- **[MLOps](../mlops/)** — General ML lifecycle (Modern AI is LLM-specific)
- **[System Design](../system-design/)** — Patterns for building scalable AI systems

---

## Start Here

Recommend starting with [01-llm-evaluation-harness](concepts/01-llm-evaluation-harness.md) if you're new to modern AI engineering. Then pick your path based on what you want to build.
