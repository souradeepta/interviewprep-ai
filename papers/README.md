# Papers: Foundational & Recent AI Research

> **12 seminal papers that shaped modern AI.** Each paper includes a comprehensive explanation, code examples, and interview Q&A to help you understand AND implement the core ideas.

## What Are These Papers?

This section covers the papers that every ML/AI engineer should know:
- **Foundational classics** (Attention, ResNet, BERT) that established core concepts
- **Recent breakthroughs** (ReAct, Tree of Thoughts) that changed how we build AI systems
- **Practical techniques** (LoRA, RAG) that enable modern applications

Each paper is explained at three levels:
1. **Summary & intuition** — Understand the core idea quickly
2. **Deep dive** — See how it works and why it matters (markdown with code examples)
3. **Implementation** — Build it yourself with working notebooks (basic → advanced → real-world)

## How to Use This Section

**If you're preparing for interviews:**
1. Start with the [Interview Prep Roadmap](#interview-prep-roadmap) below
2. Read the markdown explanation for each paper
3. Run the notebook to see implementations
4. Study the interview Q&A questions

**If you want to understand a specific area:**
- [Vision Papers](#vision) — Image models, ResNets, Transformers
- [NLP/LLM Papers](#nlpllm-core) — Transformers, language models, fine-tuning
- [Retrieval & Multimodal](#retrieval--multimodal) — RAG, CLIP, combining modalities
- [Reasoning & Agents](#reasoning--agents) — Chain of Thought, planning, tool use

**If you want historical context:**
- See the [Chronological Timeline](#chronological-timeline) to understand how papers built on each other

## Papers by Domain

### Vision

| # | Paper | Year | Key Contribution | Read Time |
|---|-------|------|-----------------|-----------|
| 1 | [ResNet: Deep Residual Learning for Image Recognition](vision/concepts/01-resnet.md) | 2015 | Residual connections enable very deep networks | 15 min |
| 2 | [Vision Transformer: An Image is Worth 16x16 Words](vision/concepts/02-vision-transformer.md) | 2020 | Transformers work just as well for vision as for NLP | 18 min |

### NLP/LLM Core

| # | Paper | Year | Key Contribution | Read Time |
|---|-------|------|-----------------|-----------|
| 1 | [Attention Is All You Need](nlp/concepts/01-attention-is-all-you-need.md) | 2017 | Transformer architecture replaces RNNs for sequences | 20 min |
| 2 | [BERT: Pre-training of Deep Bidirectional Transformers](nlp/concepts/02-bert.md) | 2018 | Bidirectional pre-training beats unidirectional for understanding | 18 min |
| 3 | [Language Models are Few-Shot Learners (GPT-3)](nlp/concepts/03-gpt3.md) | 2020 | Large models can do tasks with just a few examples | 20 min |
| 4 | [Scaling Laws for Neural Language Models](nlp/concepts/04-scaling-laws.md) | 2020 | Model size, data size, compute follow predictable scaling | 18 min |
| 5 | [LoRA: Low-Rank Adaptation of Large Language Models](nlp/concepts/05-lora.md) | 2021 | Fine-tune large models efficiently with low-rank updates | 16 min |

### Retrieval & Multimodal

| # | Paper | Year | Key Contribution | Read Time |
|---|-------|------|-----------------|-----------|
| 1 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP](retrieval/concepts/01-rag.md) | 2020 | Combine language models with external knowledge retrieval | 18 min |
| 2 | [Learning Transferable Visual Models from Natural Language (CLIP)](retrieval/concepts/02-clip.md) | 2021 | Train vision + language jointly for zero-shot recognition | 19 min |

### Reasoning & Agents

| # | Paper | Year | Key Contribution | Read Time |
|---|-------|------|-----------------|-----------|
| 1 | [Chain-of-Thought Prompting Elicits Reasoning in LLMs](agents/concepts/01-chain-of-thought.md) | 2022 | Breaking problems into steps helps models solve them correctly | 16 min |
| 2 | [ReAct: Synergizing Reasoning and Acting in Language Models](agents/concepts/02-react.md) | 2022 | Combine reasoning (thinking) with actions (tool use) | 17 min |
| 3 | [Tree of Thoughts: Deliberate Problem Solving with LLMs](agents/concepts/03-tree-of-thoughts.md) | 2023 | Explore multiple reasoning paths, not just one linear path | 17 min |

## Interview Prep Roadmap

### Path 1: ML Fundamentals for Interviews (4 papers)
Best for: Pure ML roles, infrastructure, optimization

1. [ResNet](vision/concepts/01-resnet.md) — 45 min
   - **Why first:** Establish how deep networks work
   - **Interview Q:** "Why do residual connections help? When would you use them?"

2. [Attention Is All You Need](nlp/concepts/01-attention-is-all-you-need.md) — 60 min
   - **Why:** Core architecture of modern ML
   - **Interview Q:** "Walk me through self-attention. Why is it better than RNNs?"

3. [Scaling Laws for Neural Language Models](nlp/concepts/04-scaling-laws.md) — 50 min
   - **Why:** Understand model behavior at scale
   - **Interview Q:** "How do you estimate how much data/compute you need?"

4. [LoRA: Low-Rank Adaptation](nlp/concepts/05-lora.md) — 45 min
   - **Why:** Practical fine-tuning for resource constraints
   - **Interview Q:** "Why is LoRA better than full fine-tuning? Trade-offs?"

**Total time:** ~3.5 hours reading + 2-3 hours notebooks

**Interview prep:** You can explain modern ML architecture, understand why scale matters, and code up the key techniques.

---

### Path 2: LLM & Language Models (5 papers)
Best for: NLP roles, LLM engineering, RAG systems

1. [Attention Is All You Need](nlp/concepts/01-attention-is-all-you-need.md) — 60 min
2. [BERT: Pre-training of Deep Bidirectional Transformers](nlp/concepts/02-bert.md) — 55 min
3. [Language Models are Few-Shot Learners (GPT-3)](nlp/concepts/03-gpt3.md) — 60 min
4. [Scaling Laws for Neural Language Models](nlp/concepts/04-scaling-laws.md) — 50 min
5. [LoRA: Low-Rank Adaptation](nlp/concepts/05-lora.md) — 45 min
6. [Retrieval-Augmented Generation](retrieval/concepts/01-rag.md) — 55 min

**Total time:** ~5.5 hours reading + 3-4 hours notebooks

---

### Path 3: AI Agents & Systems (6 papers)
Best for: Agent roles, tool use, multi-step reasoning

1. [Attention Is All You Need](nlp/concepts/01-attention-is-all-you-need.md) — Foundation
2. [Chain-of-Thought Prompting](agents/concepts/01-chain-of-thought.md) — Reasoning
3. [ReAct: Reasoning + Acting](agents/concepts/02-react.md) — Agents
4. [Tree of Thoughts](agents/concepts/03-tree-of-thoughts.md) — Exploration
5. [CLIP](retrieval/concepts/02-clip.md) — Multimodal
6. [Scaling Laws](nlp/concepts/04-scaling-laws.md) — System behavior

**Total time:** ~6-7 hours reading + 3-4 hours notebooks

---

## Chronological Timeline

Understanding how papers built on each other:

```
2015: ResNet (deep networks work)
        ↓
2017: Attention Is All You Need (transformer architecture)
        ↓
2018: BERT (bidirectional pre-training), Vision Transformer (2020)
        ↓
2020: GPT-3, Scaling Laws, RAG, CLIP (multimodal learning)
        ↓
2021: LoRA (efficient fine-tuning)
        ↓
2022: Chain of Thought, ReAct (reasoning + acting)
        ↓
2023: Tree of Thoughts (structured reasoning)
```

## Quick Reference: All 12 Papers

| # | Paper | Year | Domain | Interview Frequency | Implementation Complexity |
|---|-------|------|--------|-------------------|---------------------------|
| 1 | ResNet | 2015 | Vision | Medium | Intermediate |
| 2 | Vision Transformer | 2020 | Vision | Medium | Intermediate |
| 3 | Attention Is All You Need | 2017 | NLP | Very High | Intermediate |
| 4 | BERT | 2018 | NLP | Very High | Intermediate |
| 5 | GPT-3 | 2020 | NLP | High | Basic |
| 6 | Scaling Laws | 2020 | NLP | High | Basic |
| 7 | LoRA | 2021 | NLP | High | Intermediate |
| 8 | RAG | 2020 | Retrieval | High | Intermediate |
| 9 | CLIP | 2021 | Multimodal | Medium | Intermediate |
| 10 | Chain of Thought | 2022 | Agents | Very High | Basic |
| 11 | ReAct | 2022 | Agents | High | Intermediate |
| 12 | Tree of Thoughts | 2023 | Agents | High | Intermediate |

## Contributing

To add a new paper:
1. Follow the 8-section markdown format (see any example paper)
2. Create a 12-cell implementation notebook
3. Update this README with links and metadata
4. Cross-link from related concepts in other domains

See the [main CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

---

**Last Updated:** 2026-05-31
