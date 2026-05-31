# Papers Roadmap: Learn from Seminal AI Research

> A guided path through 12 foundational & recent papers that shaped modern AI. Each paper is explained at interview depth: understand the idea AND code it up.

## Quick Navigation

| Goal | Start Here | Time |
|------|-----------|------|
| **Crack interviews** | [Interview Prep Paths](#interview-prep-paths-by-role) | 4-7 hours |
| **Understand Transformers** | [Attention Is All You Need](../papers/nlp/concepts/01-attention-is-all-you-need.md) | 2 hours |
| **Learn about LLMs** | [LLM Masterclass](#llm-masterclass) | 6-8 hours |
| **Build AI agents** | [Agents & Reasoning](#agents--reasoning) | 5-7 hours |
| **Get the full picture** | [Complete Timeline](#complete-papers-timeline) | 12-16 hours |

## Interview Prep Paths by Role

### ML Engineer (Focus: Fundamentals & Scale)

**Goal:** Understand how modern models work and why scale matters

**Papers (in order):**
1. [ResNet](../papers/vision/concepts/01-resnet.md) — 45 min
   - **Why first:** Establish how deep networks work
   - **Interview Q:** "Why do residual connections help? When would you use them?"

2. [Attention Is All You Need](../papers/nlp/concepts/01-attention-is-all-you-need.md) — 60 min
   - **Why:** Core architecture of modern ML
   - **Interview Q:** "Walk me through self-attention. Why is it better than RNNs?"

3. [Scaling Laws for Neural Language Models](../papers/nlp/concepts/04-scaling-laws.md) — 50 min
   - **Why:** Understand model behavior at scale
   - **Interview Q:** "How do you estimate how much data/compute you need?"

4. [LoRA: Low-Rank Adaptation](../papers/nlp/concepts/05-lora.md) — 45 min
   - **Why:** Practical fine-tuning for resource constraints
   - **Interview Q:** "Why is LoRA better than full fine-tuning? Trade-offs?"

**Total time:** ~3.5 hours reading + 2-3 hours notebooks

**Interview prep:** You can explain modern ML architecture, understand why scale matters, and code up the key techniques.

---

### LLM/NLP Engineer (Focus: Language Models & Retrieval)

**Goal:** Deep understanding of language models, from pre-training to production

**Papers (in order):**
1. [Attention Is All You Need](../papers/nlp/concepts/01-attention-is-all-you-need.md) — 60 min
   - Foundation: How transformers work

2. [BERT: Pre-training of Deep Bidirectional Transformers](../papers/nlp/concepts/02-bert.md) — 55 min
   - **Why:** Understanding vs. generation; bidirectional context

3. [Language Models are Few-Shot Learners (GPT-3)](../papers/nlp/concepts/03-gpt3.md) — 60 min
   - **Why:** Generation, few-shot learning, scaling to billions of parameters

4. [Scaling Laws for Neural Language Models](../papers/nlp/concepts/04-scaling-laws.md) — 50 min
   - **Why:** Predicting model capabilities based on scale

5. [LoRA: Low-Rank Adaptation](../papers/nlp/concepts/05-lora.md) — 45 min
   - **Why:** Efficient fine-tuning for domain adaptation

6. [Retrieval-Augmented Generation](../papers/retrieval/concepts/01-rag.md) — 55 min
   - **Why:** Connecting language models to external knowledge

**Total time:** ~5.5 hours reading + 3-4 hours notebooks

**Interview prep:** You understand language models end-to-end: pre-training, scaling, fine-tuning, and retrieval augmentation.

---

### AI Agents & Systems Engineer (Focus: Reasoning & Tool Use)

**Goal:** Understand how models reason and interact with tools

**Papers (in order):**
1. [Attention Is All You Need](../papers/nlp/concepts/01-attention-is-all-you-need.md) — 60 min
   - Foundation: Transformer architecture

2. [Chain-of-Thought Prompting](../papers/agents/concepts/01-chain-of-thought.md) — 50 min
   - **Why:** Breaking complex problems into steps helps models solve them

3. [ReAct: Synergizing Reasoning and Acting](../papers/agents/concepts/02-react.md) — 55 min
   - **Why:** Combine thinking (reasoning) with doing (tool use)

4. [Tree of Thoughts: Deliberate Problem Solving](../papers/agents/concepts/03-tree-of-thoughts.md) — 55 min
   - **Why:** Explore multiple paths, not just greedy generation

5. [Learning Transferable Visual Models (CLIP)](../papers/retrieval/concepts/02-clip.md) — 60 min
   - **Why:** Multimodal understanding for agents that see and act

6. [GPT-3: Few-Shot Learners](../papers/nlp/concepts/03-gpt3.md) — 60 min
   - **Why:** Understand in-context learning and generalization

**Total time:** ~5.5 hours reading + 3-4 hours notebooks

**Interview prep:** You can explain how agents think (reasoning), act (tools), plan (tree search), and perceive (multimodal).

---

## Complete Papers Timeline

Understand how papers built on each other:

### 2015: Deep Networks With Residuals
- [ResNet](../papers/vision/concepts/01-resnet.md)
  - **Problem:** Deep networks don't train well (vanishing gradients)
  - **Solution:** Residual connections skip layers
  - **Impact:** Enables very deep networks (152+ layers)

### 2017: Transformers Replace RNNs
- [Attention Is All You Need](../papers/nlp/concepts/01-attention-is-all-you-need.md)
  - **Problem:** RNNs are sequential (slow to train)
  - **Solution:** Self-attention allows parallel processing
  - **Impact:** Foundation of modern NLP (BERT, GPT, etc.)

### 2018: Bidirectional Context Matters
- [BERT](../papers/nlp/concepts/02-bert.md)
  - **Problem:** Unidirectional models (like GPT) miss context
  - **Solution:** Train bidirectionally, then fine-tune for tasks
  - **Impact:** SOTA on NLP benchmarks; changed pre-training paradigm

### 2020: Four Papers, Four Directions
- [Scaling Laws](../papers/nlp/concepts/04-scaling-laws.md)
  - **Finding:** Loss follows power law with scale; predictable
  - **Impact:** Justifies trillion-parameter models
  
- [GPT-3](../papers/nlp/concepts/03-gpt3.md)
  - **Finding:** Very large models can do tasks with few examples
  - **Impact:** Few-shot learning becomes viable
  
- [Retrieval-Augmented Generation](../papers/retrieval/concepts/01-rag.md)
  - **Solution:** Retrieve context before generating
  - **Impact:** Enables knowledge-intensive tasks without retraining
  
- [CLIP](../papers/retrieval/concepts/02-clip.md)
  - **Solution:** Train vision + language jointly
  - **Impact:** Multimodal zero-shot learning works

- [Vision Transformer](../papers/vision/concepts/02-vision-transformer.md)
  - **Finding:** Transformers work for vision too
  - **Impact:** Unified architecture across modalities

### 2021: Efficient Fine-tuning
- [LoRA](../papers/nlp/concepts/05-lora.md)
  - **Problem:** Fine-tuning huge models is expensive
  - **Solution:** Update only low-rank components
  - **Impact:** Makes fine-tuning practical for 7B+ models

### 2022: Reasoning Emerges
- [Chain-of-Thought Prompting](../papers/agents/concepts/01-chain-of-thought.md)
  - **Finding:** Asking models to think step-by-step improves accuracy
  - **Impact:** Unlocks reasoning in large models
  
- [ReAct](../papers/agents/concepts/02-react.md)
  - **Finding:** Reasoning + acting together > either alone
  - **Impact:** Enables agents that think AND use tools

### 2023: Structured Exploration
- [Tree of Thoughts](../papers/agents/concepts/03-tree-of-thoughts.md)
  - **Finding:** Exploring multiple paths improves problem-solving
  - **Impact:** Agents can backtrack and explore alternatives

---

## Reading Strategies

### Strategy 1: Deep Dive (2-3 hours per paper)
Best for: Understanding one paper deeply

1. Read the markdown explanation (20-30 min)
2. Study the diagram and code examples (15-20 min)
3. Answer the interview Q&A questions (15-20 min)
4. Run the notebook from scratch (60-90 min)
5. Modify code and experiment (30-45 min)

### Strategy 2: Breadth First (30-45 min per paper)
Best for: Understanding the landscape

1. Read the "Paper Overview" section (10 min)
2. Skim the "Key Ideas" and diagram (10 min)
3. Quick look at code examples (10-15 min)
4. Skip notebooks for now

### Strategy 3: Problem-Focused (varies)
Best for: "I need to understand X"

1. Identify which papers discuss X
2. Jump to the relevant section in that paper's markdown
3. Run the notebook code that demonstrates X
4. Read the interview Q&A about X

### Strategy 4: Interview Prep (6-8 hours total)
Best for: Preparing for interviews

1. Pick your role path above (ML / LLM / Agents)
2. Spend 40-50 min per paper: read markdown + skim notebook
3. Practice answering the interview Q&A from memory
4. Come back to papers you struggled with for deeper dives

---

## FAQ

**Q: How long does it take to learn all 12 papers?**
A: ~12-16 hours total if you read thoroughly and do notebooks. ~6-8 hours if you skim for interviews.

**Q: Should I read papers in chronological order?**
A: No — follow your role path above. But understanding the timeline helps see how ideas built on each other.

**Q: Do I need to read every paper to understand modern AI?**
A: No. "Attention Is All You Need", "BERT", and "Chain of Thought" are the essentials. Others deepen your understanding.

**Q: Can I skip the notebooks?**
A: For interviews, understanding the markdown is usually enough. Notebooks help if you want to actually build things.

**Q: What if I disagree with a paper or think it's outdated?**
A: Great! That's the sign you understand it. Some papers have been superseded (e.g., BERT by newer models), but the core ideas remain.

---

**Next Steps:**
- [📖 Read the first paper](../papers/nlp/concepts/01-attention-is-all-you-need.md)
- [💻 Run the first notebook](../papers/nlp/notebooks/01-attention-is-all-you-need.ipynb)
- [🎯 Pick your interview prep path](#interview-prep-paths-by-role)

**Last Updated:** 2026-05-31
