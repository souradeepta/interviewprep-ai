# ML/AI Interview Prep & Learning Hub — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive, public open-source repository covering ML, LLM, agentic AI, system design, and coding — organized as a structured learning curriculum and interview prep resource.

**Architecture:** Domain-first top-level folders (ml/, llm/, agentic-ai/, system-design/, coding/), each with consistent internal sub-structure (concepts/, implementations/, interview-prep/). Shared roadmaps/ at the root. Three content templates enforced via CONTRIBUTING.md: Template A (concept note .md), Template B (interview simulation .md), Template C (implementation notebook .ipynb).

**Tech Stack:** Markdown, Jupyter Notebooks (Python 3.10+), NumPy, scikit-learn, PyTorch, HuggingFace Transformers, LangChain/LangGraph, Anthropic/OpenAI SDK, matplotlib, plotly.

---

## Task 1: Directory Scaffolding

**Files:**
- Create all directories listed in the spec

- [ ] **Step 1: Create all directories**

```bash
mkdir -p roadmaps
mkdir -p ml/concepts/deep-learning
mkdir -p ml/implementations
mkdir -p ml/interview-prep
mkdir -p llm/concepts
mkdir -p llm/implementations
mkdir -p llm/system-design
mkdir -p llm/interview-prep
mkdir -p agentic-ai/concepts
mkdir -p agentic-ai/implementations
mkdir -p agentic-ai/system-design
mkdir -p agentic-ai/interview-prep
mkdir -p system-design/patterns
mkdir -p system-design/case-studies
mkdir -p system-design/interview-prep
mkdir -p coding/data-structures
mkdir -p coding/algorithms
mkdir -p coding/ml-coding
```

- [ ] **Step 2: Verify structure**

```bash
find . -type d | grep -v '.git' | grep -v 'docs' | sort
```

Expected output:
```
.
./agentic-ai
./agentic-ai/concepts
./agentic-ai/implementations
./agentic-ai/interview-prep
./agentic-ai/system-design
./coding
./coding/algorithms
./coding/data-structures
./coding/ml-coding
./llm
./llm/concepts
./llm/implementations
./llm/interview-prep
./llm/system-design
./ml
./ml/concepts
./ml/concepts/deep-learning
./ml/implementations
./ml/interview-prep
./roadmaps
./system-design
./system-design/case-studies
./system-design/interview-prep
./system-design/patterns
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "scaffold: create full directory structure"
```

---

## Task 2: Root Files — README.md and CONTRIBUTING.md

**Files:**
- Create: `README.md`
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Create README.md**

```markdown
# ML/AI Interview Prep & Learning Hub

> A comprehensive, open-source reference for AI, ML, LLM, and agentic systems —
> covering theory, hands-on implementation, system design, and coding interview prep.

---

## What's Inside

| Domain | Topics |
|--------|--------|
| **Machine Learning** | Supervised/unsupervised learning, neural nets, deep learning, optimization |
| **Large Language Models** | Transformers, RAG, fine-tuning, inference optimization, evals |
| **Agentic AI** | Tool use, memory, planning, multi-agent systems, production agents |
| **System Design** | ML system patterns, case studies, MLOps, LLM/agent system design |
| **Coding** | DSA (data structures + algorithms) + ML algorithm implementation |

---

## Who Is This For

- Engineers preparing for ML/AI engineering interviews at top tech companies
- Practitioners leveling up in LLMs, RAG, and agentic systems
- Students building a structured foundation in machine learning
- Anyone who wants runnable, well-explained implementation notebooks

---

## Start Here — Pick Your Path

| Goal | Start here |
|------|-----------|
| Crack ML engineering interviews | [ML Roadmap](roadmaps/ml-roadmap.md) |
| Learn LLMs from scratch to production | [LLM Roadmap](roadmaps/llm-roadmap.md) |
| Build and understand AI agents | [Agentic AI Roadmap](roadmaps/agentic-roadmap.md) |
| Ace ML system design interviews | [System Design Roadmap](roadmaps/system-design-roadmap.md) |
| DSA + ML coding practice | [Coding](coding/README.md) |
| Browse all roadmaps | [Roadmaps Overview](roadmaps/README.md) |

---

## Repository Structure

```
interviewprep-ml/
├── roadmaps/          # Learning paths per domain (Beginner → Advanced)
├── ml/                # Classical ML + deep learning
├── llm/               # Large language models: theory → production
├── agentic-ai/        # Agents, tool use, multi-agent systems
├── system-design/     # ML/AI system design patterns and case studies
└── coding/            # DSA + ML algorithm coding
```

Each domain folder has:
- `concepts/` — Theory notes (Template A)
- `implementations/` — Jupyter notebooks with runnable code (Template C)
- `interview-prep/` — Full interview simulations (Template B)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add content, file templates, naming conventions, and the PR checklist.
```

- [ ] **Step 2: Create CONTRIBUTING.md**

```markdown
# Contributing

Thank you for contributing! This repo is only as good as its content.
Read this before opening a PR.

---

## What to Contribute

Highest priority gaps (add to these first):
- Any concept file marked as stub (contains only the TL;DR section)
- Interview questions for domains that have fewer than 10 questions
- Notebook implementations — always welcome
- System design case studies

---

## File Templates

All files must follow one of three templates. Do not invent your own structure.

---

### Template A: Concept Note (`.md`)

Use for all files in `concepts/`.

```markdown
# Topic Name

## TL;DR
One paragraph. What it is, why it matters, when you'd use it. No jargon yet.

## Core Intuition
Plain-English explanation before any math. Use an analogy.
If you can't explain it simply, you don't understand it yet.

## How It Works
Theory and math. Use LaTeX for equations: $y = mx + b$.
Use Mermaid or ASCII for diagrams.

## Key Properties / Trade-offs
- Bullet point each property
- Include computational complexity where relevant
- Compare to alternatives

## Common Mistakes / Gotchas
- What people get wrong in interviews
- Subtle edge cases
- Confusions between similar concepts

## Code Example
Minimal runnable Python. No unnecessary imports.

\```python
# Example: comment explains the non-obvious part
\```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain X" | ... |
| "When would you use X over Y?" | ... |
| "What is the time complexity of X?" | ... |

## Related Topics
- [Related concept](../path/to/file.md)

## Resources
- [Paper/post title](url) — one-line description of what you get from it
```

---

### Template B: Full Interview Simulation (`.md`)

Use for all files in `interview-prep/`.

Each question block:

```markdown
---

## Q: [Question exactly as an interviewer would phrase it]

**Difficulty:** Easy | Medium | Hard
**Domain:** ML Theory | LLM | System Design | Coding
**Companies known to ask:** Google, Meta, OpenAI, etc.

### Step 1 — Clarifying Questions to Ask
- Question you should ask the interviewer before answering
- Another clarifying question

### Step 2 — Approach Discussion
Walk through your thinking out loud before committing to an answer or writing code.

### Step 3 — Answer / Solution
Complete answer. Include code where the question is coding-focused.

\```python
# solution code
\```

### Step 4 — Test Cases
For coding questions: list inputs + expected outputs + why each tests something important.

| Input | Expected | Why |
|---|---|---|
| ... | ... | ... |

### Step 5 — Complexity Analysis
**Time:** O(?)  **Space:** O(?)

Explain the dominant term, not just the answer.

### Step 6 — Follow-up Questions
- "What if X changes — how does your solution adapt?"
- "Can you reduce the space complexity?"

### Common Mistakes
- What candidates typically get wrong on this question

---
```

---

### Template C: Implementation Notebook (`.ipynb`)

Use for all files in `implementations/`. Create notebooks in Jupyter.

Required cell sequence:
1. **Markdown — Header:** Topic name, what you'll build, prerequisites (what you need to know first)
2. **Markdown — Concept Recap:** 3–5 cells of theory before any code. Cover the math.
3. **Code — From-Scratch Implementation:** Pure Python + NumPy only. No sklearn, no PyTorch.
4. **Code — Library Implementation:** Same algorithm using sklearn / PyTorch / HuggingFace.
5. **Code — Visualization:** matplotlib or plotly. Make it interpretable with labels and titles.
6. **Markdown + Code — Exercises:** 2–3 challenges for the reader to complete themselves.
7. **Markdown — Summary:** What was covered, what to study next, links to related notebooks.

---

## Naming Conventions

| Type | Convention | Example |
|---|---|---|
| Concept notes | `kebab-case.md` | `attention-mechanism.md` |
| Interview Q&A files | `<domain>-questions.md` | `llm-theory-questions.md` |
| Notebooks | `verb-topic.ipynb` | `implement-attention.ipynb` |
| Roadmaps | `<domain>-roadmap.md` | `agentic-roadmap.md` |

---

## PR Checklist

Before opening a PR, confirm:

- [ ] File follows the correct template (A, B, or C)
- [ ] File name matches naming convention
- [ ] TL;DR section is present and complete (not a stub)
- [ ] All internal links point to files that actually exist
- [ ] Code runs without errors (test it locally)
- [ ] Notebook outputs are cleared before committing (`Kernel > Restart & Clear Output`)
- [ ] No broken Mermaid diagrams (preview locally or on GitHub)
- [ ] New files are cross-linked from at least one related concept file

---

## Content Quality Bar

A concept note is "done" when:
- Someone with no prior knowledge of the topic could read it and understand the core idea
- Someone preparing for an interview could use the Quick-Reference table as a cheat sheet
- The code example runs as-is (copy-paste into a Python REPL and it works)

An interview simulation is "done" when:
- Every section is filled (no skipped steps)
- The follow-up questions are ones a real interviewer would actually ask
- The Common Mistakes section reflects real errors, not obvious ones

A notebook is "done" when:
- All cells run in order without errors on a fresh kernel
- The from-scratch implementation produces the same results as the library implementation
- At least one visualization is present
```

- [ ] **Step 3: Commit**

```bash
git add README.md CONTRIBUTING.md
git commit -m "docs: add root README and CONTRIBUTING with all three templates"
```

---

## Task 3: Roadmaps

**Files:**
- Create: `roadmaps/README.md`
- Create: `roadmaps/ml-roadmap.md`
- Create: `roadmaps/llm-roadmap.md`
- Create: `roadmaps/agentic-roadmap.md`
- Create: `roadmaps/system-design-roadmap.md`

- [ ] **Step 1: Create roadmaps/README.md**

```markdown
# Roadmaps

Choose your path based on what you're preparing for.

| Roadmap | Best for | Time estimate |
|---------|----------|---------------|
| [ML Roadmap](ml-roadmap.md) | ML engineering interviews, ML fundamentals | 8–12 weeks |
| [LLM Roadmap](llm-roadmap.md) | LLM engineering, RAG, fine-tuning interviews | 6–10 weeks |
| [Agentic AI Roadmap](agentic-roadmap.md) | AI agent engineering, agentic system design | 4–6 weeks |
| [System Design Roadmap](system-design-roadmap.md) | ML/AI system design interview rounds | 4–6 weeks |

**Recommended order if starting from scratch:**
ML Roadmap → LLM Roadmap → Agentic Roadmap → System Design Roadmap

Each roadmap has three phases. You do not need to complete Phase 3 before starting interviews —
Phase 2 is interview-ready for most roles.
```

- [ ] **Step 2: Create roadmaps/ml-roadmap.md**

```markdown
# Machine Learning Roadmap

## Who This Is For
Engineers who want to deeply understand classical ML, neural networks, and deep learning — and be
ready for ML theory and coding interviews at any top tech company. You should be comfortable
with Python and have basic linear algebra / calculus knowledge.

---

## Phase 1 — Foundations (Beginner)
**Goal:** Explain and implement core ML algorithms from scratch. Pass easy/medium ML theory questions.
**Estimated time:** 3–4 weeks at 10 hrs/week

- [ ] [Supervised Learning](../ml/concepts/supervised-learning.md)
- [ ] [Probability & Statistics](../ml/concepts/probability-statistics.md)
- [ ] [Evaluation Metrics](../ml/concepts/evaluation-metrics.md)
- [ ] [Optimization](../ml/concepts/optimization.md)
- [ ] [Regularization](../ml/concepts/regularization.md)
- [ ] Implement: [Linear Regression](../ml/implementations/linear-regression.ipynb)
- [ ] Implement: [Logistic Regression](../ml/implementations/logistic-regression.ipynb)
- [ ] Practice: [ML Theory Questions](../ml/interview-prep/ml-theory-questions.md) — Q1–Q20

**Phase 1 exit check:**
- Can you derive the gradient descent update rule from scratch?
- Can you explain bias-variance trade-off with a diagram?
- Can you implement linear regression in NumPy without looking it up?

---

## Phase 2 — Core Depth (Intermediate)
**Goal:** Cover tree models, unsupervised learning, neural networks, and backprop. Pass medium/hard ML theory questions.
**Estimated time:** 3–4 weeks at 10 hrs/week

- [ ] [Unsupervised Learning](../ml/concepts/unsupervised-learning.md)
- [ ] [Feature Engineering](../ml/concepts/feature-engineering.md)
- [ ] [Ensemble Methods](../ml/concepts/ensemble-methods.md)
- [ ] [Neural Networks](../ml/concepts/neural-networks.md)
- [ ] [Deep Learning — CNNs](../ml/concepts/deep-learning/cnns.md)
- [ ] [Deep Learning — RNNs & LSTMs](../ml/concepts/deep-learning/rnns-lstms.md)
- [ ] Implement: [Decision Tree](../ml/implementations/decision-tree.ipynb)
- [ ] Implement: [Random Forest](../ml/implementations/random-forest.ipynb)
- [ ] Implement: [K-Means From Scratch](../ml/implementations/kmeans-from-scratch.ipynb)
- [ ] Implement: [Neural Net From Scratch](../ml/implementations/neural-net-from-scratch.ipynb)
- [ ] Implement: [Backpropagation](../ml/implementations/backpropagation.ipynb)
- [ ] Practice: [ML Theory Questions](../ml/interview-prep/ml-theory-questions.md) — Q21–Q50
- [ ] Practice: [ML Coding Questions](../ml/interview-prep/ml-coding-questions.md) — Q1–Q10

**Phase 2 exit check:**
- Can you explain how a random forest reduces variance without increasing bias?
- Can you implement backprop for a 2-layer network with no libraries?
- Can you describe when gradient boosting beats random forest and why?

---

## Phase 3 — Advanced + Production (Advanced)
**Goal:** Cover transformers, distributed training, MLOps. Ready for senior ML engineer interviews.
**Estimated time:** 2–4 weeks at 10 hrs/week

- [ ] [Deep Learning — Attention Mechanism](../ml/concepts/deep-learning/attention-mechanism.md)
- [ ] [Deep Learning — Transformers](../ml/concepts/deep-learning/transformers.md)
- [ ] Implement: [CNN Image Classifier](../ml/implementations/cnn-image-classifier.ipynb)
- [ ] Implement: [Backpropagation](../ml/implementations/backpropagation.ipynb) — verify gradient correctness
- [ ] Practice: [ML Coding Questions](../ml/interview-prep/ml-coding-questions.md) — Q11–Q20
- [ ] Practice: [ML Case Studies](../ml/interview-prep/case-studies.md)
- [ ] [System Design — MLOps Overview](../system-design/patterns/mlops-overview.md)

**Phase 3 exit check:**
- Can you explain the attention mechanism and why it replaced RNNs?
- Can you design an end-to-end training pipeline for a large model?
- Can you answer: "How would you deploy a model that needs to serve 10k QPS?"

---

## Interview Readiness Checklist
- [ ] Implemented linear/logistic regression, decision tree, neural net from scratch (NumPy only)
- [ ] Can explain gradient descent, backprop, attention without notes
- [ ] Completed at least 20 ML theory questions in simulation format
- [ ] Completed at least 5 ML coding questions from scratch
- [ ] Done one full case study mock (recommendation or ranking system)

---

## Suggested Weekly Schedule

| Week | Focus | Files |
|------|-------|-------|
| 1 | Supervised learning + probability | supervised-learning.md, probability-statistics.md |
| 2 | Evaluation + optimization + regularization | evaluation-metrics.md, optimization.md, regularization.md |
| 3 | Implement linear + logistic regression | linear-regression.ipynb, logistic-regression.ipynb |
| 4 | Theory practice + unsupervised | ml-theory-questions.md Q1–20, unsupervised-learning.md |
| 5 | Ensemble methods + trees | ensemble-methods.md, decision-tree.ipynb, random-forest.ipynb |
| 6 | Neural networks + backprop | neural-networks.md, neural-net-from-scratch.ipynb, backpropagation.ipynb |
| 7 | Deep learning (CNN, RNN) | cnns.md, rnns-lstms.md, cnn-image-classifier.ipynb |
| 8 | Attention + transformers + coding Qs | attention-mechanism.md, transformers.md, ml-coding-questions.md |
```

- [ ] **Step 3: Create roadmaps/llm-roadmap.md**

```markdown
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
```

- [ ] **Step 4: Create roadmaps/agentic-roadmap.md**

```markdown
# Agentic AI Roadmap

## Who This Is For
Engineers who want to build, evaluate, and deploy AI agents — from simple tool-calling loops
to complex multi-agent systems. Assumes basic LLM familiarity (complete LLM Roadmap Phase 1 first).

---

## Phase 1 — Foundations (Beginner)
**Goal:** Understand what agents are, build a basic agent loop, use tools and memory.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [What Is an Agent](../agentic-ai/concepts/what-is-an-agent.md)
- [ ] [Tool Use](../agentic-ai/concepts/tool-use.md)
- [ ] [Memory Types](../agentic-ai/concepts/memory-types.md)
- [ ] Implement: [Basic Agent Loop](../agentic-ai/implementations/basic-agent-loop.ipynb)
- [ ] Implement: [Tool Calling Agent](../agentic-ai/implementations/tool-calling-agent.ipynb)
- [ ] Practice: [Agentic Theory Questions](../agentic-ai/interview-prep/agentic-theory-questions.md) — Q1–Q10

**Phase 1 exit check:**
- Can you build a ReAct agent from scratch using only raw API calls (no framework)?
- Can you explain the difference between in-context and external memory?

---

## Phase 2 — Core Depth (Intermediate)
**Goal:** Planning/reasoning strategies, multi-agent patterns, RAG agents.
**Estimated time:** 2–3 weeks at 10 hrs/week

- [ ] [Planning & Reasoning](../agentic-ai/concepts/planning-reasoning.md)
- [ ] [Multi-Agent Systems](../agentic-ai/concepts/multi-agent-systems.md)
- [ ] Implement: [RAG Agent](../agentic-ai/implementations/rag-agent.ipynb)
- [ ] Implement: [Multi-Agent Workflow](../agentic-ai/implementations/multi-agent-workflow.ipynb)
- [ ] Implement: [LangGraph Agent](../agentic-ai/implementations/langgraph-agent.ipynb)
- [ ] Implement: [Memory Agent](../agentic-ai/implementations/memory-agent.ipynb)
- [ ] Practice: [Agentic Theory Questions](../agentic-ai/interview-prep/agentic-theory-questions.md) — Q11–Q25

**Phase 2 exit check:**
- Can you implement a multi-agent workflow where one agent routes tasks to specialist agents?
- Can you explain Tree of Thought vs. ReAct vs. MCTS for planning?

---

## Phase 3 — Advanced + Production (Advanced)
**Goal:** Agent evals, safety, production deployment, system design.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [Agent Evals](../agentic-ai/concepts/agent-evals.md)
- [ ] [Safety & Alignment](../agentic-ai/concepts/safety-alignment.md)
- [ ] [System Design — Agentic System Design](../agentic-ai/system-design/agentic-system-design.md)
- [ ] [System Design — Multi-Agent Orchestration](../agentic-ai/system-design/multi-agent-orchestration.md)
- [ ] [System Design — Production Agents](../agentic-ai/system-design/production-agents.md)
- [ ] Practice: [Agentic System Design Questions](../agentic-ai/interview-prep/agentic-system-design-questions.md)

**Phase 3 exit check:**
- Can you design an eval harness for a customer support agent?
- Can you design a multi-agent orchestration system with human-in-the-loop checkpoints?

---

## Interview Readiness Checklist
- [ ] Built a ReAct agent from raw API (no framework)
- [ ] Built a multi-agent workflow with at least 2 agents
- [ ] Can explain 3 planning/reasoning strategies with trade-offs
- [ ] Completed 20+ agentic theory questions in simulation format
- [ ] Completed one full agentic system design mock

---

## Suggested Weekly Schedule

| Week | Focus | Files |
|------|-------|-------|
| 1 | What is an agent + tool use + basic loop | what-is-an-agent.md, tool-use.md, basic-agent-loop.ipynb, tool-calling-agent.ipynb |
| 2 | Memory + RAG agent + planning | memory-types.md, planning-reasoning.md, rag-agent.ipynb, memory-agent.ipynb |
| 3 | Multi-agent + LangGraph | multi-agent-systems.md, multi-agent-workflow.ipynb, langgraph-agent.ipynb |
| 4 | Evals + safety + system design | agent-evals.md, safety-alignment.md, system design files, system design Qs |
```

- [ ] **Step 5: Create roadmaps/system-design-roadmap.md**

```markdown
# ML System Design Roadmap

## Who This Is For
Engineers preparing for ML/AI system design interview rounds. Assumes you can already discuss
ML models and have some engineering background. System design rounds typically last 45–60 minutes
and expect you to design a complete ML-powered product feature.

---

## Phase 1 — Foundations (Beginner)
**Goal:** Learn the ML system design framework. Understand core infrastructure patterns.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [System Design Framework](../system-design/interview-prep/system-design-framework.md)
- [ ] [Online vs. Batch Inference](../system-design/patterns/online-vs-batch-inference.md)
- [ ] [Feature Store](../system-design/patterns/feature-store.md)
- [ ] [Data Pipelines](../system-design/patterns/data-pipelines.md)
- [ ] Practice: [System Design Questions](../system-design/interview-prep/system-design-questions.md) — Q1–Q5 (use framework, don't worry about depth yet)

**Phase 1 exit check:**
- Can you describe the ML system design framework in under 2 minutes?
- Can you explain the difference between online and batch feature computation?

---

## Phase 2 — Case Studies (Intermediate)
**Goal:** Practice designing complete ML systems end-to-end.
**Estimated time:** 2–3 weeks at 10 hrs/week

- [ ] [Model Registry](../system-design/patterns/model-registry.md)
- [ ] [A/B Testing](../system-design/patterns/ab-testing.md)
- [ ] [MLOps Overview](../system-design/patterns/mlops-overview.md)
- [ ] [Case Study — Recommendation System](../system-design/case-studies/recommendation-system.md)
- [ ] [Case Study — Search Ranking](../system-design/case-studies/search-ranking.md)
- [ ] [Case Study — Fraud Detection](../system-design/case-studies/fraud-detection.md)
- [ ] Practice: [System Design Questions](../system-design/interview-prep/system-design-questions.md) — Q6–Q15

**Phase 2 exit check:**
- Can you design a recommendation system end-to-end in 45 minutes?
- Can you explain how to run an A/B test for an ML model and detect interference effects?

---

## Phase 3 — Advanced + LLM/Agentic Design (Advanced)
**Goal:** Design LLM-powered systems and agentic platforms under interview conditions.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [Case Study — Content Moderation](../system-design/case-studies/content-moderation.md)
- [ ] [Case Study — Ads Click Prediction](../system-design/case-studies/ads-click-prediction.md)
- [ ] [LLM System Design — RAG System](../llm/system-design/rag-system-design.md)
- [ ] [LLM System Design — LLM Serving](../llm/system-design/llm-serving-design.md)
- [ ] [Agentic System Design](../agentic-ai/system-design/agentic-system-design.md)
- [ ] Practice: [System Design Questions](../system-design/interview-prep/system-design-questions.md) — Q16–Q25

**Phase 3 exit check:**
- Can you design a production RAG system with latency SLAs, observability, and fallbacks?
- Can you design an ad click prediction system at 1M QPS?

---

## Interview Readiness Checklist
- [ ] Memorized the ML system design framework (can recite structure from memory)
- [ ] Completed at least 3 full case study mocks (recommendation, fraud, ranking)
- [ ] Can design a RAG system with production concerns addressed
- [ ] Completed 15+ system design questions in simulation format

---

## Suggested Weekly Schedule

| Week | Focus | Files |
|------|-------|-------|
| 1 | Framework + core patterns | system-design-framework.md, online-vs-batch-inference.md, feature-store.md |
| 2 | Case studies: recommendation + ranking | recommendation-system.md, search-ranking.md |
| 3 | Case studies: fraud + moderation + ads | fraud-detection.md, content-moderation.md, ads-click-prediction.md |
| 4 | LLM + agentic system design | rag-system-design.md, llm-serving-design.md, agentic-system-design.md |
```

- [ ] **Step 6: Commit roadmaps**

```bash
git add roadmaps/
git commit -m "docs: add all four domain roadmaps and roadmaps index"
```

---

## Task 4: ML Concepts — Core Files

**Files:** `ml/README.md`, `ml/resources.md`, `ml/concepts/supervised-learning.md`, `unsupervised-learning.md`, `neural-networks.md`, `optimization.md`, `regularization.md`, `evaluation-metrics.md`, `feature-engineering.md`, `ensemble-methods.md`, `probability-statistics.md`

- [ ] **Step 1: Create ml/README.md**

```markdown
# Machine Learning

Classical ML, deep learning foundations, and implementation from scratch.

## Contents
| Folder | What's inside |
|--------|--------------|
| [concepts/](concepts/) | Theory notes for each ML topic (Template A) |
| [implementations/](implementations/) | Jupyter notebooks with from-scratch code |
| [interview-prep/](interview-prep/) | Full interview simulations (Template B) |

Start with the [ML Roadmap](../roadmaps/ml-roadmap.md) for a guided path.
```

- [ ] **Step 2: Create ml/resources.md**

```markdown
# ML Resources

## Textbooks
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) — Hastie, Tibshirani, Friedman. Free PDF. Authoritative reference for classical ML.
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville. Free online.
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Bishop. Bayesian perspective.

## Courses
- [Stanford CS229](https://cs229.stanford.edu/) — Andrew Ng. Lecture notes are excellent.
- [fast.ai Practical Deep Learning](https://course.fast.ai/) — Top-down, code-first.
- [Stanford CS231n](https://cs231n.stanford.edu/) — Best for CNNs.

## Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. Required reading.
- [XGBoost](https://arxiv.org/abs/1603.02754) — How gradient boosting works at scale.

## Practice
- [Kaggle](https://kaggle.com) — Competitions and datasets.
- [ML Interviews Book — Chip Huyen](https://huyenchip.com/ml-interviews-book/) — Free.
```

- [ ] **Step 3: Create ml/concepts/supervised-learning.md**

```markdown
# Supervised Learning

## TL;DR
Supervised learning trains a model on labeled input-output pairs so it can predict outputs for
unseen inputs. The most common ML paradigm — underpins classification, regression, and ranking.

## Core Intuition
A student learning from an answer key. You show the model thousands of (question, correct answer)
pairs. It adjusts internal parameters until it reliably predicts answers. The "supervision" is
the label — the signal that tells the model when it is wrong.

## How It Works
Given dataset $\{(x_i, y_i)\}_{i=1}^n$, learn $f: X \to Y$ minimizing a loss $\mathcal{L}$.

**Regression** (continuous $y$): Mean Squared Error
$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Classification** (discrete $y$): Cross-Entropy Loss
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \sum_{c} y_{ic} \log(\hat{p}_{ic})$$

Optimization via gradient descent: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.

## Key Properties / Trade-offs
- Requires labeled data — expensive at scale
- Generalizes poorly outside training distribution (distribution shift)
- Bias-variance trade-off: high-capacity models have low bias but high variance

## Common Mistakes / Gotchas
- Evaluating on training set — always use a held-out test set
- Data leakage: future information in training features
- Class imbalance: accuracy misleads when 99% of samples are one class
- Not normalizing features for gradient-based or distance-based models

## Code Example
```python
import numpy as np

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)
X_b = np.hstack([np.ones((len(X), 1)), X])
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # normal equation
print(theta)  # [intercept, slope]
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is supervised learning?" | Learning input→output mapping from labeled data by minimizing a loss function |
| "What is the bias-variance trade-off?" | Bias = underfitting (too simple); Variance = overfitting (too sensitive to training data) |
| "How do you prevent overfitting?" | Regularization, dropout, early stopping, more data, cross-validation |
| "What is data leakage?" | Future or test information contaminating training — causes inflated metrics that don't hold in production |

## Related Topics
- [Optimization](optimization.md) — [Regularization](regularization.md) — [Evaluation Metrics](evaluation-metrics.md)

## Resources
- [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf)
- [Understanding Bias-Variance](http://scott.fortmann-roe.com/docs/BiasVariance.html)
```

- [ ] **Step 4: Create ml/concepts/optimization.md**

```markdown
# Optimization

## TL;DR
Finding model parameters that minimize a loss function. Gradient descent and its variants (SGD,
Adam) are the standard. Understanding optimization is essential for debugging training, tuning
hyperparameters, and explaining why models behave as they do.

## Core Intuition
Standing on a hilly landscape in fog, trying to reach the lowest valley. Gradient descent says:
check which direction is downhill (the gradient), take a step that way. The learning rate controls
step size — too large and you overshoot; too small and you never arrive.

## How It Works

**Gradient Descent:** $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$

**SGD (stochastic):** one sample per update — noisier but faster iteration.

**Mini-batch SGD:** gradient over B samples — best trade-off. Standard in practice.

**Momentum:** adds velocity term to smooth oscillations:
$v \leftarrow \beta v + (1-\beta)\nabla\mathcal{L}$, $\theta \leftarrow \theta - \eta v$

**Adam:** adaptive per-parameter learning rates via first and second moment estimates.
Defaults: lr=1e-3, β₁=0.9, β₂=0.999. Converges fast; may not generalize as well as SGD.

## Key Properties / Trade-offs
- SGD generalizes better than Adam in some settings (finds flatter minima)
- Adam converges faster but can overfit to sharp minima
- Learning rate is the most impactful hyperparameter — use a scheduler
- Larger batch = less gradient noise = faster convergence but sometimes worse generalization

## Common Mistakes / Gotchas
- Not normalizing inputs — gradients explode or vanish
- Learning rate too high: loss oscillates; too low: painfully slow
- Forgetting learning rate warmup for transformers

## Code Example
```python
import numpy as np

def adam(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t += 1
    m = b1 * m + (1 - b1) * grads
    v = b2 * v + (1 - b2) * grads**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    params -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v, t
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Why use Adam over SGD?" | Adam adapts learning rates per parameter and converges faster; SGD with momentum can generalize better |
| "What is a learning rate scheduler?" | Adjusts lr during training — warmup avoids early instability, cosine decay improves final convergence |
| "Why does SGD generalize better?" | Noisier gradients find flatter minima that tend to generalize better |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Regularization](regularization.md) — [Neural Networks](neural-networks.md)

## Resources
- [Overview of Gradient Descent Optimizers](https://arxiv.org/abs/1609.04747) — Ruder
- [Adam paper](https://arxiv.org/abs/1412.6980) — Kingma & Ba
```

- [ ] **Step 5: Create ml/concepts/regularization.md**

```markdown
# Regularization

## TL;DR
Regularization reduces overfitting by constraining model complexity. The main techniques:
L1 (Lasso), L2 (Ridge), dropout, early stopping, and data augmentation. Every production
ML model uses at least one form of regularization.

## Core Intuition
A penalty for complexity. Without it, a model memorizes training data. With it, the model
is forced to learn simpler rules that generalize. Occam's Razor baked into the math.

## How It Works

**L2 (Ridge):** $\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_j w_j^2$ — drives weights toward zero, not exactly zero.

**L1 (Lasso):** $\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_j |w_j|$ — drives many weights to exactly zero (sparsity).

**Dropout:** randomly zero neurons during training with probability p. At inference, disable.

**Early stopping:** monitor validation loss and stop when it starts increasing. Free regularization.

**Data augmentation:** artificially expand training data (flips, crops, noise). Especially effective for images.

## Key Properties / Trade-offs
- L1 vs L2: L1 for feature selection (sparse); L2 otherwise (smooth)
- Higher λ = stronger regularization = more bias, less variance
- Dropout most effective in fully connected layers

## Common Mistakes / Gotchas
- Not tuning λ — regularization strength needs cross-validation
- Applying dropout at inference time
- Over-regularizing: causes underfitting

## Code Example
```python
import numpy as np

def l2_loss(y_true, y_pred, weights, lam=0.01):
    return np.mean((y_true - y_pred)**2) + lam * np.sum(weights**2)

def l1_loss(y_true, y_pred, weights, lam=0.01):
    return np.mean((y_true - y_pred)**2) + lam * np.sum(np.abs(weights))
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "L1 vs L2?" | L1 penalizes absolute weights — drives some to exactly zero (feature selection). L2 penalizes squared weights — all shrink toward zero but none exactly zero. |
| "How does dropout work?" | Randomly zeros neurons during training, forcing redundant representations. Acts as an ensemble of sub-networks. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Optimization](optimization.md) — [Neural Networks](neural-networks.md)

## Resources
- [Dropout paper](https://jmlr.org/papers/v15/srivastava14a.html) — Srivastava et al.
```

- [ ] **Step 6: Create ml/concepts/evaluation-metrics.md**

```markdown
# Evaluation Metrics

## TL;DR
Choosing the right metric is as important as choosing the right model. Accuracy misleads on
imbalanced data. Precision/recall trade-off depends on the cost of each error type. AUC-ROC
summarizes performance across all thresholds.

## Core Intuition
The metric you optimize determines what your model learns. If you optimize accuracy on a
99%-negative dataset, predicting "negative" always gets 99% accuracy while being useless.
Match the metric to the business cost of different error types.

## How It Works

**Confusion matrix:** TP, TN, FP (Type I error), FN (Type II error)

$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}$$
$$\text{F1} = 2 \cdot \frac{P \cdot R}{P + R}$$

**AUC-ROC:** probability that a random positive ranks above a random negative. Threshold-independent.

**Regression:** MSE (penalizes outliers), MAE (robust), RMSE (interpretable units), R².

## Key Properties / Trade-offs
- High precision = few false alarms; high recall = few missed detections
- Increasing threshold: raises precision, lowers recall
- Use F1 when both matter equally; AUC when threshold is unknown

## Common Mistakes / Gotchas
- Using accuracy on imbalanced datasets
- Reporting precision/recall without specifying threshold
- AUC doesn't tell you performance at your operating threshold

## Code Example
```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
y_prob = np.array([0.9, 0.2, 0.8, 0.4, 0.1, 0.85])
print(f"F1: {f1_score(y_true, y_pred):.2f}, AUC: {roc_auc_score(y_true, y_prob):.2f}")
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "When would you use precision over recall?" | When FP is expensive (spam filter). Use recall when FN is expensive (cancer detection). |
| "What is AUC-ROC?" | P(random positive ranks above random negative). Threshold-independent. 0.5=random, 1.0=perfect. |
| "How do you handle class imbalance?" | Resample, class weights, use F1/AUC over accuracy, precision-recall curve. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Feature Engineering](feature-engineering.md)

## Resources
- [sklearn metrics docs](https://scikit-learn.org/stable/modules/model_evaluation.html)
```

- [ ] **Step 7: Create ml/concepts/unsupervised-learning.md**

```markdown
# Unsupervised Learning

## TL;DR
Finding structure in unlabeled data. Key tasks: clustering (K-Means, DBSCAN), dimensionality
reduction (PCA, t-SNE, UMAP), and density estimation. Used for EDA, feature learning, and
anomaly detection.

## Core Intuition
No answer key. Like sorting foreign coins you've never seen — grouping by size, color, and
markings without knowing their value. The algorithm must find patterns on its own.

## How It Works

**K-Means:** assign points to nearest centroid, update centroids to cluster mean, repeat.
Minimizes inertia (within-cluster sum of squares). Requires choosing k.

**PCA:** finds orthogonal directions of maximum variance. Projects data to top-k components.
Linear — use t-SNE/UMAP for nonlinear visualization.

**DBSCAN:** density-based. Groups closely packed points, marks sparse points as outliers.
Doesn't require specifying k. Parameters: ε (neighborhood radius), min_samples.

## Key Properties / Trade-offs
- K-Means assumes spherical clusters; fails on complex shapes
- PCA is linear; t-SNE/UMAP better for visualization but not invertible
- DBSCAN handles noise naturally but ε is sensitive
- Evaluation is hard — no ground truth. Use silhouette score or domain validation.

## Common Mistakes / Gotchas
- Not normalizing before K-Means (Euclidean distance — scale matters)
- Choosing k arbitrarily — use elbow method and silhouette scores
- Confusing t-SNE distances as meaningful (only topology is preserved, not distances)

## Code Example
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

X = np.random.randn(200, 10)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
X_2d = PCA(n_components=2).fit_transform(X)
print(f"Explained variance: {PCA(2).fit(X).explained_variance_ratio_}")
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "How does K-Means work?" | Init k centroids, assign each point to nearest, update centroids to mean, repeat until convergence. |
| "What is PCA?" | Finds directions of maximum variance (principal components), projects data onto them. Reduces dimensionality while preserving variance. |
| "How do you choose k?" | Elbow method (plot inertia vs k, pick knee) + silhouette score (higher = better-separated clusters). |

## Related Topics
- [Feature Engineering](feature-engineering.md) — [Implementations: K-Means](../implementations/kmeans-from-scratch.ipynb)

## Resources
- [A Tutorial on PCA](https://arxiv.org/abs/1404.1100) — Shlens
```

- [ ] **Step 8: Create ml/concepts/feature-engineering.md**

```markdown
# Feature Engineering

## TL;DR
Transforming raw data into informative inputs for ML models. Often the highest-leverage activity
in applied ML — better features beat better models. Includes encoding categoricals, handling
missing values, creating interaction features, and scaling.

## Core Intuition
Models can only learn from what you give them. A raw date "2023-11-15" is opaque to a linear model
but powerful when decomposed into day_of_week=2, month=11, is_holiday=False. Feature engineering
encodes domain knowledge into a form models can exploit.

## How It Works

**Numerical:** StandardScaler (zero mean, unit variance), log transform for right-skewed distributions, binning for tree models.

**Categorical:**
- One-hot encoding: low-cardinality (< 50 categories)
- Target encoding: high-cardinality — replace category with target mean (use inside CV folds to prevent leakage)
- Embeddings: deep learning, high-cardinality

**Missing values:** mean/median imputation (simple), model-based imputation, add "was_missing" binary feature.

**Interactions:** price_per_sqft = price / sqft, polynomial features (x², xy), date decomposition.

## Key Properties / Trade-offs
- Tree models (XGBoost, RF) don't need feature scaling; linear/distance models do
- Target encoding risks leakage if not done inside CV folds
- One-hot + high cardinality = high-dimensional sparse matrix (bad)

## Common Mistakes / Gotchas
- Scaling on full dataset before train/test split → data leakage
- One-hot encoding a feature with thousands of unique values
- Ignoring missingness pattern — "missing" can be informative

## Code Example
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ('num', Pipeline([('scaler', StandardScaler())]), ['age', 'salary']),
    ('cat', Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['city'])
])
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Target vs one-hot encoding?" | One-hot for low-cardinality; target encoding for high-cardinality. Target encoding must be inside CV folds to avoid leakage. |
| "Do tree models need scaling?" | No — tree splits on thresholds, invariant to monotonic transforms. Linear models and KNN do. |
| "Missing value strategies?" | Mean/median imputation + add binary "was_missing" indicator column. Or model-based imputation for complex missingness. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Evaluation Metrics](evaluation-metrics.md)

## Resources
- [Feature Engineering and Selection](http://www.feat.engineering/) — Kuhn & Johnson. Free online.
```

- [ ] **Step 9: Create ml/concepts/ensemble-methods.md**

```markdown
# Ensemble Methods

## TL;DR
Combining multiple models to outperform any individual model. Three paradigms: bagging
(Random Forest), boosting (XGBoost, LightGBM), and stacking. Gradient boosted trees are
the dominant approach for tabular data in industry.

## Core Intuition
Ask 100 experts instead of one. Each expert is sometimes wrong, but errors are uncorrelated,
so averaging cancels noise. This reduces variance (bagging) or reduces bias iteratively (boosting).

## How It Works

**Bagging:** train T models on different bootstrap samples, aggregate by vote/mean.
Random Forest adds feature subsampling at each split to de-correlate trees.

**Boosting:** train sequentially, each model correcting previous errors.
- AdaBoost: upweight misclassified samples
- Gradient Boosting: fit new tree to negative gradient (residuals)
- XGBoost/LightGBM: GPU-optimized, regularized, histogram-based gradient boosting

**Stacking:** train a meta-model on out-of-fold predictions of diverse base models.

## Key Properties / Trade-offs
- Bagging: reduces variance — best when base models overfit
- Boosting: reduces bias and variance — best when base models underfit; sensitive to outliers
- XGBoost > Random Forest on most tabular benchmarks, but harder to tune

## Common Mistakes / Gotchas
- n_estimators with boosting: lower learning_rate needs more trees
- Stacking without proper CV leaks meta-training labels
- Boosting on noisy labels amplifies noise — consider robust loss functions

## Code Example
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
print("RF:", cross_val_score(rf, X, y, cv=5, scoring='roc_auc').mean())
print("GB:", cross_val_score(gb, X, y, cv=5, scoring='roc_auc').mean())
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "How does Random Forest reduce variance?" | Bootstrap sampling + feature subsampling at each split produces uncorrelated trees. Averaging uncorrelated predictions reduces variance. |
| "Bagging vs boosting?" | Bagging: parallel, reduces variance. Boosting: sequential, corrects errors, reduces bias. |
| "XGBoost vs Random Forest?" | XGBoost typically achieves lower error but requires more tuning; RF is faster and more robust to hyperparameters. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Implementations: Random Forest](../implementations/random-forest.ipynb)

## Resources
- [XGBoost paper](https://arxiv.org/abs/1603.02754) — Chen & Guestrin
- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) — Breiman
```

- [ ] **Step 10: Create ml/concepts/probability-statistics.md**

```markdown
# Probability & Statistics

## TL;DR
The mathematical foundation of ML. Key concepts: probability distributions, Bayes' theorem,
MLE, hypothesis testing, CLT. These appear directly in interviews and underpin every ML algorithm.

## Core Intuition
ML is reasoning under uncertainty. A model that outputs "80% confident" is making a probability
claim. Statistics tells us: is this signal real, or just noise?

## How It Works

**Bayes' Theorem:** $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$ — prior belief updated by evidence.

**Common distributions:**
- Normal $\mathcal{N}(\mu, \sigma^2)$: sum of many independent r.v.s; central to statistics
- Bernoulli / Binomial: binary outcomes
- Poisson: count of events in a fixed interval

**MLE:** $\hat{\theta} = \arg\max_\theta \prod P(x_i | \theta)$ — parameters most likely to produce observed data.
In practice use log-likelihood (converts product to sum).

**Central Limit Theorem:** sample mean of n i.i.d. r.v.s converges to Normal as n→∞, regardless
of underlying distribution.

**p-value:** P(data this extreme | null hypothesis is true). p < 0.05 → reject null.

## Key Properties / Trade-offs
- MLE can overfit with small data — MAP (+ prior) = regularized MLE
- p-values measure significance, not effect size or practical importance
- Multiple comparisons inflate false positive rate — Bonferroni or FDR correction

## Common Mistakes / Gotchas
- Confusing P(A|B) with P(B|A) — base rate fallacy
- Thinking p < 0.05 means the result is "real" or "important"
- Mixing up standard deviation (σ) and standard error (σ/√n)

## Code Example
```python
import numpy as np
from scipy import stats

data = np.random.normal(loc=5, scale=2, size=1000)
mu_mle, sigma_mle = np.mean(data), np.std(data, ddof=0)
print(f"MLE: mu={mu_mle:.2f}, sigma={sigma_mle:.2f}")

a, b = np.random.normal(10, 2, 100), np.random.normal(10.5, 2, 100)
t, p = stats.ttest_ind(a, b)
print(f"t={t:.3f}, p={p:.4f}")
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is Bayes' theorem?" | P(A|B) = P(B|A)·P(A)/P(B). Updates prior belief with observed evidence to get posterior. |
| "What is MLE?" | Find parameters that maximize P(observed data | parameters). For Gaussian: sample mean and variance. |
| "What is CLT?" | Distribution of sample means → Normal as n→∞, regardless of underlying distribution. Justifies many statistical tests. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [System Design: A/B Testing](../../system-design/patterns/ab-testing.md)

## Resources
- [Think Stats](https://greenteapress.com/wp/think-stats-2e/) — Downey. Free. Python-based.
```

- [ ] **Step 11: Create ml/concepts/neural-networks.md**

```markdown
# Neural Networks

## TL;DR
A neural network approximates any function via layers of linear transformations + nonlinear
activations. Trained by backpropagation and gradient descent. Architecture and training
details matter enormously in practice.

## Core Intuition
A pipeline of feature detectors. Early layers detect simple patterns (edges in images, character
n-grams in text). Later layers combine them into complex abstractions (faces, intent). Each
layer transforms its input to make the next layer's job easier.

## How It Works

**Layer:** $h = \sigma(Wx + b)$ where σ is an activation function.

**Activations:**
- ReLU: $\max(0, x)$ — most common. No vanishing gradient for positive inputs.
- Sigmoid: $(1+e^{-x})^{-1}$ — squashes to (0,1). Output layer for binary classification.
- Softmax: multinomial output probabilities. Output layer for multi-class.
- GELU: smooth ReLU variant used in transformers.

**Backprop:** chain rule from output to input. Gradient of loss w.r.t. each weight.

**Initialization:** He init for ReLU ($\sqrt{2/n_{in}}$); Xavier for tanh/sigmoid. Never zeros.

**Batch Normalization:** normalize activations within mini-batch, then scale/shift with learned γ, β.
Stabilizes training, allows higher learning rates.

## Key Properties / Trade-offs
- Residual connections (skip connections): $F(x) + x$ — solve vanishing gradient in deep nets
- Width vs depth: deeper = more abstract representations; wider = more patterns per layer
- Batch norm behavior differs train vs eval — always call `model.eval()` at inference

## Common Mistakes / Gotchas
- Sigmoid/tanh in hidden layers → vanishing gradient. Use ReLU.
- Zero initialization → symmetric weights, all neurons learn identically
- Forgetting `model.eval()` at inference — dropout and batchnorm behave differently
- Exploding gradients → loss NaN → use gradient clipping

## Code Example
```python
import numpy as np

def relu(x): return np.maximum(0, x)
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()

class TwoLayerNet:
    def __init__(self, d_in, d_h, d_out):
        self.W1 = np.random.randn(d_in, d_h) * np.sqrt(2/d_in)  # He init
        self.b1 = np.zeros(d_h)
        self.W2 = np.random.randn(d_h, d_out) * np.sqrt(2/d_h)
        self.b2 = np.zeros(d_out)

    def forward(self, x):
        self.h = relu(x @ self.W1 + self.b1)
        return softmax(self.h @ self.W2 + self.b2)
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Why ReLU over sigmoid in hidden layers?" | ReLU doesn't saturate for positive inputs — gradient always 1, no vanishing. Sigmoid saturates at both ends, killing gradients. |
| "What is batch normalization?" | Normalizes activations within mini-batch to zero mean/unit variance, then scales/shifts with learned params. Stabilizes training. |
| "What is the vanishing gradient problem?" | In deep nets with sigmoid/tanh, gradients shrink exponentially through layers. Fixed by: ReLU, skip connections, batch norm. |

## Related Topics
- [Optimization](optimization.md) — [Regularization](regularization.md) — [CNNs](deep-learning/cnns.md)
- [Implementations: Neural Net From Scratch](../implementations/neural-net-from-scratch.ipynb)

## Resources
- [CS231n Neural Networks](https://cs231n.github.io/neural-networks-1/)
- [Deep Learning book Ch. 6–8](https://www.deeplearningbook.org/)
```

- [ ] **Step 12: Commit ML core concepts**

```bash
git add ml/
git commit -m "docs(ml): add README, resources, and 9 core concept files"
```

---

## Task 5: ML Deep Learning Concepts (4 files)

**Files:** `ml/concepts/deep-learning/cnns.md`, `rnns-lstms.md`, `attention-mechanism.md`, `transformers.md`

- [ ] **Step 1: Create ml/concepts/deep-learning/cnns.md**

```markdown
# Convolutional Neural Networks (CNNs)

## TL;DR
Specialized neural networks for grid-structured data (images, audio spectrograms). Use convolutional
filters with weight sharing to detect the same pattern anywhere in the input. Foundation for
computer vision — ResNet, EfficientNet, and Vision Transformers all build on this.

## Core Intuition
A human recognizes a cat regardless of where in an image it appears. CNNs achieve this via weight
sharing: the same filter applied at every spatial location. Early filters detect edges; later
filters detect textures, parts, and objects.

## How It Works

**Convolution:** filter $W \in \mathbb{R}^{k \times k}$ slides across input with stride s.
Output size: $\lfloor(H - k)/s + 1\rfloor$.

**Padding:** "same" padding preserves spatial dimensions; "valid" reduces them.

**Pooling:** max pooling takes the max in each window — reduces spatial dims, adds translation invariance.

**Standard CNN block:** Conv → BatchNorm → ReLU → MaxPool

**ResNet skip connection:** $F(x) + x$ — learn the residual. Allows 100+ layer networks.
Gradient highway: $F(x) = 0$ is the easy identity path.

## Key Properties / Trade-offs
- Parameter sharing: a 3×3 conv has only 9k params regardless of input spatial size
- Receptive field grows with depth — deeper networks capture larger context
- Depthwise separable convolutions (MobileNet): 8–9× fewer ops than standard conv

## Common Mistakes / Gotchas
- Confusing equivariance (conv) and invariance (pooling)
- Not using batch norm — training very deep CNNs without it is extremely difficult
- Forgetting to normalize inputs to [0,1] or [-1,1]

## Code Example
```python
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Why are CNNs good for images?" | Weight sharing + local connectivity = efficient, translation-equivariant feature detection. Far fewer params than fully connected layers. |
| "What is a skip connection?" | $F(x) + x$ adds input directly to output. Creates gradient highways for backprop; learning identity is trivial, enabling very deep networks. |

## Related Topics
- [Neural Networks](../neural-networks.md) — [Attention Mechanism](attention-mechanism.md)
- [Implementations: CNN Image Classifier](../../implementations/cnn-image-classifier.ipynb)

## Resources
- [CS231n Conv Networks](https://cs231n.github.io/convolutional-networks/)
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) — He et al.
```

- [ ] **Step 2: Create ml/concepts/deep-learning/rnns-lstms.md**

```markdown
# RNNs and LSTMs

## TL;DR
Recurrent Neural Networks process sequential data by maintaining a hidden state across timesteps.
LSTMs solve the vanishing gradient problem with gating mechanisms. Largely superseded by
transformers for long sequences, but still used in streaming and time-series applications.

## Core Intuition
Reading a sentence word by word with a "working memory." The problem: memory decays. LSTMs add a
long-term memory cell with explicit gates controlling what to write, read, and forget.

## How It Works

**Vanilla RNN:** $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$
Problem: gradient of $h_t$ w.r.t. $h_{t-k}$ involves $W_{hh}^k$ — explodes or vanishes.

**LSTM gates** (all use sigmoid → values in [0,1]):
- Forget: $f_t = \sigma(W_f [h_{t-1}, x_t])$ — what to erase from cell state
- Input: $i_t = \sigma(W_i [h_{t-1}, x_t])$ — what new info to write
- Output: $o_t = \sigma(W_o [h_{t-1}, x_t])$ — what to output
- Cell: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
- Hidden: $h_t = o_t \odot \tanh(c_t)$

**GRU:** simplified LSTM with 2 gates (reset, update). Fewer params, similar performance.

## Key Properties / Trade-offs
- Non-parallelizable across timesteps — slow on GPUs
- Transformers dominate for sequences > ~200 tokens
- RNNs still used in: streaming audio, online RL, edge devices

## Common Mistakes / Gotchas
- Vanilla RNNs can't capture long-range dependencies — always use LSTM/GRU
- Not clipping gradients — exploding gradients are common in RNN training
- LSTM has two states: $h_t$ (hidden) and $c_t$ (cell) — both matter

## Code Example
```python
import torch.nn as nn
lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
# input shape: (batch, seq_len, features)
# output: hidden state at each timestep, final hidden + cell state
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What problem does LSTM solve?" | Vanilla RNN suffers vanishing/exploding gradients. LSTM adds a cell state with forget/input/output gates — gradient highways for long-range dependencies. |
| "Why did transformers replace RNNs?" | Transformers process all positions in parallel (no sequential bottleneck), scale better on GPUs, and handle longer context via attention. |
| "LSTM vs GRU?" | GRU merges cell/hidden state, uses 2 gates instead of 3. Fewer parameters, similar performance. |

## Related Topics
- [Attention Mechanism](attention-mechanism.md) — [Transformers](transformers.md)

## Resources
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Colah. Best LSTM explainer.
```

- [ ] **Step 3: Create ml/concepts/deep-learning/attention-mechanism.md**

```markdown
# Attention Mechanism

## TL;DR
Attention lets a model focus on the most relevant parts of an input when producing each output.
Scaled dot-product attention is the core operation in all transformers. Understanding attention
is mandatory for any LLM or agentic AI engineering role.

## Core Intuition
Translating "The bank by the river" — the word "bank" is ambiguous. Attention lets the model
look back at "river" to resolve it. Each position directly attends to all other positions,
weighted by relevance.

## How It Works

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Q (queries):** what this position is looking for
- **K (keys):** what each position offers
- **V (values):** what each position contributes if attended to
- **$\sqrt{d_k}$ scaling:** prevents dot products from growing large (softmax saturation)

**Self-attention:** Q, K, V all from the same sequence.
**Cross-attention:** Q from decoder, K/V from encoder.
**Multi-head:** h heads in parallel, each with different projections. Each head captures different relationship types.
**Causal masking:** mask future positions → each token attends only to past positions. Required for autoregressive generation.

## Key Properties / Trade-offs
- O(T²) memory and compute — quadratic scaling limits context length
- Flash Attention: O(T) memory via tiling (no change to output, just compute order)
- Permutation-invariant — positional encodings are needed to inject order

## Common Mistakes / Gotchas
- Forgetting $\sqrt{d_k}$ scaling → softmax saturation → gradients vanish
- Confusing self-attention (one sequence) and cross-attention (two sequences)
- Missing causal mask in language modeling → model peeks at future tokens

## Code Example
```python
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return w @ V

T, d = 5, 8
out = attention(np.random.randn(T, d), np.random.randn(T, d), np.random.randn(T, d))
print(out.shape)  # (5, 8)
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain scaled dot-product attention." | Compute Q·Kᵀ/√d_k, softmax → attention weights, weighted sum of V. Q=what I want, K=what I offer, V=what I contribute. |
| "Why scale by √d_k?" | Without scaling, large dot products → near one-hot softmax → vanishing gradients. |
| "What is multi-head attention?" | Run h heads in parallel with different learned Q/K/V projections. Concatenate and project. Each head captures different relationship types. |

## Related Topics
- [Transformers](transformers.md) — [Coding: Implement Attention](../../../coding/ml-coding/implement-attention.md)
- [LLM: Context Window](../../../llm/concepts/context-window.md)

## Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
```

- [ ] **Step 4: Create ml/concepts/deep-learning/transformers.md**

```markdown
# Transformers

## TL;DR
The transformer architecture (Vaswani et al., 2017) replaces recurrence with self-attention.
Foundation of all modern LLMs: GPT, BERT, T5, Llama, Claude. Understanding transformers
mechanistically is required for any LLM engineering role.

## Core Intuition
RNNs process tokens sequentially — slow and forgetful over long sequences. Transformers process
all tokens at once via self-attention, letting every token directly attend to every other.
This parallelism enables training on massive datasets and predictable scaling.

## How It Works

**Transformer block:**
```
x → LayerNorm → MultiHeadSelfAttention → Residual add
  → LayerNorm → FFN(GELU) → Residual add → x_out
```
FFN: two linear layers with nonlinearity, width = 4× model dimension.

**Architectures:**
- **Encoder-only (BERT):** bidirectional attention, masked language modeling. Used for classification, embeddings.
- **Decoder-only (GPT/Llama/Claude):** causal attention, next-token prediction. Used for generation.
- **Encoder-Decoder (T5/BART):** for seq2seq tasks (translation, summarization).

**Positional Encoding:**
- Sinusoidal (original paper): $PE(pos, 2i) = \sin(pos/10000^{2i/d})$
- Learned embeddings (GPT-2)
- RoPE (Llama): relative position baked into Q/K rotation — extends to longer contexts

**Scaling laws:** performance scales predictably with params, data, compute.
Chinchilla: ~20 tokens per parameter for compute-optimal training.

## Key Properties / Trade-offs
- O(T²) attention — context length limited by memory
- Pre-LN (LayerNorm before attention) more stable than Post-LN
- Residual connections are critical — not optional

## Common Mistakes / Gotchas
- BERT ≠ GPT: BERT is encoder (bidirectional), GPT is decoder (causal)
- FFN accounts for ~⅔ of parameters, not attention — often forgotten
- Confusing d_model and d_k = d_model/h_heads

## Code Example
```python
import torch, torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d=512, h=8, ff=2048, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0])
        x = x + self.drop(self.ff(self.ln2(x)))
        return x
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain the transformer." | N blocks of: Pre-LN → multi-head self-attention → residual → Pre-LN → FFN → residual. Input needs positional encoding since self-attention is permutation-invariant. |
| "BERT vs GPT?" | BERT: encoder-only, bidirectional, MLM pretraining. For classification/embeddings. GPT: decoder-only, causal, next-token prediction. For generation. |
| "Why do transformers scale better than RNNs?" | Full parallelism → efficient GPU use. Short gradient paths (no sequential bottleneck). Predictable scaling with data + compute. |

## Related Topics
- [Attention Mechanism](attention-mechanism.md) — [LLM: Pretraining](../../../llm/concepts/pretraining.md)
- [Coding: Implement Transformer](../../../coding/ml-coding/implement-transformer.md)

## Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [GPT-3 paper](https://arxiv.org/abs/2005.14165)
```

- [ ] **Step 5: Commit deep learning concepts**

```bash
git add ml/concepts/deep-learning/
git commit -m "docs(ml): add 4 deep learning concept files (CNNs, RNNs, attention, transformers)"
```

---

## Task 6: ML Implementations (8 Notebooks)

**Files:** `ml/implementations/` — 8 .ipynb notebooks, each following Template C.

All notebooks must run cell-by-cell on a fresh kernel with no errors.
Required dependency stack: `numpy`, `scikit-learn`, `matplotlib`, `torch` (for cnn-image-classifier.ipynb).

- [ ] **Step 1: Create linear-regression.ipynb**

7 cells following Template C:

```python
# Cell 1 (Markdown): Header — "Linear Regression From Scratch. Build: normal equation + GD. Prerequisites: NumPy, matrix math."
# Cell 2 (Markdown): Theory — MSE loss, normal equation derivation, GD update rule
# Cell 3 (Code): From-scratch implementation:
import numpy as np, matplotlib.pyplot as plt
np.random.seed(42)
X = np.random.randn(100, 1); y = 3*X.squeeze() + 2 + np.random.randn(100)*0.5
X_b = np.hstack([np.ones((100,1)), X])
# Normal equation
w_ne = np.linalg.inv(X_b.T@X_b) @ X_b.T @ y
# Gradient descent
w = np.zeros(2); lr = 0.1; losses = []
for _ in range(1000):
    grad = (2/100)*X_b.T@(X_b@w - y); w -= lr*grad; losses.append(np.mean((X_b@w-y)**2))
# Cell 4 (Code): sklearn LinearRegression comparison
# Cell 5 (Code): Visualization — scatter + fitted lines, loss curve
# Cell 6 (Markdown + Code): Exercises — add L2 regularization, mini-batch GD, multi-feature
# Cell 7 (Markdown): Summary + link to logistic-regression.ipynb
```

- [ ] **Step 2: Create logistic-regression.ipynb**

```python
# Cell 1 (Markdown): Header — "Logistic Regression. Build: sigmoid + cross-entropy + GD. Prerequisites: linear-regression.ipynb."
# Cell 2 (Markdown): Theory — sigmoid, cross-entropy loss, gradient derivation
# Cell 3 (Code):
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def sigmoid(z): return 1/(1+np.exp(-z))

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
w = np.zeros(2); b = 0.0; lr = 0.1
for _ in range(1000):
    p = sigmoid(X_tr@w + b)
    w -= lr*(1/len(y_tr))*(X_tr.T@(p-y_tr))
    b -= lr*(1/len(y_tr))*np.sum(p-y_tr)
acc = np.mean((sigmoid(X_te@w+b)>=0.5)==y_te)
print(f"Accuracy: {acc:.4f}")
# Cell 4 (Code): sklearn LogisticRegression comparison
# Cell 5 (Code): Decision boundary visualization using contourf
# Cell 6 (Markdown + Code): Exercises — L2 regularization, multiclass one-vs-rest, learning curve
# Cell 7 (Markdown): Summary + link to decision-tree.ipynb
```

- [ ] **Step 3: Create decision-tree.ipynb**

```python
# Cell 1 (Markdown): Header — "Decision Tree. Build: Gini-based binary decision tree. Prerequisites: recursion, numpy."
# Cell 2 (Markdown): Theory — Gini impurity, information gain, stopping criteria
# Cell 3 (Code): Full DecisionTree class with gini(), best_split(), _grow(), predict()
# (copy from ml-coding-questions.md Q1 solution — it is complete and tested)
# Cell 4 (Code): sklearn DecisionTreeClassifier comparison on Iris
# Cell 5 (Code): plot_tree visualization from sklearn
# Cell 6 (Markdown + Code): Exercises — entropy splitting, regression tree, feature importance
# Cell 7 (Markdown): Summary + link to random-forest.ipynb
```

- [ ] **Step 4: Create random-forest.ipynb**

```python
# Cell 1 (Markdown): Header — "Random Forest. Build: bagging over DecisionTree + feature subsampling."
# Cell 2 (Markdown): Theory — bagging, bootstrap sampling, feature subsampling, OOB score
# Cell 3 (Code):
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features='sqrt'):
        self.n_trees = n_trees; self.max_depth = max_depth; self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n, d = X.shape
        n_feat = max(1, int(np.sqrt(d)) if self.max_features=='sqrt' else d)
        for _ in range(self.n_trees):
            idx = np.random.choice(n, n, replace=True)          # bootstrap
            feat_idx = np.random.choice(d, n_feat, replace=False)
            tree = DecisionTree(max_depth=self.max_depth)        # from decision-tree notebook
            tree.fit(X[idx][:, feat_idx], y[idx])
            self.trees.append((tree, feat_idx))

    def predict(self, X):
        preds = np.array([t.predict(X[:, fi]) for t, fi in self.trees])
        return np.apply_along_axis(lambda col: np.bincount(col).argmax(), 0, preds)
# Cell 4 (Code): sklearn RandomForestClassifier comparison
# Cell 5 (Code): Feature importance bar chart + OOB score explanation
# Cell 6 (Markdown + Code): Exercises — vary n_trees and plot variance vs n_trees, implement OOB scoring
# Cell 7 (Markdown): Summary + link to neural-net-from-scratch.ipynb
```

- [ ] **Step 5: Create kmeans-from-scratch.ipynb**

```python
# Cell 1 (Markdown): Header — "K-Means. Build: full K-Means with random init and convergence check."
# Cell 2 (Markdown): Theory — objective (inertia), assignment step, update step, convergence
# Cell 3 (Code):
import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, n_iter=100, tol=1e-4, random_state=42):
    np.random.seed(random_state)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for i in range(n_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)  # (n, k)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels==j].mean(axis=0) for j in range(k)])
        if np.linalg.norm(new_centroids - centroids) < tol:
            print(f"Converged at iteration {i+1}"); break
        centroids = new_centroids
    return labels, centroids

X = np.vstack([np.random.randn(100,2)+c for c in [[0,0],[5,5],[-3,5]]])
labels, centers = kmeans(X, k=3)
# Cell 4 (Code): sklearn KMeans comparison
# Cell 5 (Code): scatter plot of clusters + centroids + elbow curve
# Cell 6 (Markdown + Code): Exercises — K-Means++, silhouette score, DBSCAN comparison
# Cell 7 (Markdown): Summary + links to unsupervised-learning.md and neural-net-from-scratch.ipynb
```

- [ ] **Step 6: Create neural-net-from-scratch.ipynb**

```python
# Cell 1 (Markdown): Header — "Neural Net from Scratch. Build: 2-layer MLP with manual backprop."
# Cell 2 (Markdown): Theory — forward pass, cross-entropy loss, backprop chain rule
# Cell 3 (Code):
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def relu(x): return np.maximum(0, x)
def softmax(x): e=np.exp(x-x.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

class MLP:
    def __init__(self, d_in, d_h, d_out, lr=0.01):
        self.W1 = np.random.randn(d_in, d_h)*np.sqrt(2/d_in)
        self.b1 = np.zeros(d_h)
        self.W2 = np.random.randn(d_h, d_out)*np.sqrt(2/d_h)
        self.b2 = np.zeros(d_out); self.lr = lr

    def forward(self, X):
        self.X=X; self.z1=X@self.W1+self.b1; self.h=relu(self.z1)
        self.z2=self.h@self.W2+self.b2; return softmax(self.z2)

    def backward(self, y_one_hot):
        n=len(y_one_hot); dz2=self.forward(self.X)-y_one_hot
        self.W2-=self.lr*(self.h.T@dz2/n); self.b2-=self.lr*dz2.mean(0)
        dh=dz2@self.W2.T; dz1=dh*(self.z1>0)
        self.W1-=self.lr*(self.X.T@dz1/n); self.b1-=self.lr*dz1.mean(0)

X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                            n_informative=10, random_state=42)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)
net = MLP(20, 64, 3)
y_oh = np.eye(3)[y_tr]
for _ in range(500): net.forward(X_tr); net.backward(y_oh)
preds = net.forward(X_te).argmax(1)
print(f"Test accuracy: {np.mean(preds==y_te):.4f}")
# Cell 4 (Code): sklearn MLPClassifier comparison
# Cell 5 (Code): Training loss curve + decision boundary (2D PCA projection)
# Cell 6 (Markdown + Code): Exercises — add L2 reg, add a third layer, implement momentum
# Cell 7 (Markdown): Summary + link to backpropagation.ipynb
```

- [ ] **Step 7: Create backpropagation.ipynb**

```python
# Cell 1 (Markdown): Header — "Backpropagation Deep Dive. Numerically verify gradients."
# Cell 2 (Markdown): Theory — chain rule, computation graph, gradient flow through each op
# Cell 3 (Code): Implement individual ops (add, mul, relu, matmul) with forward + backward
# Cell 4 (Code): Numerical gradient check via finite differences:
def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy(); x_plus.flat[i] += eps
        x_minus = x.copy(); x_minus.flat[i] -= eps
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2*eps)
    return grad
# Verify: analytical_grad ≈ numerical_grad within 1e-5 relative error
# Cell 5 (Code): Train 3-layer MLP on small dataset, verify convergence
# Cell 6 (Markdown + Code): Exercises — implement sigmoid backward, cross-entropy backward, batch norm backward
# Cell 7 (Markdown): Summary + link to cnn-image-classifier.ipynb
```

- [ ] **Step 8: Create cnn-image-classifier.ipynb**

```python
# Cell 1 (Markdown): Header — "CNN Image Classifier. Build and train CNN on MNIST using PyTorch."
# Cell 2 (Markdown): Theory — convolutional layers, receptive field, pooling, why CNNs for images
# Cell 3 (Code):
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
test_data  = datasets.MNIST('.', train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(64*16, 128), nn.ReLU(), nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

model = CNN(); opt = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
for epoch in range(3):
    for X, y in train_loader:
        opt.zero_grad(); loss = criterion(model(X), y); loss.backward(); opt.step()
    # eval
    correct = sum((model(X).argmax(1)==y).sum().item() for X,y in test_loader)
    print(f"Epoch {epoch+1}: {correct/len(test_data):.4f}")
# Cell 4 (Code): Library comparison — torchvision pretrained ResNet on same task
# Cell 5 (Code): Visualize first conv layer filters + feature maps for a sample image
# Cell 6 (Markdown + Code): Exercises — add data augmentation, try 3 conv layers, experiment with dropout
# Cell 7 (Markdown): Summary + links to attention-mechanism.md, llm/ folder
```

- [ ] **Step 9: Commit notebooks**

```bash
git add ml/implementations/
git commit -m "docs(ml): add 8 implementation notebooks (linear/logistic regression, decision tree, random forest, k-means, neural net, backprop, CNN)"
```

---

## Task 7: ML Interview Prep Files

**Files:** `ml/interview-prep/README.md`, `ml-theory-questions.md`, `ml-coding-questions.md`, `case-studies.md`

- [ ] **Step 1: Create ml/interview-prep/README.md**

```markdown
# ML Interview Prep

How to use:
1. **Simulate the interview** — cover the question, draft your answer, then reveal.
2. **Time yourself** — theory: 3–5 min. Coding: 20–30 min. Case studies: 45 min.
3. **Work all 6 steps** — don't skip to the answer.

Files:
- [ML Theory Questions](ml-theory-questions.md) — 50 Q&A on all ML fundamentals
- [ML Coding Questions](ml-coding-questions.md) — implement algorithms from scratch
- [ML Case Studies](case-studies.md) — end-to-end 45-minute system design scenarios
```

- [ ] **Step 2: Create ml/interview-prep/ml-theory-questions.md**

Create this file with 50 questions in Template B format (all 6 steps per question).
The first 5 questions are fully written below. Questions 6–50 must follow the same format.

Full content of the first 5 questions:

```markdown
# ML Theory Interview Questions

---

## Q: What is the bias-variance trade-off?

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, Meta, Amazon, Microsoft

### Step 1 — Clarifying Questions to Ask
- "Are you asking in general, or about a specific model or task?"
- "Should I include the mathematical decomposition?"

### Step 2 — Approach Discussion
Start with intuition (U-shaped test error curve), then formalize the decomposition.
Connect to practical actions at the end.

### Step 3 — Answer
**Bias** = error from wrong assumptions (underfitting). Model too simple to capture patterns.
**Variance** = sensitivity to training data fluctuations (overfitting). Model fits noise.

Expected test error decomposes as:
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}(\hat{f})^2 + \text{Var}(\hat{f}) + \sigma^2_\text{noise}$$

Noise is irreducible. As model complexity increases: bias ↓, variance ↑.
Optimal complexity minimizes their sum.

**Practical actions:**
- High bias (underfitting): add features, increase model capacity, reduce regularization
- High variance (overfitting): more data, regularize, reduce capacity, use ensemble methods

### Step 4 — Test Cases
N/A (theory question)

### Step 5 — Complexity Analysis
N/A

### Step 6 — Follow-up Questions
- "How do ensemble methods change the trade-off?"
  → Bagging reduces variance without increasing bias. Boosting reduces bias (sequentially).
- "Does more data help bias or variance?"
  → Primarily variance. Bias requires changing the model, not adding data.

### Common Mistakes
- Saying "complex models have high bias" — it's the opposite
- Not connecting to practical actions (what do you do in each case?)
- Forgetting noise is irreducible

---

## Q: Explain gradient descent and its variants.

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, Meta, OpenAI, DeepMind

### Step 1 — Clarifying Questions to Ask
- "High-level intuition, or full mathematical derivation?"
- "Should I cover Adam and adaptive methods?"

### Step 2 — Approach Discussion
Cover vanilla GD first, then frame each variant as a solution to a specific failure mode.

### Step 3 — Answer
**Core:** $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$

Three variants by batch size:
- **Batch GD:** full dataset per update. Exact gradient, slow on large datasets.
- **SGD:** one sample. Noisy, fast, can escape saddle points.
- **Mini-batch:** B samples (32–256). Best trade-off. Standard in practice.

**Momentum:** velocity term smooths oscillations and accelerates along consistent directions.

**Adam:** adapts per-parameter learning rates using first moment (mean) + second moment (variance) of gradients. Fastest convergence. Defaults: lr=1e-3, β₁=0.9, β₂=0.999.

**LR scheduling:** cosine annealing or warmup+decay. Warmup is essential for transformers.

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
Per update: Batch O(n·d); SGD O(d); Mini-batch O(B·d)

### Step 6 — Follow-up Questions
- "Why does SGD sometimes outperform Adam?"
  → SGD noise → flatter minima → better generalization than Adam's sharp minima.
- "What is learning rate warmup?"
  → Gradually increase lr from 0 for first N steps. Prevents instability when params are far from optimum.

### Common Mistakes
- Not distinguishing batch/SGD/mini-batch
- Claiming Adam is always better — it converges faster but doesn't always generalize as well
- Forgetting gradient clipping for RNNs/transformers

---

## Q: How does cross-validation work and when do you use it?

**Difficulty:** Easy | **Domain:** ML Theory | **Companies:** Any ML company

### Step 1 — Clarifying Questions to Ask
- "Should I cover k-fold specifically or all variants?"

### Step 2 — Approach Discussion
Problem → solution → variants → code.

### Step 3 — Answer
**Problem:** single train/test split gives a noisy estimate of generalization.

**K-Fold CV:** split data into k folds. Train on k-1, evaluate on remaining fold. Repeat k times.
Average k scores. Reduces evaluation variance by factor of k.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
scores = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring='roc_auc')
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Stratified k-fold:** preserves class proportions per fold. Always use for classification.

**When to use:** hyperparameter tuning, limited data, reliable generalization estimates.

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
O(k · training_time) — k times more expensive than a single train.

### Step 6 — Follow-up Questions
- "CV vs train/val/test split?" → Use train/val/test with enough data. CV for limited data.
- "What is nested CV?" → Outer loop evaluates generalization; inner loop tunes hyperparameters.

### Common Mistakes
- Non-stratified k-fold for imbalanced classification
- Fitting a scaler on full dataset before CV — data leakage
- Reporting only mean, not std across folds

---

## Q: How does L1 regularization produce sparse weights?

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, Meta

### Step 1 — Clarifying Questions to Ask
- "Do you want the geometric explanation or the sub-gradient explanation?"

### Step 2 — Approach Discussion
Both explanations are useful — geometric is intuitive, sub-gradient is rigorous.

### Step 3 — Answer
**L1 penalty:** $\lambda \sum_j |w_j|$

**Geometric:** L1 constrains parameters to a diamond (L1 ball). The optimal point is where
loss contours touch the ball — at a corner, where weights are zero. L2's spherical ball
has no corners — optimal point lands on the surface but rarely at zero.

**Sub-gradient:** at $w_j=0$, the L1 sub-gradient is in $[-\lambda, \lambda]$. If the loss
gradient magnitude is less than $\lambda$, the optimal condition holds at $w_j=0$ — weight stays zero.

**Practical implication:** L1 performs automatic feature selection. Features that don't contribute
enough to reduce the loss get zeroed out.

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
N/A

### Step 6 — Follow-up Questions
- "When L1 over L2?" → When you want feature selection / sparse interpretable model.
- "What is Elastic Net?" → L1 + L2 combined. Handles groups of correlated features better than L1 alone.

### Common Mistakes
- Saying only "L1 gives sparsity" without explaining why geometrically
- Forgetting L1 is non-differentiable at 0 — this is precisely what produces sparsity

---

## Q: Explain the EM algorithm with an example.

**Difficulty:** Hard | **Domain:** ML Theory | **Companies:** Google, DeepMind

### Step 1 — Clarifying Questions to Ask
- "Should I use Gaussian Mixture Models as the concrete example?"

### Step 2 — Approach Discussion
Abstract framework first, then GMM as the grounding example. Connect to K-Means.

### Step 3 — Answer
EM finds MLE when latent (hidden) variables exist and direct optimization is intractable.

**E-step:** compute expected log-likelihood given current parameters and observed data.
For GMM: compute soft cluster assignments (responsibilities) for each point.

**M-step:** update parameters to maximize the expected log-likelihood from E-step.
For GMM: update cluster means, covariances, and mixing weights using responsibilities.

```python
from sklearn.mixture import GaussianMixture
import numpy as np

X = np.vstack([np.random.randn(100,2)+[0,0], np.random.randn(100,2)+[5,5]])
gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
print(gmm.means_)  # learned cluster centers
```

**Convergence:** EM is guaranteed to non-decrease the marginal log-likelihood each iteration.
Converges to a local maximum — initialization matters.

**K-Means = EM** with hard cluster assignments (responsibilities are 0 or 1) and isotropic Gaussians.

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
Per iteration: O(n·k·d²) for GMM with k components, d features.

### Step 6 — Follow-up Questions
- "Limitations of EM?" → Local optima (use multiple restarts), slow convergence, must choose k.
- "Why is K-Means a special case?" → Hard assignments + isotropic Gaussians = K-Means objective.

### Common Mistakes
- Not knowing a concrete example (GMM)
- Claiming EM converges to a global maximum — it's only local

---

*Questions 6–50 follow the same Template B format. Remaining topics to cover:*
*SVMs (kernel trick, margin maximization, support vectors), PCA (eigenvector derivation, when to use),*
*Naive Bayes (conditional independence, Laplace smoothing), Random Forest (how OOB works, feature importance),*
*Batch Normalization (train vs inference behavior), Dropout (ensemble interpretation),*
*transfer learning (feature extraction vs fine-tuning), multi-task learning, class imbalance strategies,*
*A/B testing statistical power, NDCG and ranking metrics, and more.*
```

- [ ] **Step 3: Create ml/interview-prep/ml-coding-questions.md**

```markdown
# ML Coding Interview Questions

Implement from scratch — NumPy only for the core implementation.

---

## Q: Implement K-Nearest Neighbors (KNN) from scratch.

**Difficulty:** Medium | **Domain:** ML Coding | **Companies:** Google, Meta, Amazon

### Step 1 — Clarifying Questions to Ask
- "What distance metric? Euclidean?"
- "Classification or regression?"
- "Should I make k configurable?"

### Step 2 — Approach Discussion
For each test point: compute distance to all training points, find k nearest, majority vote.
Brute force O(n·d) per query. Mention KD-tree as O(d log n) optimization.

### Step 3 — Implementation
```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3): self.k = k

    def fit(self, X, y): self.X_tr = X; self.y_tr = y

    def predict(self, X): return np.array([self._pred(x) for x in X])

    def _pred(self, x):
        dists = np.sqrt(np.sum((self.X_tr - x)**2, axis=1))
        k_idx = np.argsort(dists)[:self.k]
        return Counter(self.y_tr[k_idx]).most_common(1)[0][0]

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNN(k=5); knn.fit(X_tr, y_tr)
print(f"Accuracy: {np.mean(knn.predict(X_te)==y_te):.4f}")
```

### Step 4 — Test Cases
| Input | Expected | Why |
|---|---|---|
| k=1, test=train | 100% accuracy | k=1 memorizes training data |
| k > n_train | clip k to n_train | Edge case — handle gracefully |
| unnormalized features | poor accuracy | Distance-based — scale matters |

### Step 5 — Complexity Analysis
**Time:** fit O(1); predict O(n·d) per query, O(m·n·d) for m test points
**Space:** O(n·d) to store training data

### Step 6 — Follow-up Questions
- "Make it faster for large datasets?" → KD-tree O(d log n), Ball tree, FAISS/HNSW for ANN
- "Curse of dimensionality?" → High dimensions: all points equidistant — distance loses meaning. Degrades for d > ~20.
- "Regression KNN?" → Predict mean (or distance-weighted mean) of k nearest neighbors.

### Common Mistakes
- Unvectorized distance loop — always vectorize: `(X_tr - x)**2` broadcasts
- Not normalizing features — scale affects Euclidean distance

---

## Q: Implement scaled dot-product attention from scratch.

**Difficulty:** Hard | **Domain:** ML Coding | **Companies:** OpenAI, Google, Anthropic, Meta AI

### Step 1 — Clarifying Questions to Ask
- "Should I implement single-head or multi-head?"
- "Do you need causal masking?"

### Step 2 — Approach Discussion
$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$
Walk through each step: dot products, scaling, (optional masking), softmax, weighted sum.

### Step 3 — Implementation
```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)   # (..., T_q, T_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)                 # (..., T_q, T_k)
    return weights @ V                                 # (..., T_q, d_v)

# Test single head
T, d_k, d_v = 6, 8, 8
Q = np.random.randn(T, d_k)
K = np.random.randn(T, d_k)
V = np.random.randn(T, d_v)
out = attention(Q, K, V)
assert out.shape == (T, d_v)

# Test causal mask
causal_mask = np.tril(np.ones((T, T), dtype=bool))
out_causal = attention(Q, K, V, mask=causal_mask)
print(out.shape, out_causal.shape)
```

### Step 4 — Test Cases
| Input | Expected | Why |
|---|---|---|
| Q=K=V, identity | Each position attends to itself | Self-similarity |
| Causal mask | Upper triangle weights ≈ 0 | Future tokens masked |
| d_k very large, no scaling | Near-zero gradients | Softmax saturation without √d_k |

### Step 5 — Complexity Analysis
**Time:** O(T²·d_k) for scores, O(T²·d_v) for output — O(T²d) total
**Space:** O(T²) for attention matrix (Flash Attention reduces this to O(T))

### Step 6 — Follow-up Questions
- "Implement multi-head attention?" → Split d_model into h heads of d_k=d_model/h, run in parallel, concat, project.
- "What is Flash Attention?" → Recompute attention in tiles to avoid materializing the full T×T matrix. O(T) memory.
- "Why does softmax saturate without scaling?" → Large d_k → large dot products → near one-hot softmax → gradients ≈ 0.

### Common Mistakes
- Forgetting to subtract max before softmax (numerical stability)
- Using `np.exp` directly on large values → overflow
- Mask values of 0 vs -inf: use a large negative number (-1e9), not 0 (0 → exp(0)=1, not masked)

---

*Questions 3–20 follow the same format. Topics: implement softmax + cross-entropy from scratch,
implement PCA via eigendecomposition, implement K-Means from scratch, implement mini-batch SGD
on a neural net, implement gradient descent with momentum, implement beam search (LLM decoding).*
```

- [ ] **Step 4: Create ml/interview-prep/case-studies.md**

```markdown
# ML Case Studies

End-to-end ML system design. Simulate a 45-minute interview.

---

## Case Study: Content Recommendation System

**Scenario:** Senior ML engineer at a streaming platform. Design the ML system that recommends
the next piece of content to each user.

### Step 1 — Clarifying Questions to Ask
- "What scale? DAU, catalog size, latency SLA?"
- "Business metric: clicks, watch time, or retention?"
- "Real-time personalization or precomputed?"
- "Cold-start users?"

*Assumed: 50M DAU, 10M items, <100ms response, optimize watch time, real-time for existing users.*

### Step 2 — ML Problem Formulation
- **Task:** ranking — given user u, rank candidates by predicted watch time
- **Labels:** implicit (watched ≥ X min = positive, skip = negative)
- **Output:** relevance score per (user, item) pair

### Step 3 — System Design

**Candidate Generation (offline, batch):**
- Two-tower model: user embedding × item embedding → cosine similarity → top 500 candidates
- Rerun nightly; triggered by large user activity shift

**Ranking (online, <100ms):**
- Features: user embedding, item embedding, user-item interactions, context (time, device)
- Model: LightGBM or small 3-layer MLP (low latency)
- Output: ranked top-20 items

**Feature Store:**
- User features (30d history, demographics) → Redis (precomputed)
- Item features (genre, popularity, embeddings) → Redis
- Session features (last 3 items watched) → Flink stream processing

**Training Pipeline:**
- Event logs → data warehouse → feature engineering → weekly retrain
- Offline metrics: AUC, NDCG@20
- Online: A/B test (5% traffic), primary metric = watch time per session

**Cold Start:**
- New users: popularity + content-based from onboarding preferences
- New items: content-based embedding until sufficient interaction data

### Step 4 — Key Trade-offs
- Two-tower vs MF: two-tower supports richer features but more complex to train/serve
- Online vs offline ranking: online = fresher, but latency risk; offline = fast, but stale
- Explore vs exploit: ε-greedy or UCB bandit for content exploration

### Step 5 — Failure Modes and Mitigations
- **Filter bubble:** homogeneous recommendations → add diversity constraint (MMR), exploration budget
- **Popularity bias:** popular items dominate → add popularity feature + regularize, or debias labels
- **Feedback loop:** model reinforces itself → counterfactual logging, propensity scoring

### Step 6 — Follow-up Questions
- "New user with no history?" → Content-based (item features) + global popularity + onboarding quiz
- "Model degrading in production?" → Monitor online metrics vs baseline; feature distribution drift alerts
- "Reduce latency from 100ms to 30ms?" → Precompute more offline, ANN for retrieval (FAISS), cache user embeddings

---

*Additional case studies (same format):*
*- Fraud Detection at 1M transactions/minute*
*- Search Query Understanding and Ranking*
```

- [ ] **Step 5: Commit ML interview prep**

```bash
git add ml/interview-prep/
git commit -m "docs(ml): add interview prep — README, theory questions, coding questions, case studies"
```

---
