# Repository Design: ML/AI Interview Prep & Learning Hub

**Date:** 2026-05-16
**Status:** Approved
**Scope:** Full repository structure, content standards, roadmaps, coding section, navigation

---

## Overview

A comprehensive, open-source reference repository covering AI, ML, LLM, agentic systems, system design, and coding questions. Serves two equal goals: structured learning curriculum and interview preparation. Primary audience is the public — students preparing for ML/AI interviews and practitioners leveling up.

---

## 1. Repository Structure

Option C (Domain-First with Layered Sub-structure) was selected. Each domain folder has a consistent internal layout: `concepts/`, `implementations/`, `interview-prep/`, and optionally `system-design/`. A shared `roadmaps/` folder and a shared `coding/` folder live at the top level.

```
interviewprep-ml/
│
├── README.md
├── CONTRIBUTING.md
│
├── roadmaps/
│   ├── README.md
│   ├── ml-roadmap.md
│   ├── llm-roadmap.md
│   ├── agentic-roadmap.md
│   └── system-design-roadmap.md
│
├── ml/
│   ├── README.md
│   ├── concepts/
│   │   ├── supervised-learning.md
│   │   ├── unsupervised-learning.md
│   │   ├── neural-networks.md
│   │   ├── optimization.md
│   │   ├── regularization.md
│   │   ├── evaluation-metrics.md
│   │   ├── feature-engineering.md
│   │   ├── ensemble-methods.md
│   │   ├── probability-statistics.md
│   │   └── deep-learning/
│   │       ├── cnns.md
│   │       ├── rnns-lstms.md
│   │       ├── attention-mechanism.md
│   │       └── transformers.md
│   ├── implementations/
│   │   ├── linear-regression.ipynb
│   │   ├── logistic-regression.ipynb
│   │   ├── decision-tree.ipynb
│   │   ├── random-forest.ipynb
│   │   ├── kmeans-from-scratch.ipynb
│   │   ├── neural-net-from-scratch.ipynb
│   │   ├── backpropagation.ipynb
│   │   └── cnn-image-classifier.ipynb
│   ├── interview-prep/
│   │   ├── README.md
│   │   ├── ml-theory-questions.md
│   │   ├── ml-coding-questions.md
│   │   └── case-studies.md
│   └── resources.md
│
├── llm/
│   ├── README.md
│   ├── concepts/
│   │   ├── tokenization.md
│   │   ├── pretraining.md
│   │   ├── finetuning.md
│   │   ├── prompting.md
│   │   ├── rag.md
│   │   ├── embeddings.md
│   │   ├── context-window.md
│   │   ├── quantization.md
│   │   ├── inference-optimization.md
│   │   ├── evaluation.md
│   │   └── multimodal.md
│   ├── implementations/
│   │   ├── build-rag-pipeline.ipynb
│   │   ├── finetune-llm.ipynb
│   │   ├── prompt-engineering.ipynb
│   │   ├── embeddings-search.ipynb
│   │   ├── llm-evals.ipynb
│   │   └── structured-output.ipynb
│   ├── system-design/
│   │   ├── rag-system-design.md
│   │   ├── llm-serving-design.md
│   │   ├── fine-tuning-pipeline.md
│   │   └── llm-observability.md
│   ├── interview-prep/
│   │   ├── llm-theory-questions.md
│   │   ├── llm-system-design-questions.md
│   │   └── prompting-questions.md
│   └── resources.md
│
├── agentic-ai/
│   ├── README.md
│   ├── concepts/
│   │   ├── what-is-an-agent.md
│   │   ├── tool-use.md
│   │   ├── memory-types.md
│   │   ├── planning-reasoning.md
│   │   ├── multi-agent-systems.md
│   │   ├── agent-evals.md
│   │   └── safety-alignment.md
│   ├── implementations/
│   │   ├── basic-agent-loop.ipynb
│   │   ├── tool-calling-agent.ipynb
│   │   ├── rag-agent.ipynb
│   │   ├── multi-agent-workflow.ipynb
│   │   ├── langgraph-agent.ipynb
│   │   └── memory-agent.ipynb
│   ├── system-design/
│   │   ├── agentic-system-design.md
│   │   ├── multi-agent-orchestration.md
│   │   └── production-agents.md
│   ├── interview-prep/
│   │   ├── agentic-theory-questions.md
│   │   └── agentic-system-design-questions.md
│   └── resources.md
│
├── system-design/             # No implementations/ or resources.md — uses patterns/ and case-studies/ instead
│   ├── README.md
│   ├── patterns/
│   │   ├── feature-store.md
│   │   ├── model-registry.md
│   │   ├── online-vs-batch-inference.md
│   │   ├── data-pipelines.md
│   │   ├── ab-testing.md
│   │   └── mlops-overview.md
│   ├── case-studies/
│   │   ├── recommendation-system.md
│   │   ├── search-ranking.md
│   │   ├── fraud-detection.md
│   │   ├── content-moderation.md
│   │   └── ads-click-prediction.md
│   └── interview-prep/
│       ├── system-design-framework.md
│       └── system-design-questions.md
│
└── coding/
    ├── README.md
    ├── data-structures/
    │   ├── arrays-strings.md
    │   ├── linked-lists.md
    │   ├── trees-graphs.md
    │   ├── heaps.md
    │   ├── hashmaps.md
    │   └── tries.md
    ├── algorithms/
    │   ├── sorting.md
    │   ├── binary-search.md
    │   ├── dynamic-programming.md
    │   ├── backtracking.md
    │   ├── sliding-window.md
    │   ├── two-pointers.md
    │   └── graph-traversal.md
    └── ml-coding/
        ├── implement-knn.md
        ├── implement-kmeans.md
        ├── implement-gradient-descent.md
        ├── implement-attention.md
        └── implement-transformer.md
```

---

## 2. Content Standards

Three file templates are used throughout the repo. All contributors must follow them.

### Template A: Concept Note (`.md`)

Sections in order:
1. **TL;DR** — one-paragraph summary
2. **Core Intuition** — plain-English, analogy-first, before any math
3. **How It Works** — theory, math (LaTeX), diagrams (Mermaid or ASCII)
4. **Key Properties / Trade-offs** — bullet list
5. **Common Mistakes / Gotchas** — what people get wrong in interviews
6. **Code Example** — minimal runnable Python snippet
7. **Interview Quick-Reference** — table of question types and what to say
8. **Related Topics** — links to other files in this repo
9. **Resources** — curated external links

### Template B: Full Interview Simulation (`.md`)

Each question block contains:
1. **Question** — phrased as an interviewer would ask it
2. **Metadata** — Difficulty, Domain, Companies known to ask
3. **Step 1 — Clarifying Questions to Ask**
4. **Step 2 — Approach Discussion**
5. **Step 3 — Answer / Solution** (with code if applicable)
6. **Step 4 — Test Cases** (for coding questions)
7. **Step 5 — Complexity Analysis** — Time and Space
8. **Step 6 — Follow-up Questions**
9. **Common Mistakes**

### Template C: Implementation Notebook (`.ipynb`)

Standard cell sequence:
1. Header cell — topic, what you'll build, prerequisites
2. Concept recap — 3–5 markdown cells
3. Minimal from-scratch implementation — pure Python/NumPy
4. Library implementation — sklearn/PyTorch/HuggingFace for comparison
5. Visualization cell — matplotlib/plotly output
6. Exercises cell — 2–3 reader challenges
7. Summary + links

### Naming Conventions

| Type | Convention | Example |
|---|---|---|
| Concept notes | `kebab-case.md` | `attention-mechanism.md` |
| Interview Q&A | `<domain>-questions.md` | `llm-theory-questions.md` |
| Notebooks | `verb-topic.ipynb` | `implement-attention.ipynb` |
| Roadmaps | `<domain>-roadmap.md` | `agentic-roadmap.md` |

---

## 3. Roadmaps Design

Each roadmap follows the same structure:

```
# [Domain] Roadmap
## Who This Is For
## Phase 1 — Foundations (Beginner)
  Goal, estimated time, checklist of files, practice Q range
## Phase 2 — Core Depth (Intermediate)
## Phase 3 — Advanced + Production
## Interview Readiness Checklist
## Suggested Weekly Schedule
```

### Per-Domain Phase Breakdown

**ML Roadmap:**
- Phase 1: Linear/logistic regression, bias-variance, evaluation metrics, probability basics
- Phase 2: Ensemble methods, SVMs, neural nets, backprop, CNNs, RNNs
- Phase 3: Transformers, optimization deep-dives, distributed training, MLOps

**LLM Roadmap:**
- Phase 1: Tokenization, attention, transformer architecture, prompting basics
- Phase 2: RAG, embeddings, fine-tuning (SFT/LoRA), evaluation frameworks
- Phase 3: RLHF/DPO, inference optimization, LLM system design, production concerns

**Agentic AI Roadmap:**
- Phase 1: What is an agent, tool calling, basic ReAct loop, memory types
- Phase 2: Multi-agent patterns, planning/reasoning strategies, LangGraph/raw API
- Phase 3: Production agents, evals, safety, observability, human-in-the-loop

**System Design Roadmap:**
- Phase 1: ML system design framework, feature stores, online vs batch inference
- Phase 2: Case studies (recommendation, search ranking, fraud detection)
- Phase 3: LLM system design, agentic system design, full mock interviews

---

## 4. Coding Section Design

`coding/` is a top-level shared folder (not inside any domain) because DSA is domain-agnostic.

### DSA Coverage

| File | Key problems |
|---|---|
| `arrays-strings.md` | Two Sum, Sliding Window Maximum, Longest Substring Without Repeating |
| `linked-lists.md` | Reverse LL, Detect Cycle, Merge K Sorted Lists |
| `trees-graphs.md` | BFS/DFS, Lowest Common Ancestor, Course Schedule, Word Ladder |
| `heaps.md` | Top K Elements, Merge K Lists, Median from Stream |
| `hashmaps.md` | Group Anagrams, LRU Cache, Subarray Sum Equals K |
| `tries.md` | Word Search II, Autocomplete |
| `sorting.md` | QuickSort, MergeSort, counting sort |
| `binary-search.md` | Search in Rotated Array, Find Peak Element, Binary Search on Answer |
| `dynamic-programming.md` | Knapsack, LCS, Edit Distance, Coin Change |
| `backtracking.md` | N-Queens, Permutations, Sudoku Solver |
| `sliding-window.md` | Pattern recognition + template |
| `two-pointers.md` | Pattern recognition + template |
| `graph-traversal.md` | Topological Sort, Union-Find, Dijkstra |

### ML Coding Coverage

| File | Implementation | Why asked |
|---|---|---|
| `implement-knn.md` | KNN from scratch + KD-tree | Distance metrics, lazy learning |
| `implement-kmeans.md` | K-Means with convergence | EM intuition, cluster assignment |
| `implement-gradient-descent.md` | SGD, mini-batch, momentum, Adam | Core optimization literacy |
| `implement-attention.md` | Scaled dot-product attention | Asked at every LLM company |
| `implement-transformer.md` | Multi-head attention + FFN block | End-to-end transformer layer |

Each ML coding file includes NumPy-only implementation (interview standard) + PyTorch comparison.

### `coding/README.md` progression paths:
- Beginner: arrays → hashmaps → linked-lists → binary-search
- Intermediate: trees → heaps → sliding-window → two-pointers
- Advanced: graphs → DP → backtracking → ml-coding

---

## 5. Navigation, Discovery & Community

### Root `README.md`

Answers in under 30 seconds: what is this, is it for me, where do I start.

Sections:
1. One-line description + tagline
2. "What's Inside" — 4-cell domain grid
3. "Who Is This For" — bullet list of target users
4. "Start Here" — table mapping goals to roadmap links
5. Quick stats badges (topics, notebooks, interview questions, contributors)
6. Contributing link

### `CONTRIBUTING.md`

- What to contribute (domain gap list)
- Links to the 3 file templates
- Naming conventions table
- PR checklist (TL;DR present, template followed, cross-links added, code runs)
- Content quality bar definition

### Discoverability

- GitHub Topics: `machine-learning`, `llm`, `interview-prep`, `system-design`, `deep-learning`, `agents`, `rag`, `python`
- Internal cross-linking: every concept note links to related topics; every roadmap item links to its file; every notebook links back to its concept note
- Phase 2 (future): GitHub Pages via MkDocs or Docusaurus for rendered navigation

---

## Decisions Made

| Decision | Choice | Reason |
|---|---|---|
| Repo structure | Domain-first with layered sub-structure | Scales to 500+ files without reorganization |
| Content format | Mixed: Markdown + Jupyter | Theory in .md, hands-on in .ipynb |
| Interview format | Full simulation (all 6 steps) | Public repo standard; mimics real interviews |
| LLM/Agentic depth | Full stack: theory to production | Covers what top companies actually ask |
| Roadmap style | Per-domain with 3 phases | Explicit paths for different starting points |
| DSA placement | Shared top-level `coding/` | Domain-agnostic; applies across all domains |
