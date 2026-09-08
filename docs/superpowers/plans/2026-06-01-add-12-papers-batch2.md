# Add 12 More Papers (Batch 2) Implementation Plan

> **Status:** Complete in the repository. The 12 paper concept/notebook pairs,
> documentation updates, and remote delivery are present; the task checkboxes
> below are retained as the original execution record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 12 new papers across 3 new domains (safety-alignment, code-systems, reasoning-search) and 1 enhanced domain (efficiency-scaling), expanding the Papers section from 15 to 27 papers.

**Architecture:** Create 3 new domain directories with concepts/ and notebooks/ subdirectories. Each domain gets 2-3 papers in 8-section markdown + 12-cell notebook format. Enhance existing efficiency-scaling domain with 3 additional papers. Update papers/README.md, roadmaps/papers-roadmap.md, and main README.md to reflect new coverage.

**Tech Stack:** Jupyter notebooks, Python (torch, transformers, numpy, matplotlib), Markdown

---

## File Structure Overview

**New Directories:**
```
papers/safety-alignment/          (3 papers)
papers/code-systems/              (3 papers)
papers/reasoning-search/          (3 papers)
papers/efficiency-scaling/        (enhanced: 1→4 papers)
```

**Files Created:** 18 new files
- 9 markdown concept files (safety-alignment: 3, code-systems: 3, reasoning-search: 3)
- 9 Jupyter notebooks (same split)
- Plus 6 files for efficiency-scaling enhancement (3 markdown + 3 notebooks)

**Files Modified:** 3 documentation files

---

## Task 1: Create Directory Structure

**Files:**
- Create: `papers/safety-alignment/concepts/`, `papers/safety-alignment/notebooks/`
- Create: `papers/code-systems/concepts/`, `papers/code-systems/notebooks/`
- Create: `papers/reasoning-search/concepts/`, `papers/reasoning-search/notebooks/`

- [ ] **Step 1: Create all new domain directories**

```bash
mkdir -p papers/safety-alignment/{concepts,notebooks}
mkdir -p papers/code-systems/{concepts,notebooks}
mkdir -p papers/reasoning-search/{concepts,notebooks}
```

- [ ] **Step 2: Verify structure**

```bash
find papers -maxdepth 2 -type d | grep -E "(safety-alignment|code-systems|reasoning-search)"
```

Expected: 6 directories created (3 domains × 2 subdirs each)

- [ ] **Step 3: Create .gitkeep files**

```bash
touch papers/safety-alignment/concepts/.gitkeep papers/safety-alignment/notebooks/.gitkeep
touch papers/code-systems/concepts/.gitkeep papers/code-systems/notebooks/.gitkeep
touch papers/reasoning-search/concepts/.gitkeep papers/reasoning-search/notebooks/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add papers/safety-alignment papers/code-systems papers/reasoning-search
git commit -m "build: create 3 new paper domains (safety-alignment, code-systems, reasoning-search)"
```

---

## Task 2: Safety & Alignment Papers (3 papers)

**Files:**
- Create: `papers/safety-alignment/concepts/01-constitutional-ai.md`
- Create: `papers/safety-alignment/concepts/02-dpo.md`
- Create: `papers/safety-alignment/concepts/03-rlhf-instructgpt.md`
- Create: `papers/safety-alignment/notebooks/01-constitutional-ai.ipynb`
- Create: `papers/safety-alignment/notebooks/02-dpo.ipynb`
- Create: `papers/safety-alignment/notebooks/03-rlhf-instructgpt.ipynb`

[This task will be delegated to subagent. Each markdown file: 1500-2500 words, 8 sections. Each notebook: 12 cells, 600+ code lines. Focus on alignment techniques, safety mechanisms, preference optimization, RLHF mechanisms, production considerations.]

- [ ] **Step 1: Subagent creates Constitutional AI paper** (markdown + notebook)
  - Markdown: ~2200 words covering AI feedback generation, harmlessness criteria, scalable oversight
  - Notebook: 12 cells with Constitutional AI prompts, feedback generation, alignment mechanisms

- [ ] **Step 2: Subagent creates DPO paper** (markdown + notebook)
  - Markdown: ~2300 words covering preference pairs, direct optimization, comparison to RLHF
  - Notebook: 12 cells with preference pair generation, DPO loss implementation, training examples

- [ ] **Step 3: Subagent creates RLHF/InstructGPT paper** (markdown + notebook)
  - Markdown: ~2400 words covering supervised fine-tuning, reward modeling, RL training
  - Notebook: 12 cells with SFT examples, reward model training, RLHF loop implementation

- [ ] **Step 4: Verify all files created**

```bash
ls -la papers/safety-alignment/concepts/*.md papers/safety-alignment/notebooks/*.ipynb
```

Expected: 3 markdown files + 3 notebooks

- [ ] **Step 5: Commit**

```bash
git add papers/safety-alignment/
git commit -m "feat: add Safety & Alignment papers (Constitutional AI, DPO, RLHF)"
```

---

## Task 3: Code & Systems Papers (3 papers)

**Files:**
- Create: `papers/code-systems/concepts/01-codet5.md`
- Create: `papers/code-systems/concepts/02-codex.md`
- Create: `papers/code-systems/concepts/03-code-evaluation.md`
- Create: `papers/code-systems/notebooks/01-codet5.ipynb`
- Create: `papers/code-systems/notebooks/02-codex.ipynb`
- Create: `papers/code-systems/notebooks/03-code-evaluation.ipynb`

[This task will be delegated to subagent. Each markdown file: 1500-2500 words. Each notebook: 12 cells, 600+ code lines. Focus on code understanding, generation, evaluation methodologies, production deployment.]

- [ ] **Step 1: Subagent creates CodeT5 paper** (markdown + notebook)
  - Markdown: ~2300 words covering unified encoder-decoder, code understanding tasks, multi-task learning
  - Notebook: 12 cells with code summarization, code search, code generation examples

- [ ] **Step 2: Subagent creates Codex paper** (markdown + notebook)
  - Markdown: ~2400 words covering LLM code generation, HumanEval benchmark, production deployment
  - Notebook: 12 cells with code generation examples, HumanEval evaluation, error analysis

- [ ] **Step 3: Subagent creates Code Evaluation paper** (markdown + notebook)
  - Markdown: ~2200 words covering HumanEval, MBPP, evaluation metrics, benchmark design
  - Notebook: 12 cells with code execution, pass-at-k calculation, evaluation framework

- [ ] **Step 4: Verify all files created**

```bash
ls -la papers/code-systems/concepts/*.md papers/code-systems/notebooks/*.ipynb
```

Expected: 3 markdown files + 3 notebooks

- [ ] **Step 5: Commit**

```bash
git add papers/code-systems/
git commit -m "feat: add Code & Systems papers (CodeT5, Codex, Code Evaluation)"
```

---

## Task 4: Reasoning & Search Papers (3 papers)

**Files:**
- Create: `papers/reasoning-search/concepts/01-pal.md`
- Create: `papers/reasoning-search/concepts/02-least-to-most.md`
- Create: `papers/reasoning-search/concepts/03-neural-search.md`
- Create: `papers/reasoning-search/notebooks/01-pal.ipynb`
- Create: `papers/reasoning-search/notebooks/02-least-to-most.ipynb`
- Create: `papers/reasoning-search/notebooks/03-neural-search.ipynb`

[This task will be delegated to subagent. Each markdown file: 1500-2500 words. Each notebook: 12 cells, 600+ code lines. Focus on advanced reasoning, decomposition strategies, retrieval mechanisms.]

- [ ] **Step 1: Subagent creates PAL paper** (markdown + notebook)
  - Markdown: ~2300 words covering program generation, symbolic reasoning, combining with LLMs
  - Notebook: 12 cells with program generation, code execution, verification examples

- [ ] **Step 2: Subagent creates Least-to-Most Prompting paper** (markdown + notebook)
  - Markdown: ~2200 words covering problem decomposition, multi-step solving, compositional generalization
  - Notebook: 12 cells with decomposition examples, multi-step chains, performance analysis

- [ ] **Step 3: Subagent creates Neural Search paper** (markdown + notebook)
  - Markdown: ~2400 words covering dense embeddings, retrieval ranking, scaling to millions
  - Notebook: 12 cells with embedding generation, similarity search, FAISS integration

- [ ] **Step 4: Verify all files created**

```bash
ls -la papers/reasoning-search/concepts/*.md papers/reasoning-search/notebooks/*.ipynb
```

Expected: 3 markdown files + 3 notebooks

- [ ] **Step 5: Commit**

```bash
git add papers/reasoning-search/
git commit -m "feat: add Reasoning & Search papers (PAL, Least-to-Most, Neural Search)"
```

---

## Task 5: Enhance Efficiency-Scaling Domain (3 new papers)

**Files:**
- Create: `papers/efficiency-scaling/concepts/02-flash-attention.md`
- Create: `papers/efficiency-scaling/concepts/03-llm-int8.md`
- Create: `papers/efficiency-scaling/concepts/04-knowledge-distillation.md`
- Create: `papers/efficiency-scaling/notebooks/02-flash-attention.ipynb`
- Create: `papers/efficiency-scaling/notebooks/03-llm-int8.ipynb`
- Create: `papers/efficiency-scaling/notebooks/04-knowledge-distillation.ipynb`

[This task will be delegated to subagent. Add to existing efficiency-scaling domain. Each markdown file: 1500-2500 words. Each notebook: 12 cells, 600+ code lines. Focus on inference optimization, quantization, model compression.]

- [ ] **Step 1: Subagent creates Flash-Attention paper** (markdown + notebook)
  - Markdown: ~2400 words covering IO-aware computation, memory efficiency, speedups
  - Notebook: 12 cells with attention computation, memory analysis, performance benchmarks

- [ ] **Step 2: Subagent creates LLM.int8 paper** (markdown + notebook)
  - Markdown: ~2300 words covering 8-bit quantization, outlier detection, inference on consumer GPUs
  - Notebook: 12 cells with quantization process, VRAM comparison, inference speed analysis

- [ ] **Step 3: Subagent creates Knowledge Distillation paper** (markdown + notebook)
  - Markdown: ~2200 words covering teacher-student training, knowledge transfer, efficiency gains
  - Notebook: 12 cells with distillation loss, model compression, accuracy vs size trade-offs

- [ ] **Step 4: Verify all files created**

```bash
ls -la papers/efficiency-scaling/concepts/{02,03,04}*.md papers/efficiency-scaling/notebooks/{02,03,04}*.ipynb
```

Expected: 3 new markdown files + 3 new notebooks (efficiency-scaling now has 4 papers total)

- [ ] **Step 5: Commit**

```bash
git add papers/efficiency-scaling/concepts/02-* papers/efficiency-scaling/concepts/03-* papers/efficiency-scaling/concepts/04-*
git add papers/efficiency-scaling/notebooks/02-* papers/efficiency-scaling/notebooks/03-* papers/efficiency-scaling/notebooks/04-*
git commit -m "feat: enhance efficiency-scaling domain with 3 new papers (Flash-Attention, LLM.int8, Knowledge Distillation)"
```

---

## Task 6: Update papers/README.md

**Files:**
- Modify: `papers/README.md`

- [ ] **Step 1: Read current README**

```bash
head -50 papers/README.md
```

- [ ] **Step 2: Update header count (15→27 papers)**

Find and replace:
```
> **15 seminal papers that shaped modern AI.**
```

With:
```
> **27 seminal papers that shaped modern AI.**
```

- [ ] **Step 3: Add 3 new domains to navigation section**

After "- [Reasoning & Agents](#reasoning--agents)" add:
```markdown
- [Safety & Alignment](#safety--alignment) — Alignment techniques, preference optimization, safety mechanisms
- [Code & Systems](#code--systems) — Code generation, code understanding, evaluation
- [Reasoning & Search](#reasoning--search) — Advanced reasoning, program-aided methods, dense retrieval
- [Efficiency & Scaling](#efficiency--scaling) — (Updated section with now 4 papers)
```

- [ ] **Step 4: Add new domain sections before "Interview Prep Roadmap"**

Insert sections for safety-alignment, code-systems, reasoning-search with paper tables matching existing format.

- [ ] **Step 5: Update Quick Reference table**

Add 12 rows for the new papers.

- [ ] **Step 6: Commit**

```bash
git add papers/README.md
git commit -m "docs: update papers/README.md with 12 new papers and 3 new domains"
```

---

## Task 7: Update roadmaps/papers-roadmap.md

**Files:**
- Modify: `roadmaps/papers-roadmap.md`

- [ ] **Step 1: Update header count (15→27 papers)**

Find and replace:
```
> A guided path through 15 foundational & recent papers
```

With:
```
> A guided path through 27 foundational & recent papers
```

- [ ] **Step 2: Add new papers to interview prep paths where relevant**

- ML Fundamentals path: Add DPO or Constitutional AI (alignment)
- LLM path: Add any alignment papers
- Agents path: Add any code/reasoning papers
- Create new paths if desired (e.g., "Code Generation Engineer" path)

- [ ] **Step 3: Update Complete Papers Timeline section**

Add new papers in chronological order (2020-2023).

- [ ] **Step 4: Update time estimates**

Recalculate total hours for each interview prep path.

- [ ] **Step 5: Commit**

```bash
git add roadmaps/papers-roadmap.md
git commit -m "docs: update papers-roadmap.md with 12 new papers in interview paths"
```

---

## Task 8: Update main README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update "What's Inside" table**

Find Papers row:
```
| **Papers** | 15 foundational & recent papers...
```

Replace with:
```
| **Papers** | 27 foundational & recent papers across 10 domains (vision, NLP, retrieval, agents, efficiency, foundation models, multimodal, safety-alignment, code-systems, reasoning-search) |
```

- [ ] **Step 2: Update "Repository Structure" section**

Find:
```
├── papers/            # 15 foundational & recent papers
```

Replace with:
```
├── papers/            # 27 foundational & recent papers across 10 domains
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update main README with expanded papers coverage (27 papers, 10 domains)"
```

---

## Task 9: Validation & Final Commit

**Files:**
- Verify all 18 new files created
- Verify all 3 documentation files updated

- [ ] **Step 1: Count all paper files**

```bash
echo "Markdown files:" && find papers -name "*.md" -path "*/concepts/*" | wc -l
echo "Notebook files:" && find papers -name "*.ipynb" -path "*/notebooks/*" | wc -l
```

Expected: 15 markdown files (9 new + 6 in efficiency-scaling), 15 notebooks

- [ ] **Step 2: Verify directory structure**

```bash
find papers -maxdepth 1 -type d | sort
```

Expected: 10 domains (vision, nlp, retrieval, agents, efficiency-scaling, foundation-models, multimodal, safety-alignment, code-systems, reasoning-search)

- [ ] **Step 3: Final git status check**

```bash
git status
```

Expected: Clean working tree (all changes committed)

- [ ] **Step 4: Push to remote**

```bash
git push origin master
```

---

## Summary

**Total Tasks:** 9
- Task 1: Create directories
- Tasks 2-5: Create 12 papers (4 subagent tasks)
- Tasks 6-8: Update documentation
- Task 9: Validation & commit

**Deliverables:** 18 new files + 3 updated files
- 9 markdown concept files (safety-alignment: 3, code-systems: 3, reasoning-search: 3)
- 9 Jupyter notebooks (same split)
- 6 additional files for efficiency-scaling domain (3 markdown + 3 notebooks)
- Updated papers/README.md, roadmaps/papers-roadmap.md, main README.md

**Success:** Papers section expanded from 15→27 papers across 7→10 domains
