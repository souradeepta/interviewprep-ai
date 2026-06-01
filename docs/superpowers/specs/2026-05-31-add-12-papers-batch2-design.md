# Add 12 More Papers: Batch 2 - Safety, Code, Reasoning, Efficiency

**Date:** 2026-05-31  
**Project:** interviewprep-ml  
**Feature:** Expand Papers section from 15 to 27 papers across 10 domains (adding 3 new domains + enhancing 1 existing)

---

## Overview

Add 12 new research papers organized across **3 new domains + 1 enhanced domain**:

1. **safety-alignment/** (NEW) — 3 papers on alignment, safety, preference optimization
2. **code-systems/** (NEW) — 3 papers on code generation, evaluation, systems
3. **reasoning-search/** (NEW) — 3 papers on advanced reasoning and search
4. **efficiency-scaling/** (ENHANCED) — Add 3 papers to existing domain (now 4 total)

Each paper follows the established 8-section markdown + 12-cell notebook pattern.

---

## Scope

### 3 New Domains with Papers

**Domain 1: safety-alignment/** (3 papers)
1. Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)
   - Core idea: AI-generated feedback guides model alignment without human feedback
   - Interview value: High (alignment techniques, scalable oversight)

2. Direct Preference Optimization (DPO) (Rafailov et al., 2023)
   - Core idea: Fine-tune models using preference pairs without RL
   - Interview value: Very High (modern alignment approach, alternative to RLHF)

3. RLHF / InstructGPT: Training Language Models to Follow Instructions (Ouyang et al., 2022)
   - Core idea: Combine SFT + RLHF to align models with human preferences
   - Interview value: Very High (foundational alignment technique)

**Domain 2: code-systems/** (3 papers)
1. CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code (Wang et al., 2021)
   - Core idea: Unified architecture for code understanding and generation tasks
   - Interview value: High (production code models, multi-task learning)

2. Evaluating Large Language Models Trained on Code (Codex) (Chen et al., 2021)
   - Core idea: LLMs can generate working code; introduces HumanEval benchmark
   - Interview value: Very High (code generation capabilities, evaluation methodology)

3. Benchmarking Language Models on Code Completion (HumanEval, MBPP)
   - Core idea: Standardized evaluation for code generation quality
   - Interview value: High (how to evaluate code models, benchmark design)

**Domain 3: reasoning-search/** (3 papers)
1. Program-Aided Language Models (PAL) (Gao et al., 2023)
   - Core idea: Use LLM to generate programs, execute them for reasoning
   - Interview value: High (combining symbolic + neural reasoning)

2. Least-to-Most Prompting: Compositional Generalization with Chained Prompts (Zhou et al., 2023)
   - Core idea: Decompose hard problems, solve easy ones first, chain solutions
   - Interview value: High (problem decomposition, prompt engineering)

3. Neural Search / Dense Passage Retrieval (Karpukhin et al., 2020)
   - Core idea: Learn dense embeddings for retrieval instead of sparse keywords
   - Interview value: High (modern retrieval, scaling knowledge search)

### 1 Enhanced Domain

**efficiency-scaling/** (Add 3 papers, now 4 total)
- Existing: Mixture of Experts (2016-2023)
- **New papers:**
  1. Flash-Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
     - Core idea: Reorganize attention computation for hardware efficiency
     - Interview value: High (inference optimization, systems thinking)

  2. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022)
     - Core idea: Run large models on consumer GPUs via 8-bit quantization
     - Interview value: High (practical deployment, quantization techniques)

  3. Knowledge Distillation / Model Compression (Hinton et al., 2015 or modern variants)
     - Core idea: Train small models to match large model outputs
     - Interview value: Medium-High (efficiency techniques, training tricks)

---

## Directory Structure

```
papers/
├── safety-alignment/         (NEW)
│   ├── concepts/
│   │   ├── 01-constitutional-ai.md
│   │   ├── 02-dpo.md
│   │   └── 03-rlhf-instructgpt.md
│   └── notebooks/
│       ├── 01-constitutional-ai.ipynb
│       ├── 02-dpo.ipynb
│       └── 03-rlhf-instructgpt.ipynb
├── code-systems/            (NEW)
│   ├── concepts/
│   │   ├── 01-codet5.md
│   │   ├── 02-codex.md
│   │   └── 03-code-evaluation.md
│   └── notebooks/
│       ├── 01-codet5.ipynb
│       ├── 02-codex.ipynb
│       └── 03-code-evaluation.ipynb
├── reasoning-search/        (NEW)
│   ├── concepts/
│   │   ├── 01-pal.md
│   │   ├── 02-least-to-most.md
│   │   └── 03-neural-search.md
│   └── notebooks/
│       ├── 01-pal.ipynb
│       ├── 02-least-to-most.ipynb
│       └── 03-neural-search.ipynb
├── efficiency-scaling/      (ENHANCED: 1→4 papers)
│   ├── concepts/
│   │   ├── 01-mixture-of-experts.md     (existing)
│   │   ├── 02-flash-attention.md        (NEW)
│   │   ├── 03-llm-int8.md               (NEW)
│   │   └── 04-knowledge-distillation.md (NEW)
│   └── notebooks/
│       ├── 01-mixture-of-experts.ipynb     (existing)
│       ├── 02-flash-attention.ipynb        (NEW)
│       ├── 03-llm-int8.ipynb               (NEW)
│       └── 04-knowledge-distillation.ipynb (NEW)
└── [existing 6 domains unchanged]
```

---

## Content Format (Consistent with Existing Papers)

### Markdown Concept File (1500-2500 words, 8 sections)
1. **Paper Overview** — Title, authors, venue, year, DOI, significance
2. **Core Contribution** — Main innovation and impact
3. **Key Ideas & Algorithm** — Step-by-step with Mermaid diagram
4. **Architecture / Trade-offs** — Comparison tables, design choices
5. **Interview Q&A** — 5-8 judgment-focused questions
6. **Best Practices** — 5-8 actionable production tips
7. **Common Pitfalls** — 3-5 real mistakes with solutions
8. **Code Examples** — 2-3 real implementations with imports

### Jupyter Notebook (12 cells, 600+ code lines)
- Cell 1: Title + Learning Objectives
- Cell 2: Imports + Device Setup
- Cells 3-4: Level 1 Basic Implementation
- Cells 5-6: Level 2 Advanced Implementation
- Cells 7-8: Real-World Example 1
- Cells 9-10: Real-World Example 2
- Cell 11: Key Takeaways
- Cell 12: Visualization

---

## Integration Points

### Documentation Updates

**1. papers/README.md**
- Update header: "15 papers" → "27 papers"
- Add 3 new domain sections + update navigation links
- Update "Quick Reference" table (now 27 papers)
- Update Chronological Timeline

**2. roadmaps/papers-roadmap.md**
- Update header: "15 papers" → "27 papers"
- Add new papers to interview prep paths where relevant
- Update Complete Papers Timeline

**3. main README.md**
- Update "What's Inside" table (15 → 27 papers, 7 → 10 domains)
- Update "Repository Structure" section

---

## File Manifest

### New Files (18 total)

**Markdown Concept Files (9):**
- `papers/safety-alignment/concepts/01-constitutional-ai.md`
- `papers/safety-alignment/concepts/02-dpo.md`
- `papers/safety-alignment/concepts/03-rlhf-instructgpt.md`
- `papers/code-systems/concepts/01-codet5.md`
- `papers/code-systems/concepts/02-codex.md`
- `papers/code-systems/concepts/03-code-evaluation.md`
- `papers/reasoning-search/concepts/01-pal.md`
- `papers/reasoning-search/concepts/02-least-to-most.md`
- `papers/reasoning-search/concepts/03-neural-search.md`

**Jupyter Notebooks (9):**
- `papers/safety-alignment/notebooks/01-constitutional-ai.ipynb`
- `papers/safety-alignment/notebooks/02-dpo.ipynb`
- `papers/safety-alignment/notebooks/03-rlhf-instructgpt.ipynb`
- `papers/code-systems/notebooks/01-codet5.ipynb`
- `papers/code-systems/notebooks/02-codex.ipynb`
- `papers/code-systems/notebooks/03-code-evaluation.ipynb`
- `papers/reasoning-search/notebooks/01-pal.ipynb`
- `papers/reasoning-search/notebooks/02-least-to-most.ipynb`
- `papers/reasoning-search/notebooks/03-neural-search.ipynb`

**Plus 3 new markdown files for efficiency-scaling domain:**
- `papers/efficiency-scaling/concepts/02-flash-attention.md`
- `papers/efficiency-scaling/concepts/03-llm-int8.md`
- `papers/efficiency-scaling/concepts/04-knowledge-distillation.md`
- `papers/efficiency-scaling/notebooks/02-flash-attention.ipynb`
- `papers/efficiency-scaling/notebooks/03-llm-int8.ipynb`
- `papers/efficiency-scaling/notebooks/04-knowledge-distillation.ipynb`

### Modified Files (3)
- `papers/README.md`
- `roadmaps/papers-roadmap.md`
- `README.md` (main)

---

## Execution Strategy

**Phase 1 (Main Session):**
- Create 3 new domain directories
- Update documentation

**Phase 2 (Parallel Subagents):**
- Subagent 1: 3 safety-alignment papers
- Subagent 2: 3 code-systems papers
- Subagent 3: 3 reasoning-search papers
- Subagent 4: 3 efficiency-scaling papers (additions to existing domain)

**Phase 3 (Main Session):**
- Validation & final commit
- Push to remote

---

## Success Criteria

✅ **Definition of Done:**
- All 12 new papers created (9 markdown + 12 notebooks including 3 efficiency papers)
- 3 new domains created with proper structure
- efficiency-scaling domain enhanced (1→4 papers)
- papers/README.md updated with all new content
- roadmaps/papers-roadmap.md updated
- main README.md updated with new counts
- All files committed and pushed
- No TODOs or placeholder text

---

## Timeline & Effort

**Estimated Effort:**
- Phase 1 (Directories + initial setup): 15 minutes
- Phase 2 (12 papers, parallel subagents): 2-3 hours
- Phase 3 (Documentation + validation): 30 minutes
- **Total: ~3-4 hours**

**Execution: Parallel subagents recommended** (same as previous batch)

---

## Version History

**v1.0 (2026-05-31):**
- Initial design spec
- 12 new papers across 3 new domains + 1 enhanced domain
- Expansion from 15→27 papers, 7→10 domains
