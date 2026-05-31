# Papers Section Design Spec

**Date:** 2026-05-31  
**Project:** interviewprep-ml  
**Feature:** Add "Papers" section for foundational & recent AI papers with interview-prep implementations

---

## Overview

Add a new `papers/` section to the repository containing 10-15 curated papers (foundational classics + recent breakthroughs) organized by domain, with the same depth and structure as other learning domains:
- **Markdown files:** 8-section detailed explanations with interview Q&A (1500-2500 words each)
- **Jupyter notebooks:** 12-cell implementation-focused notebooks with basic → advanced → real-world examples
- **Target audience:** ML/AI engineers preparing for interviews who need to understand papers deeply AND be able to code up the key ideas

---

## Scope

### Papers Included (12 total)

**Vision (2 papers):**
1. ResNet: Deep Residual Learning for Image Recognition (He et al., 2015)
2. Vision Transformer (ViT): An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)

**NLP/LLM Core (5 papers):**
3. Attention Is All You Need (Vaswani et al., 2017)
4. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)
5. Language Models are Few-Shot Learners (GPT-3) (Brown et al., 2020)
6. Scaling Laws for Neural Language Models (Hoffmann et al., 2020)
7. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

**Retrieval & Multimodal (2 papers):**
8. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
9. Learning Transferable Visual Models from Natural Language Supervision (CLIP) (Radford et al., 2021)

**Reasoning & Agents (3 papers):**
10. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)
11. ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)
12. Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)

### Out of Scope

- Alignment/safety papers (Constitutional AI) — can add in Phase 2
- Efficiency papers (Flash Attention, Mixture of Experts) — can add in Phase 2
- Vision-language papers beyond CLIP — future expansion

---

## Architecture & Organization

### Directory Structure

```
papers/
├── README.md                          # Papers section overview, navigation guide, learning paths
├── vision/
│   ├── concepts/
│   │   ├── 01-resnet.md
│   │   └── 02-vision-transformer.md
│   └── notebooks/
│       ├── 01-resnet.ipynb
│       └── 02-vision-transformer.ipynb
├── nlp/
│   ├── concepts/
│   │   ├── 01-attention-is-all-you-need.md
│   │   ├── 02-bert.md
│   │   ├── 03-gpt3.md
│   │   ├── 04-scaling-laws.md
│   │   └── 05-lora.md
│   └── notebooks/
│       ├── 01-attention-is-all-you-need.ipynb
│       ├── 02-bert.ipynb
│       ├── 03-gpt3.ipynb
│       ├── 04-scaling-laws.ipynb
│       └── 05-lora.ipynb
├── retrieval/
│   ├── concepts/
│   │   ├── 01-rag.md
│   │   └── 02-clip.md
│   └── notebooks/
│       ├── 01-rag.ipynb
│       └── 02-clip.ipynb
└── agents/
    ├── concepts/
    │   ├── 01-chain-of-thought.md
    │   ├── 02-react.md
    │   └── 03-tree-of-thoughts.md
    └── notebooks/
        ├── 01-chain-of-thought.ipynb
        ├── 02-react.ipynb
        └── 03-tree-of-thoughts.ipynb
```

**Total deliverables:** 12 markdown files + 12 notebooks + 1 README = 25 new files

### File Naming Convention

- **Markdown:** `NN-paper-slug.md` (e.g., `01-resnet.md`, `02-vision-transformer.md`)
  - NN: 01-02 for vision, 01-05 for nlp, 01-02 for retrieval, 01-03 for agents
- **Notebooks:** `NN-paper-slug.ipynb` (matching markdown)
- **Consistency:** Filename slug matches paper short name used in navigation

### Papers README Structure

The `papers/README.md` file includes:
1. **What are these papers?** — Quick orientation (why papers matter for interviews)
2. **How to use this section** — Reading guide for different goals
3. **Navigation by domain** — Links to each domain's papers with brief descriptions
4. **Chronological view** — Timeline of papers (2015-2023) showing evolution
5. **Interview prep roadmap** — Suggested reading order for different interview focuses
6. **Quick reference table** — Paper name, year, venue, key contribution, interview relevance

---

## Content Format Per Paper

### Markdown File Format (8 Sections)

Each concept markdown file follows the standard 8-section format used across the repo:

#### Section 1: Paper Overview (150-200 words)
- Full paper title, authors, publication venue (NeurIPS/ICML/arXiv), year
- Link to original paper (DOI/arXiv)
- Why this paper matters for interviews and modern ML practice
- Context: What problem did the field face before this paper?

#### Section 2: Core Contribution (200-250 words)
- What is the main innovation?
- Why is it significant? (impact on the field)
- How did it change the landscape? (papers that built on it, adoption)
- Connection to larger ML concepts

#### Section 3: Key Ideas & Algorithm (300-400 words)
- Step-by-step explanation of the core algorithm/mechanism
- Mathematical notation where relevant (but plain language first)
- Mermaid diagram showing the architecture or flow
- Intuitive explanation of why this works

#### Section 4: Architecture / Trade-offs (300-400 words)
- Comparison tables: This paper vs. prior approaches
- Design choices: Why did authors make specific decisions?
- Trade-offs: Speed vs. accuracy, simplicity vs. power, training vs. inference
- When to use this approach vs. alternatives

#### Section 5: Interview Q&A (5-8 questions, 2-3 sentences each)
**Focus on judgment calls, not memorization:**
- "Why would you use this technique instead of X in production?"
- "What are the limitations of this approach?"
- "How would you implement this from scratch?"
- "What would go wrong if you ignored this key insight?"
- "How has this paper influenced modern systems?"

#### Section 6: Best Practices (5-8 bullets)
- How practitioners actually use ideas from this paper
- Parameter ranges, hyperparameters that matter
- Common optimizations or variants used in production
- Monitoring and debugging tips

#### Section 7: Common Pitfalls (3-5 mistakes)
- Misunderstandings people have about this paper
- What breaks when you try to apply it naively
- How to detect and fix these issues
- Why the paper's constraints matter

#### Section 8: Code Examples (2-3 blocks, 40-60 lines each)
- **Example 1 (Basic):** Simplest implementation of core idea (20-40 lines, synthetic data)
- **Example 2 (Production):** Real library usage with error handling (60-100 lines)
- **Example 3 (Real-world scenario):** How the paper's ideas apply in practice
- All code uses real imports (transformers, torch, numpy), not pseudo-code

### Notebook File Format (12 Cells)

Each implementation notebook follows the standard 12-cell structure:

| Cell # | Type | Content | Lines |
|--------|------|---------|-------|
| 1 | Markdown | Title, learning objectives (4 key things to learn) | — |
| 2 | Code | Imports, device setup, random seeds | 10-15 |
| 3 | Markdown | Level 1: Basic Implementation | — |
| 4 | Code | Level 1: Core idea in isolation (synthetic data) | 20-40 |
| 5 | Markdown | Level 2: Advanced Implementation | — |
| 6 | Code | Level 2: Production patterns, error handling, optimization | 60-100 |
| 7 | Markdown | Real-World Example 1 | — |
| 8 | Code | Real-World Example 1: Specific use case with realistic setup | 40-60 |
| 9 | Markdown | Real-World Example 2 | — |
| 10 | Code | Real-World Example 2: Different scenario showing trade-offs | 40-60 |
| 11 | Markdown | Key Takeaways + comparison table + related papers | — |
| 12 | Code | Visualization: Compare variants, show timing, memory usage | 30-50 |

**Notebook Quality Standards:**
- All imports are real and importable (transformers, torch, numpy, sklearn)
- Device management explicit throughout (cuda/cpu handling)
- Error handling for OOM, shape mismatches, gradient issues
- Code is runnable end-to-end (uses synthetic/mock data if needed)
- Execution time < 2 hours (typically 10-30 minutes)
- Comments explain WHY, not WHAT
- Type hints and docstrings on functions

---

## Integration with Main Repository

### Navigation & Cross-linking

1. **Main README.md update:**
   - Add `Papers` row to "What's Inside" table: "12 foundational & recent papers with interview Q&A and implementations"
   - Add `papers/` link to "Start Here" section: "Understand seminal papers that shaped modern AI"

2. **Roadmaps:**
   - Create `roadmaps/papers-roadmap.md` with suggested reading orders:
     - "Chronological evolution: 2015-2023"
     - "For interview prep: Start with Attention → BERT → understanding modern systems"
     - "By domain: Vision papers, NLP papers, Agent papers"

3. **Cross-linking within concepts:**
   - From `llm/concepts/01-transformers.md` → link to `papers/nlp/01-attention-is-all-you-need.md`
   - From `agentic-ai/concepts/XX-chain-of-thought.md` → link to `papers/agents/01-chain-of-thought.md`
   - From other domain concept files → link to relevant papers

4. **Papers README "See Also" sections:**
   - Each paper file includes related concept links
   - Example: Attention paper links to `llm/concepts/XX-transformers`, `llm/concepts/XX-self-attention`, etc.

### Metadata & Frontmatter

Each concept markdown file includes YAML frontmatter:
```yaml
---
title: "Attention Is All You Need"
authors: "Vaswani et al."
year: 2017
venue: "NeurIPS"
doi: "https://doi.org/10.5555/3295222.3295349"
arxiv: "https://arxiv.org/abs/1706.03762"
domain: "nlp"
difficulty: "intermediate"
interview_frequency: "very_high"
related_concepts:
  - llm/concepts/01-transformers
  - llm/concepts/02-self-attention
---
```

---

## Validation & Quality Gates

### Automated Testing

**File structure validation:**
- All 12 markdown files exist in correct folders
- All 12 notebooks exist and are valid .ipynb format
- README.md exists at `papers/README.md`
- Naming convention: files match `NN-slug.md` and `NN-slug.ipynb` pattern

**Content validation:**
- Markdown: 1500-2500 words per file (8 sections present)
- Notebooks: 12 cells, proper cell types
- Imports: All imports are real and available (no pseudo-code)
- Code quality: No undefined variables, syntax valid, runnable end-to-end
- Cross-linking: All links to concepts/papers exist

**Quality gates:**
- ✅ All tests pass
- ✅ All notebooks execute without errors
- ✅ Markdown follows 8-section format
- ✅ Interview Q&A has no memorization questions
- ✅ Code examples use real libraries

### Manual Review Checklist

Before marking complete:
- [ ] Papers README is comprehensive and navigation-friendly
- [ ] Main README updated with Papers section
- [ ] Roadmaps created and linked
- [ ] Cross-links from related concepts verified
- [ ] All papers have consistent formatting and quality
- [ ] No placeholder text or TODOs remain
- [ ] Notebooks all execute in < 2 hours

---

## Execution Plan

### Phase 1: Foundation & Setup
- Create `papers/` directory structure (4 domain folders with concepts/ + notebooks/)
- Write `papers/README.md` with navigation and roadmaps
- Update main README.md to include Papers section
- Create `roadmaps/papers-roadmap.md`
- **Duration:** ~2-3 hours

### Phase 2: Markdown Content (Concepts)
- Write 12 markdown concept files following 8-section format
- Each file: 1500-2500 words with diagrams, code examples, Q&A
- **Duration:** ~12-15 hours (1-1.5 hours per paper)

### Phase 3: Notebook Implementation
- Generate 12 Jupyter notebooks (12 cells each)
- Level 1 basic + Level 2 advanced + 2 real-world examples per paper
- Test all notebooks are runnable
- **Duration:** ~15-18 hours (1.5-2 hours per notebook)

### Phase 4: Cross-linking & Integration
- Link from related concept files to papers
- Update concept files in other domains that reference papers
- Verify all links work
- **Duration:** ~2-3 hours

### Phase 5: Validation & Polish
- Run test suite (structure, imports, notebook execution)
- Fix any issues
- Final review of quality and consistency
- Commit and push
- **Duration:** ~3-4 hours

**Total estimated effort:** 34-43 hours
**Realistic timeline:** 1-2 weeks of focused work (or spread across multiple sessions)

---

## Success Criteria

✅ **Definition of Done:**
- All 12 papers have comprehensive markdown files (8-section format, 1500-2500 words)
- All 12 papers have runnable notebooks (12 cells, 600-1000 lines of code)
- Papers README created with clear navigation and reading paths
- Main repository README updated to feature Papers section
- Roadmaps created with suggested reading orders
- Cross-links verified (from related concepts to papers)
- All tests pass (structure, imports, code quality)
- No TODOs or placeholder text remain

---

## Alternatives Considered

### Alternative A: Lighter Format (just markdown, no notebooks)
- **Pros:** Faster to execute, easier to maintain
- **Cons:** Doesn't align with interview-prep focus (code up the ideas), less valuable for practitioners
- **Decision:** Rejected — notebooks are essential for learning and implementing

### Alternative B: Just notebooks (embed paper context in cells)
- **Pros:** Less file duplication, tighter integration
- **Cons:** Notebooks become verbose, hard to reference the summary separately, inconsistent with repo pattern
- **Decision:** Rejected — maintain consistent structure with other domains

### Alternative C: Include all 20+ papers (comprehensive coverage)
- **Pros:** More comprehensive
- **Cons:** Much larger effort, harder to maintain quality, scope creep
- **Decision:** Rejected — start with 12 focused papers, expand later based on demand

---

## Dependencies & Assumptions

**Dependencies:**
- Notebooks require torch, transformers, numpy, matplotlib (already in repo)
- Some papers may need sklearn, scipy, or other existing dependencies
- No new external dependencies needed

**Assumptions:**
- Interview-prep audience wants depth over breadth (prefer 12 well-done papers over 30 rushed ones)
- Users can understand papers alongside concepts in other domains
- Cross-linking to concepts in other domains will enhance learning
- 10-15 papers is sustainable long-term; can expand in future versions

---

## Open Questions & Future Considerations

1. **Phase 2 expansion:** Should we add alignment/safety papers (Constitutional AI, RLHF) or efficiency papers (Flash Attention)?
   - **Decision point:** After Phase 1 complete, assess demand and team capacity

2. **Versioning:** Should we track which version of papers we're referencing (published vs. arXiv versions)?
   - **Decision point:** Include DOI/arXiv link in frontmatter; doesn't require versioning

3. **Reproductions:** Should some papers include "reproduction notebook" sections that try to replicate figures/results?
   - **Decision point:** Out of scope for interview prep; implementation examples are sufficient

4. **Paper updates:** As papers are updated/superseded, how do we handle versioning?
   - **Decision point:** Flag "superseded by X" in overview if relevant, but keep original paper for historical context

---

## File Manifest

**New files to create:**
```
papers/
├── README.md (1 file)
├── vision/
│   ├── concepts/ (2 files)
│   └── notebooks/ (2 files)
├── nlp/
│   ├── concepts/ (5 files)
│   └── notebooks/ (5 files)
├── retrieval/
│   ├── concepts/ (2 files)
│   └── notebooks/ (2 files)
└── agents/
    ├── concepts/ (3 files)
    └── notebooks/ (3 files)

roadmaps/papers-roadmap.md (1 new file)
```

**Modified files:**
- README.md (main) — add Papers section to table of contents
- CONTRIBUTING.md (optional) — add paper contribution guidelines

**Total new files:** 25 (12 markdown + 12 notebooks + 1 README + papers roadmap)

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Notebooks not runnable due to dependency issues | High | Test on clean environment; include device/error handling |
| Paper explanations overly technical or simplified | High | Follow 8-section template; include basic + advanced explanations |
| Code examples don't match paper descriptions | Medium | Review code against paper twice; test end-to-end |
| Cross-links break or go stale | Low | Validate all links; include redirect testing in test suite |
| Scope creeps to 30+ papers | Medium | Lock list to 12 papers; add future papers in separate PR |

---

## Timeline & Milestones

**Milestone 1 (Day 1-2):** Foundation setup
- ✓ Directory structure
- ✓ Papers README
- ✓ Main README update
- ✓ Roadmaps created

**Milestone 2 (Day 3-8):** Markdown content
- ✓ All 12 concept files written
- ✓ All 8 sections complete
- ✓ Diagrams and code examples included

**Milestone 3 (Day 9-14):** Notebooks
- ✓ All 12 notebooks created
- ✓ All executable and tested
- ✓ Code follows quality standards

**Milestone 4 (Day 15-16):** Integration
- ✓ Cross-linking complete
- ✓ Test suite passes
- ✓ Final review done

**Milestone 5 (Day 17):** Launch
- ✓ All files committed
- ✓ PR created and merged
- ✓ Section live and navigable

---

## Version History

**v1.0 (2026-05-31):**
- Initial design spec
- 12 papers, 4 domains (vision, NLP, retrieval, agents)
- 8-section markdown + 12-cell notebooks
- Interview-prep focused

