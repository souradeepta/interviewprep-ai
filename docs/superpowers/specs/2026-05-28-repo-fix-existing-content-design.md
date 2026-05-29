# Repo Fix Existing Content — Design Spec

**Date:** 2026-05-28  
**Status:** Approved  
**Approach:** Two-phase parallel batch (Approach 3)

---

## 1. Goals & Scope

**Goal:** Fix all existing content gaps so every section has complete, production-quality, interview-ready material before adding any new sections.

**In scope — 5 sub-projects:**

| # | Sub-project | Current state | Target state |
|---|-------------|--------------|-------------|
| A | modern-ai notebooks 46-55 | 10 notebooks, 320-480 lines each | 600+ lines each, 16-cell format |
| B | System-design patterns 14-31 | 18 patterns, ~1000 words each | 1800-2400 words, failure scenarios, cost models, 8-10 Q&A |
| C | Coding section | 1 file (`arrays-strings.md`) | 22+ files across `data-structures/` + new `algorithms/` dir |
| D | ml/ notebooks | 40 concepts, 0 notebooks | New `ml/notebooks/` with 40 notebooks, 12-cell format |
| E | Arch-review diagrams | 3 of 30 systems have diagrams | 81 files (3 Mermaid diagrams × 27 systems) |

**Out of scope (deferred to next phase):** New sections — RL, statistics, CV/NLP, post-mortems, cheatsheets, interview deep-dive.

---

## 2. Execution Architecture

### Phase 1 — Three agents in parallel (medium effort)

| Agent | Sub-project | Files affected |
|-------|-------------|---------------|
| Agent 1 | modern-ai notebooks 46-55 | 10 notebooks |
| Agent 2 | System-design patterns 14-31 | 18 markdown files |
| Agent 3 | Coding section expansion | ~22 new files |

Each agent works independently with no shared state. All read CLAUDE.md for spec. Each commits on completion.

**Validation gate** runs after all three Phase 1 agents complete. Phase 2 does not start until validation passes.

### Phase 2 — Two agents in parallel (large effort)

| Agent | Sub-project | Files affected |
|-------|-------------|---------------|
| Agent 4 | ml/ notebooks | 40 new notebooks |
| Agent 5 | Arch-review diagrams | 81 new Mermaid diagram files |

Phase 2 agents are fully independent — no ordering constraint between them.

---

## 3. Content Standards

### Sub-project A: modern-ai notebooks 46-55

**16-cell structure (must be exact):**
1. Title + 4 learning objectives (Markdown)
2. Imports + device setup + random seeds (Code)
3. Level 1 header (Markdown)
4. Level 1 numpy implementation, 50-80 lines (Code)
5. Level 2 header (Markdown)
6. Level 2 torch implementation, 100-140 lines (Code)
7. Real-World Example 1 header (Markdown)
8. RW1 code, 60-90 lines (Code)
9. Real-World Example 2 header (Markdown)
10. RW2 code, 60-90 lines (Code)
11. Real-World Example 3 header (Markdown)
12. RW3 code, 60-90 lines (Code)
13. Comparison header (Markdown)
14. Comparison code + matplotlib visualization (Code)
15. Key Takeaways with variant table + failure modes (Markdown)
16. Exercises (Markdown)

**Code requirements:** real library imports, explicit device management (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`), OOM error handling, batch processing patterns, 600+ total code lines.

### Sub-project B: System-design patterns 14-31

**8-section format per file:**
1. **Detailed Explanation** (200-300 words): what it is, why it matters in production
2. **Core Intuition** (2-3 sentences max): memorable analogy
3. **How It Works** (4-6 numbered steps + Mermaid `flowchart TD`)
4. **Architecture / Trade-offs** (2+ comparison tables with numbers)
5. **Failure Scenarios** (3-5 real failure walkthroughs: symptom → root cause → fix)
6. **Cost Model** (envelope calculation with actual numbers, e.g. "$0.02/request × 10M requests/day = $200K/month")
7. **Interview Q&A** (8-10 judgment-focused questions — "when would you NOT use this?", "how would you debug X?")
8. **Best Practices** (5-8 actionable bullets with numbers/ranges)

**Target:** 1800-2400 words per file.

### Sub-project C: Coding section

**`coding/data-structures/` — 10 files total (1 exists):**
- arrays-strings.md (exists)
- linked-lists.md, stacks-queues.md, trees.md, graphs.md, heaps.md, hash-tables.md, tries.md, union-find.md, segment-trees.md

**`coding/algorithms/` — 12 new files:**
- sorting.md, binary-search.md, sliding-window.md, two-pointers.md, dynamic-programming.md, backtracking.md, greedy.md, recursion.md, graph-traversal.md, divide-conquer.md, bit-manipulation.md, string-patterns.md

**Each file must include:**
- Problem patterns table (pattern name, when to use, complexity)
- Python implementation with type hints and docstrings
- Time + space complexity analysis
- Interview recognition template ("if you see X in the problem, think Y")
- 3-5 worked examples with code

### Sub-project D: ml/ notebooks

**Location:** `ml/notebooks/`  
**Count:** 40 notebooks, one per concept file in `ml/concepts/`  
**Naming:** Sequential numbering matching alphabetical order of concept files (e.g. `01-activation-functions.ipynb`)

**12-cell format:**
1. Title + 4 learning objectives (Markdown)
2. Imports + device setup + seeds (Code)
3. Level 1 header (Markdown)
4. Level 1 basic implementation, 20-40 lines numpy (Code)
5. Level 2 header (Markdown)
6. Level 2 advanced, 60-100 lines torch/sklearn (Code)
7. Real-World Example 1 (Markdown + Code, 40-60 lines)
8. Real-World Example 2 (Markdown + Code, 40-60 lines)
9. Real-World Example 3 (Markdown + Code, 40-60 lines)
10. Comparison visualization with matplotlib (Markdown + Code)
11. Key Takeaways with variant table (Markdown)
12. Exercises (Markdown)

**Code requirements:** real imports (numpy, torch, sklearn), device management, batch processing, error handling.

### Sub-project E: Arch-review diagrams

**Location:** `arch-review/diagrams/`  
**Systems covered:** all except 01, 02, 05 (those three already have diagrams) — i.e. systems 03, 04, 06-30 = 27 systems  
**Files per system:** 3 Mermaid markdown files:
- `{NN}-{system-name}-01-system-architecture.md` — high-level component diagram (`graph TD`)
- `{NN}-{system-name}-02-application-architecture.md` — application/service layer diagram (`graph TD`)
- `{NN}-{system-name}-03-process-flow.md` — request/data flow (`sequenceDiagram`)

**Mermaid rules (strict):**
- Light background, dark text (navy, charcoal) — no bright colors
- No Unicode characters — no emojis, no fancy quotes, no special symbols
- Labeled edges with clear data flow direction
- 8-15 nodes per diagram (not too sparse, not overcrowded)

---

## 4. Validation & Success Criteria

### Phase 1 validation gate

Run before dispatching Phase 2:

```bash
# Sub-project A: modern-ai notebooks 46-55
python3 -c "
import json, glob
files = sorted(glob.glob('modern-ai/notebooks/4[6-9]-*.ipynb') + glob.glob('modern-ai/notebooks/5[0-5]-*.ipynb'))
for f in files:
    nb = json.load(open(f))
    lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type']=='code')
    print('✅' if lines >= 600 else '⚠️ ', f.split('/')[-1], lines, 'lines')
print(f'Total: {len(files)} notebooks')
"

# Sub-project B: system-design patterns 14-31
for f in system-design/patterns/1[4-9]-*.md system-design/patterns/2[0-9]-*.md system-design/patterns/3[01]-*.md; do
    words=$(wc -w < "$f")
    [ "$words" -ge 1800 ] && echo "✅ $words words: $(basename $f)" || echo "⚠️  $words words: $(basename $f)"
done

# Sub-project C: coding section
echo "DS files: $(ls coding/data-structures/ | wc -l) (need 10)"
echo "Algo files: $(ls coding/algorithms/ 2>/dev/null | wc -l) (need 12)"
```

### Phase 2 success criteria

```bash
# Sub-project D: ml/ notebooks
echo "ML notebooks: $(ls ml/notebooks/*.ipynb 2>/dev/null | wc -l) (need 40)"

# Sub-project E: arch-review diagrams
echo "Diagram files: $(ls arch-review/diagrams/*.md 2>/dev/null | grep -v README | wc -l) (need 90)"
```

### Overall done when

- [ ] All Phase 1 validation checks pass (600+ lines, 1800+ words, 22+ coding files)
- [ ] `ml/notebooks/` has exactly 40 notebooks
- [ ] `arch-review/diagrams/` has 90+ diagram files (81 new + 9 existing)
- [ ] Existing test suite passes: `python3 -m pytest tests/ -v`
- [ ] README updated with accurate file counts for all sections

---

## 5. Commit Strategy

Each agent commits its own sub-project on completion using conventional commit messages:

```bash
# Agent 1
git commit -m "feat: expand modern-ai notebooks 46-55 to 600+ lines with 16-cell format"

# Agent 2  
git commit -m "feat: expand system-design patterns 14-31 to 1800-2400 words with failure scenarios and cost models"

# Agent 3
git commit -m "feat: add coding/algorithms/ directory and expand data-structures to 10 topics"

# Agent 4
git commit -m "feat: add ml/notebooks/ with 40 implementation notebooks"

# Agent 5
git commit -m "feat: add arch-review diagrams for 27 systems (81 Mermaid files)"
```

After all agents complete, a final commit updates README with accurate counts:
```bash
git commit -m "docs: update README with accurate file counts after content fixes"
```
