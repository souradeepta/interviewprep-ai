# Claude Agent Guidelines for Interviewprep-ML

This document captures the learnings from the LLM concept notebooks project. Future agents should follow these principles when working on this codebase.

## Project Context

**Goal:** Create comprehensive, production-ready learning materials for 148 core AI/ML concepts across three sections: AI Fundamentals (40), LLM (44), and Agentic AI (64), focusing on interview preparation and practical implementation.

**Deliverables:**
- 148 Jupyter notebooks with 3-level implementation (basic → advanced → real-world examples)
- 148 enhanced markdown files with comprehensive explanations, interview Q&A, best practices, common pitfalls
- 8-section markdown structure: Detailed Explanation, Core Intuition, How It Works, Architecture/Trade-offs, Interview Q&A, Best Practices, Common Pitfalls, Code Examples, Related Concepts
- Production-grade Python implementations using real libraries (transformers, torch, sklearn, etc.)

**Key Success Metric:** Materials enable candidates to both understand AND implement concepts after reading. All content follows agentic-ai 8-section markdown + 12-cell notebook pattern.

---

## Core Design Principles (MUST FOLLOW)

### 1. Code Over Theory
- **Rule:** Implementation code should exceed theory in notebooks. Code teaches faster.
- **Why:** Interview candidates need to recognize patterns and implement quickly. Generic explanations don't transfer to real problems.
- **How to apply:** 
  - Notebooks: 70% code, 30% explanation
  - Markdown: 50% explanation, 30% code examples, 20% Q&A/best practices
  - Never use pseudo-code; all imports must be real and importable

### 2. Real Libraries, Not Templates
- **Rule:** Every implementation uses actual library imports. No generic templates.
- **Why:** Pseudo-code doesn't teach transferable skills. Real code shows what people will actually write.
- **Libraries to use:**
  - `transformers` - Model loading, training, inference
  - `peft` - LoRA, Adapters, Prefix Tuning
  - `sentence-transformers` - Embeddings and semantic search
  - `torch` - Tensor operations, device management
  - `onnx` - Model optimization and export
- **How to apply:** When implementing concepts, use these libraries directly with realistic parameters and patterns.

### 3. Production Patterns Matter
- **Rule:** Show optimizations and production considerations, not just basic functionality.
- **Why:** Interview candidates need to know what "good" looks like. Production patterns teach scale and performance.
- **Key patterns to include:**
  - Device management (GPU/CPU): `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  - Batch processing: Emphasize batching for throughput, never process single items
  - Mixed precision: `fp16=True` in training for speed and memory savings
  - Error handling: Try-except for OOM, device issues, shape mismatches
  - Caching: Reuse computations where possible (embeddings, tokenization)
  - Model merging: After LoRA training, merge for single-file deployment

### 4. 3-Level Implementation Structure
- **Rule:** Each concept has basic → advanced → real-world examples.
- **Why:** Scaffolded complexity helps readers learn at their own pace.
- **Structure:**
  - **Level 1 - Basic (20-40 lines):** Core concept in isolation, synthetic data, minimal dependencies
  - **Level 2 - Advanced (60-100 lines):** Full pipeline, real data loading, error handling, optimizations
  - **Level 3 - Real-World (2-3 examples, 40-60 lines each):** Different production contexts, integration patterns, scaling
- **How to apply:** When creating new concepts, structure notebooks as: intro → basic → advanced → example1 → example2 → example3 → takeaways

### 5. Interview Question Design
- **Rule:** Questions test pattern recognition and judgment, not memorization.
- **Why:** Interviewers ask "when" and "why", not "define" or "memorize".
- **Good questions:**
  - "How would you optimize this for production?"
  - "What's the trade-off between X and Y?"
  - "When would you NOT use this approach?"
  - "How would you debug this error?"
- **Bad questions:**
  - "Define X" (memorization test)
  - "What's the exact formula?" (no one memorizes; it's in papers)
  - "List all parameters" (implementation details that change)
- **How to apply:** Interview Q&A sections should focus on judgment calls and problem-solving, not definitions.

---

## Content Structure Checklist

Before considering a concept complete, verify:

### Markdown File (llm/concepts/{NN}-{name}.md)
- [ ] Definition: 100-150 words, explains why it matters, clarifies misconceptions
- [ ] How It Works: 200-300 words, includes Mermaid diagram, step-by-step flow
- [ ] Interview Q&A: 5-8 questions focusing on when/why/how to debug
- [ ] Best Practices: 5+ practical tips for production use
- [ ] Common Pitfalls: 3-5 mistakes people actually make
- [ ] Real-World Examples: 2-3 code snippets showing actual usage

### Notebook File (llm/notebooks/{NN}-{name}.ipynb)
- [ ] Introduction: Title, learning objectives, prerequisites
- [ ] Basic Implementation: 20-40 lines, core concept clearly shown
- [ ] Advanced Implementation: 60-100 lines, error handling, optimizations
- [ ] Real-World Example 1: 40-60 lines, common industry use case
- [ ] Real-World Example 2: 40-60 lines, optimization or scaling
- [ ] Real-World Example 3: 40-60 lines, integration with other concepts
- [ ] Key Takeaways: Summary, when to use, links to related concepts
- [ ] All imports are real (not pseudo-code)
- [ ] Device handling is explicit throughout
- [ ] Code includes error handling for common issues
- [ ] Notebook completes in <2 hours

---

## Testing & Validation

**Test File:** tests/test_llm_notebooks.py

**Before committing:**
```bash
python3 -m pytest tests/test_llm_notebooks.py -v
```

**Tests validate:**
- Notebook structure (required cells present, proper cell types)
- Code quality (imports available, syntax valid, no undefined variables)
- Content completeness (required sections present in markdown)
- Consistency (same concepts across markdown and notebooks)

**Quality gates:**
- All tests must pass
- All imports must be available
- Notebooks must be runnable (with mock data if needed)
- No pending pseudo-code or TODOs

---

## Common Patterns (Copy These)

### Device Management (paste at start of production code cells)
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Then use device in tensor creation: tensor = torch.randn(..., device=device)
```

### Error Handling for OOM
```python
try:
    outputs = model(batch_ids.to(device), attention_mask.to(device))
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("❌ GPU out of memory: reduce batch_size or model size")
    else:
        raise
```

### Batch Processing Pattern
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    outputs = model(**batch)
    # Process batch results
```

### Mixed Precision Training
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    fp16=torch.cuda.is_available(),  # Automatic mixed precision
    gradient_accumulation_steps=4,   # Larger effective batch size
    # ... other args
)
```

### Model Merging After LoRA
```python
from peft import AutoPeftModelForCausalLM

# Load LoRA model
model = AutoPeftModelForCausalLM.from_pretrained("path/to/lora-model")
# Merge for single-file deployment
merged_model = model.merge_and_unload()
merged_model.save_pretrained("path/to/merged-model")
```

---

## Generator Scripts

**Primary:** `scripts/generate_realworld_notebooks.py`

**Purpose:** Create/regenerate all 32 concept notebooks with production implementations.

**Run:**
```bash
python3 scripts/generate_realworld_notebooks.py
```

**When to use:**
- After updating markdown files
- When adding new concepts
- For bulk regeneration with new patterns

**Key implementation:**
- Dictionary-based REALWORLD_IMPLEMENTATIONS for each concept
- Fallback to template for concepts without specific implementations
- Validates notebook structure before saving
- Uses nbformat for proper notebook creation

See `memory/generator_scripts.md` for complete details.

---

## Documentation Location

Full learning materials for agents are in:
```
/home/sbisw/.claude/projects/-home-sbisw-github-interviewprep-ml/memory/
```

Memories include:
- `llm_project_context.md` - Project overview and goals
- `llm_concepts_index.md` - All 32 concepts with categories
- `implementation_patterns.md` - Proven HuggingFace patterns
- `notebook_architecture.md` - Standard notebook structure
- `content_best_practices.md` - What makes content effective
- `educational_design.md` - Pedagogical principles
- `testing_patterns.md` - Testing and validation approach
- `generator_scripts.md` - How to use generation tools

---

## What NOT to Do

❌ **Don't:**
- Use pseudo-code or generic templates
- Include explanations without working code
- Create notebooks without error handling
- Process samples one-at-a-time (batch everything)
- Forget device placement (CPU vs GPU)
- Try to merge multiple concepts into one notebook
- Deploy multiple adapters without clear routing strategy
- Use large LoRA ranks (start at 4-8, max 32)
- Ignore context window limits
- Create Q&A that tests memorization instead of judgment

✅ **DO:**
- Use real library imports (transformers, peft, sentence-transformers, torch)
- Show complete working examples with error handling
- Include device management throughout
- Batch process everything (32-256 samples depending on model)
- Test on realistic data
- Validate on your specific task
- Merge adapters before deployment
- Start with small LoRA ranks and increase if underfitting
- Include performance timing and memory notes
- Create Q&A focused on when/why/how to debug

---

## Recent Evolution (Lessons Learned)

### Iteration 1: Theory + Flowcharts
- Created markdown with detailed theory + notebooks with Mermaid diagrams
- **Issue:** Mermaid doesn't render in .ipynb on GitHub
- **Fix:** Migrated all Mermaid diagrams to markdown files

### Iteration 2: Basic Implementations
- Created notebooks with simple generic templates
- **Issue:** Generic templates don't teach transferable skills
- **Feedback:** "Provide better real world implementations with real libraries"
- **Fix:** Added production implementations using actual HuggingFace code

### Iteration 3: Production Code
- All notebooks now use real library imports and production patterns
- Strong emphasis on code > theory in notebooks
- Real-world examples show actual usage patterns
- **Result:** Materials prepare candidates for both interviews and real implementation

### Iteration 4: Curriculum Expansion & Enhancement (Session 8-9, Current)
- Expanded from 32 LLM concepts to 148 total concepts (40 AI + 44 LLM + 64 Agentic-AI)
- **Session 8:** Created all 148 concept files and 148 notebooks following agentic-ai pattern
- **Session 9:** Enhanced all 40 AI fundamentals with comprehensive:
  - Detailed Explanation (150-250 words): context, applications, challenges
  - Core Intuition (2-3 sentences): memorable analogies for quick understanding
- **Completion:** All 148 concepts now follow consistent 8-section markdown + 12-cell notebook pattern
- **Result:** Comprehensive, interview-ready curriculum covering full AI/ML/LLM spectrum

---

## Quick Start for New Agents

1. **Understand project:** Read MEMORY.md index and relevant memory files
   - `ai_fundamentals_enhancement.md` - Session 9 completion (all 40 AI concepts enhanced)
   - `curriculum_expansion_complete.md` - Session 8 completion (148 concepts total)
   - `implementation_patterns.md` - Real library patterns that work
2. **Understand structure:** Follow agentic-ai pattern (8-section markdown, 12-cell notebooks)
3. **Check guidelines:** Review sections above (Code Over Theory, Real Libraries, Production Patterns)
4. **Content checklist:** Review "Content Structure Checklist" section above
5. **Validate:** Ensure all content passes quality checklist above
6. **Memory updates:** Update relevant memory files in `memory/` when completing work

---

## Contact & Updates

When you discover issues or improvements:
1. Update relevant memory file in `/memory/`
2. Update CLAUDE.md if principles change
3. Commit both with clear explanations
4. Add to plan files if systematic changes needed

This keeps knowledge accumulating and prevents regressions.
