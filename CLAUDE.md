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

## Detailed Markdown File Format (8 Sections)

Each markdown concept file MUST follow this exact structure for consistency. Use the word counts and guidelines below.

### 1. Detailed Explanation (150-250 words)
**Purpose:** Establish context, why it matters, and practical significance.

**Must include:**
- What the concept is and why practitioners care about it
- Real-world applications or consequences of getting it wrong
- Common misconceptions to clarify
- Connection to larger ML/AI workflow

**Example structure:**
```
Gradient descent is the foundational optimization algorithm used to train nearly 
all neural networks... [Context: what is it]

The algorithm comes in three main variants... [Why it matters: practical implications]

Understanding gradient descent is essential because... Modern practitioners rarely 
implement it from scratch but understanding the algorithm is crucial for... 
[Why understanding matters: debugging, architectural decisions]
```

### 2. Core Intuition (2-3 sentences MAX)
**Purpose:** One memorable analogy or metaphor for quick mental model.

**Guidelines:**
- Avoid jargon
- Use visceral, relatable imagery
- Should be memorable enough to recall under stress (interview)
- EXACTLY 2-3 sentences; no more

**Examples:**
- ✅ "Imagine you're at the top of a mountain in the dark and want to reach the valley below. You can't see far ahead, but you can feel the slope beneath your feet. Gradient descent is like repeatedly taking steps downhill."
- ❌ "Gradient descent is a first-order optimization algorithm that iteratively updates parameters in the direction of steepest descent."

### 3. How It Works (200-300 words + flowchart)
**Purpose:** Mechanistic explanation with clear steps and Mermaid diagram.

**Must include:**
- 4-6 numbered steps showing the algorithm/concept in action
- Step-by-step flow with decision points where applicable
- Mermaid diagram (NOT in notebooks, only in markdown)
- Mathematical notation if relevant (e.g., w = w - lr × ∇L)

**Example structure:**
```
1. Initialize weights randomly
2. Compute loss on training data
3. Calculate gradient ∂L/∂w
4. Update: w = w - lr × ∇L
5. Repeat until convergence

[Mermaid diagram showing the loop]
```

### 4. Architecture / Trade-offs (300-400 words)
**Purpose:** Compare variants, trade-offs, when to use each, with tables/visualizations.

**Must include:**
- 2-4 comparison tables (variants, parameter ranges, use cases)
- Clear pros/cons for each approach
- Trade-off analysis (e.g., speed vs stability, simplicity vs power)
- When to use each in practice (specific conditions, datasets, constraints)

**Example structure:**
```
### Variant 1 vs Variant 2

| Metric | Variant 1 | Variant 2 |
|--------|-----------|-----------|
| Speed | Fast | Slow |
| Stability | Unstable | Stable |
| Memory | Low | High |

### Trade-offs
- Variant 1 is faster but... (condition when this matters)
- Use Variant 2 when... (specific scenario)
```

### 5. Interview Q&A (5-8 questions, 2-3 sentences each)
**Purpose:** Test judgment, debugging skills, and practical knowledge—NOT memorization.

**Guidelines:**
- Each Q&A is exactly ONE concept-based question + short answer
- Focus on "when/why/how to debug" not "define X" or "list parameters"
- Answer should be actionable, not theoretical
- Include at least one question about failure modes

**Good pattern:**
```
Q: When would you use [Variant A] instead of [Variant B]?
A: When [specific condition/constraint], because [practical reason]. Example: [brief scenario]

Q: What are the first signs your model is failing due to [issue]?
A: You'd see [observable symptom], which indicates [root cause]. Debug by [specific action].

Q: What's the trade-off you'd face if you chose [Variant A]?
A: [Speed/complexity/accuracy trade-off]. In practice, this matters when [scenario].
```

**Bad patterns (❌ avoid):**
```
❌ Q: Define [concept]
❌ Q: What's the formula for [X]?
❌ Q: List all parameters
❌ Q: Explain the history of [concept]
```

### 6. Best Practices (5-8 bullet points)
**Purpose:** Actionable tips for production use from practitioners.

**Guidelines:**
- Each bullet is one specific, operable practice
- Start with "Normalize features to...", "Use X when...", "Monitor Y to detect..."
- Include numbers/ranges where applicable (e.g., "batch size of 32-256", "learning rate 0.001-0.1")
- Avoid generic advice like "test your code"

**Example pattern:**
```
- Normalize features to [-1, 1] or [0, 1] for stable learning
- Use mini-batch (32-256) for good gradient estimates and hardware efficiency
- Monitor loss on validation set to detect overfitting
- Use learning rate scheduling to improve convergence
- Start with lr=0.01 and adjust based on loss curves
```

### 7. Common Pitfalls (3-5 mistakes with consequences)
**Purpose:** Real mistakes practitioners make, what goes wrong, how to fix.

**Guidelines:**
- Each pitfall is ONE mistake people actually make (not theoretical)
- Describe what goes wrong (symptom) and why
- Include how to detect it and quick fix
- Prioritize by frequency (most common first)

**Example pattern:**
```
- **Learning rate too high:** Weights oscillate and diverge (loss spikes or becomes NaN)
  → Fix: Reduce lr by 10x, or use learning rate scheduling
  
- **Batch size too small:** Very noisy gradient estimates, poor convergence
  → Fix: Increase to 32-256 depending on model size
```

### 8. Code Examples (2-4 runnable code blocks, 40-60 lines each)
**Purpose:** Real, importable code demonstrating the concept.

**Guidelines:**
- Use real libraries (numpy, torch, transformers, sklearn)
- NO pseudo-code or generic templates
- Each example should run independently
- Example 1: Basic (20-40 lines) - simplest working version
- Example 2+: More complex scenarios (production patterns, error handling)
- Include comments explaining key lines
- Show output/results when relevant (print statements)

**Example structure:**
```python
# Example 1: Basic implementation
import numpy as np
def gradient_descent(...): pass

# Example 2: Production with error handling
try:
    # ... training code
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("❌ OOM: reduce batch_size")

# Example 3: Real-world library usage
from torch import ...
```

### Optional: Related Concepts (3-5 links)
Point to related concepts within the same domain.

```
- [Related Concept 1](./XX-related-1.md)
- [Related Concept 2](./XX-related-2.md)
```

---

## Detailed Notebook Structure (12 Cells)

Each notebook MUST have exactly 12 cells in this order. Each cell has a specific purpose.

### Cell 1: Title & Learning Objectives (Markdown)
```markdown
# [Concept Name]

## Learning Objectives
1. Understand [concept core mechanism]
2. Implement [concept variations]
3. Analyze [key trade-off or measurement]
4. Compare [X vs Y in practice]
```

### Cell 2: Prerequisites & Imports (Code)
```python
# Quick imports check
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Cell 3: Level 1 - Basic Implementation (Markdown header + Code)
```markdown
## Level 1: Basic [Concept]
```

**Code (20-40 lines):**
- Simplest working version of the concept
- Use synthetic/toy data
- No error handling yet
- Show the core idea clearly
- Include output/printing to verify it works

### Cell 4: Level 1 - Basic Implementation Output (Code)
```python
# Run the basic implementation above and show results
print("Result:", ...)
# or show a simple plot
```

### Cell 5: Level 2 - Advanced Implementation (Markdown header + Code)
```markdown
## Level 2: Advanced [Concept]
```

**Code (60-100 lines):**
- Add complexity: multiple variants, error handling, production patterns
- Use more realistic data (but still synthetic if needed)
- Include device management
- Add timing and memory measurements
- Compare multiple approaches
- Error handling for common issues (OOM, shape mismatch, etc.)

### Cell 6: Level 2 - Advanced Output (Code)
```python
# Run advanced version and visualize results
# Plot comparison, timing results, memory usage, etc.
```

### Cell 7: Real-World Example 1 (Markdown header + Code)
```markdown
## Real-World Example 1: [Specific use case]
```

**Code (40-60 lines):**
- One specific, realistic scenario (e.g., "Training on MNIST", "Fine-tuning a pretrained model")
- Use real libraries/data where feasible
- Include data loading, preprocessing, training loop
- Production patterns (mixed precision, gradient accumulation, etc.)
- Error handling appropriate for the scenario
- Realistic hyperparameters

### Cell 8: Real-World Example 2 (Markdown header + Code)
```markdown
## Real-World Example 2: [Different use case]
```

**Code (40-60 lines):**
- Different scenario from Example 1 (e.g., optimization strategy, scalability pattern)
- Show how the concept applies in a different context
- Demonstrate trade-off or parameter choice

### Cell 9: Real-World Example 3 (Markdown header + Code)
```markdown
## Real-World Example 3: [Third use case]
```

**Code (40-60 lines):**
- Show integration with other concepts or advanced application
- Could be: debugging a failure mode, handling a constraint, integration with related concept

### Cell 10: Comparison & Visualization (Markdown header + Code)
```markdown
## Comparison: When to Use What
```

**Code:**
- Create a comparison table or visualization
- Show performance/speed/accuracy across variants
- Include plot(s) with labeled axes and legend
- Make the trade-offs visually obvious

### Cell 11: Key Takeaways (Markdown)
```markdown
## Key Takeaways

**Core idea:** [1-2 sentence summary]

**Variants and when to use:**
| Method | Use when | Trade-off |
|--------|----------|-----------|
| ... | ... | ... |

**Common failure modes:**
- Failure 1: [symptom] → [fix]
- Failure 2: [symptom] → [fix]

**Related concepts:**
- [Concept A](./XX-concept-a.ipynb) – [how it relates]
- [Concept B](./XX-concept-b.ipynb) – [how it relates]
```

### Cell 12: Exercises (Optional, Markdown)
```markdown
## Try It Yourself

1. **Modify Example 1:** Change [parameter] from X to Y and observe [effect]
2. **Combine concepts:** Apply Example 1 + [Related Concept] and compare
3. **Debug the failure:** Example 2 fails when [scenario] — fix it
```

---

## Domain-Specific Guidelines

### AI Fundamentals (40 concepts)
**Focus:** Core ML concepts that form the foundation.

**Examples:** Gradient Descent, Cross-Validation, Activation Functions, Loss Functions, etc.

**Content emphasis:**
- **Detailed Explanation:** Why this concept is foundational (e.g., "used in every neural network")
- **Code Examples:** Implementations from scratch using numpy first, then with libraries
- **Best Practices:** When to use, parameter ranges, common misconceptions
- **Interview Q&A:** Focus on "why" questions (why this vs that, why would you choose this)

**Notebook specifics:**
- Level 1 uses numpy only (shows the math clearly)
- Level 2 introduces torch/sklearn
- Real-world examples show practical scenarios from CV/NLP

### LLM Concepts (44 concepts)
**Focus:** Concepts specific to large language models and transformers.

**Examples:** Tokenization, Embeddings, Fine-tuning, LoRA, RLHF, RAG, etc.

**Content emphasis:**
- **Detailed Explanation:** State-of-the-art context, recent advances, open problems
- **Code Examples:** Heavy use of transformers/HuggingFace libraries
- **Best Practices:** Production deployment patterns, memory optimization, inference tricks
- **Interview Q&A:** Focus on deployment trade-offs, scaling, optimization challenges

**Notebook specifics:**
- Include device management throughout (models are large)
- Use real pretrained models (HuggingFace hub)
- Include memory measurements and batching considerations
- Show both training and inference patterns

### Agentic AI Concepts (64 concepts)
**Focus:** Building and deploying AI agents.

**Examples:** Tool Use, Planning, Memory Management, Multi-Agent Systems, etc.

**Content emphasis:**
- **Detailed Explanation:** Architecture choices, reliability, safety considerations
- **Code Examples:** Agent loop patterns, tool calling implementations, state management
- **Best Practices:** Error handling, logging, monitoring agent behavior, cost optimization
- **Interview Q&A:** Focus on "how to debug agent behavior", "how to structure for reliability"

**Notebook specifics:**
- Emphasize structured output and error handling
- Show multi-step reasoning patterns
- Include logging and observability from the start
- Real examples with actual tool calling APIs

---

## Interview Q&A Writing Pattern

**Framework for each Q&A:**
1. Ask about a specific scenario or trade-off (not a definition)
2. Answer with context + reasoning + example
3. Include numbers or specific cases where applicable

**Good Q&A template:**
```
Q: When would you use [Approach A] instead of [Approach B] in production?

A: Use [A] when [specific condition], because [practical reason based on trade-off]. 
Example: [concrete scenario]. The main cost is [what you give up], which is 
acceptable when [when that cost matters less].
```

**Bad Q&A template (❌ avoid):**
```
Q: What is [concept]?
Q: Explain the difference between X and Y.
Q: What are the pros and cons of [X]?
```

---

## Code Quality Standards

**MUST have in every code example:**

1. **Real imports** - No pseudo-code
   ```python
   ✅ from transformers import AutoModel
   ❌ from imaginary_library import Model
   ```

2. **Device management** (if using torch)
   ```python
   ✅ device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.to(device)
   ❌ model = Model()  # ignores GPU
   ```

3. **Batching** - Never process samples one at a time
   ```python
   ✅ for batch in DataLoader(dataset, batch_size=32):
   ❌ for sample in dataset:
   ```

4. **Error handling** - At least for OOM and common mistakes
   ```python
   ✅ try:
        output = model(input)
      except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM: reduce batch_size")
   ❌ output = model(input)  # fails silently or cryptically
   ```

5. **Comments** - Explain WHY, not WHAT
   ```python
   ✅ # Normalize to [-1, 1] range for stable learning rates
      X = (X - X.mean()) / X.std()
   ❌ # Normalize X
      X = (X - X.mean()) / X.std()
   ```

---

## Content Validation Checklist

**Before considering a concept complete, verify ALL of these:**

### Markdown File Checklist
- [ ] Detailed Explanation: 150-250 words, includes WHY it matters
- [ ] Core Intuition: Exactly 2-3 sentences, memorable and jargon-free
- [ ] How It Works: 4-6 numbered steps + Mermaid diagram
- [ ] Architecture/Trade-offs: 2+ comparison tables, includes "use this when..."
- [ ] Interview Q&A: 5-8 questions, zero memorization questions, all have context
- [ ] Best Practices: 5-8 bullets, each is specific and actionable (numbers included)
- [ ] Common Pitfalls: 3-5 mistakes with symptoms and fixes
- [ ] Code Examples: 2-4 blocks, all with real imports, no pseudo-code
- [ ] Related Concepts: 3-5 links pointing to related files
- [ ] Length: Markdown file is 1000-1500 words total (balanced across sections)

### Notebook File Checklist
- [ ] Cell 1: Title + 4 learning objectives
- [ ] Cell 2: Imports + device setup + seed
- [ ] Cell 3-4: Level 1 basic (20-40 lines code + output)
- [ ] Cell 5-6: Level 2 advanced (60-100 lines code + output)
- [ ] Cells 7-9: Three real-world examples (40-60 lines each)
- [ ] Cell 10: Comparison visualization with table/plot
- [ ] Cell 11: Takeaways with variant table + failure modes
- [ ] Cell 12: Exercises or extensions (optional)
- [ ] All imports are real and importable
- [ ] All code is runnable (uses synthetic data if needed)
- [ ] Device management present in torch examples
- [ ] Error handling for OOM/shape/type errors
- [ ] No undefined variables or forward references
- [ ] Execution time < 2 hours (usually 5-10 minutes for AI/ML/LLM, <30 min for agentic)

### Cross-File Consistency Checklist
- [ ] Concept name matches between markdown and notebook filename
- [ ] Learning objectives in notebook align with markdown sections
- [ ] Code examples in markdown are similar quality/style as notebook code
- [ ] No concepts referenced in markdown that don't exist as files
- [ ] Notebook examples don't contradict markdown explanations

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

**Current Project Status:**
- `subproject_a_complete.md` - Sub-Project A completion (modern-ai 36-55): 20 concepts + 20 notebooks, 398/398 tests passing
- `curriculum_expansion_complete.md` - 148 concepts (40 AI + 44 LLM + 64 agentic-ai) framework
- `ai_fundamentals_enhancement.md` - All 40 AI fundamentals enhanced (150-250 word explanations + 2-3 sentence intuitions)

**Implementation Reference:**
- `implementation_patterns.md` - Proven patterns for transformers, PEFT, torch, sklearn
- `claude_guidelines_session10.md` - Detailed templates for 8-section markdown + 16+ cell notebooks
- `content_best_practices.md` - What makes content stick (code > theory, real examples)
- `educational_design.md` - Interview Q&A design, pedagogical principles
- `testing_patterns.md` - Validation approach (398 tests covering structure + correctness)

**Project Infrastructure:**
- `notebook_architecture.md` - Standard 12-16 cell structure (basic → advanced → examples → comparison)
- `generator_scripts.md` - Python scripts for batch notebook generation
- `llm_project_context.md` - Original project scope and goals

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

### Iteration 4: Curriculum Expansion & Enhancement (Session 8-9)
- Expanded from 32 LLM concepts to 148 total concepts (40 AI + 44 LLM + 64 Agentic-AI)
- **Session 8:** Created all 148 concept files and 148 notebooks following agentic-ai pattern
- **Session 9:** Enhanced all 40 AI fundamentals with comprehensive:
  - Detailed Explanation (150-250 words): context, applications, challenges
  - Core Intuition (2-3 sentences): memorable analogies for quick understanding
- **Completion:** All 148 concepts now follow consistent 8-section markdown + 12-cell notebook pattern
- **Result:** Comprehensive, interview-ready curriculum covering full AI/ML/LLM spectrum

### Iteration 5: Comprehensive Repo Enhancement (Session 11, Current)
- Implementing comprehensive enhancement plan targeting 5 sub-projects across 260+ files
- **Sub-Project A (Modern-AI concepts 36-55):** ✅ **NOW COMPLETE**
  - All 20 concept markdown files: Complete with 8-section structure (1600-2300 words each)
  - All 20 Jupyter notebooks: Complete with 16+ cells each
  - Enhanced notebooks 46-50: Expanded from 200-400 lines to 600-900 lines
  - Completed notebooks 51, 53, 54: Added missing Real-World Example 3 + Comparison sections
  - **Quality:** All code uses real libraries, 398/398 tests passing ✓
  - **Pattern:** Every concept has basic → advanced → 3 real-world examples + comparison
- **Remaining sub-projects:** B (SDE3 patterns), C (diagrams), D (coding), E (navigation) — partially started
- **Result:** Production-ready learning materials with comprehensive coverage of optimization, inference, and deployment techniques

---

## Quick Start for New Agents — One-Shot Content Generation

**Goal:** Create high-quality concepts and notebooks that need zero iteration.

### Before You Start
1. **Read this entire CLAUDE.md file** — especially the new detailed sections below
2. Read 1-2 existing concepts in your domain to understand quality bar
3. Review the appropriate "Domain-Specific Guidelines" section

### Step-by-Step for Writing a New Concept

**Markdown File (`[domain]/concepts/[NN]-[name].md`):**
1. Use the "Detailed Markdown File Format" section as your template
2. Follow the exact 8-section structure
3. Check your work against "Markdown File Checklist"

**Notebook File (`[domain]/notebooks/[NN]-[name].ipynb`):**
1. Use the "Detailed Notebook Structure" section as your 12-cell blueprint
2. Follow the exact cell order and purposes
3. Paste the Cell 2 device setup template verbatim
4. Check your work against "Notebook File Checklist"

**Validation:**
1. Run the notebook end-to-end (should complete in < 2 hours)
2. Verify all imports work and no undefined variables
3. Verify file structure matches a good example from the repo
4. Review "Content Validation Checklist" before considering done

### Knowledge Base
Read these memory files if you need deep context:
- `subproject_a_complete.md` - **Session 11:** Sub-Project A (modern-ai 36-55) completion with 40 files
- `ai_fundamentals_enhancement.md` - **Session 9:** All 40 AI concepts with 150-250 word explanations
- `curriculum_expansion_complete.md` - **Session 8:** 148 concepts framework (40 AI + 44 LLM + 64 agentic-ai)
- `implementation_patterns.md` - Real library patterns that work (transformers, PEFT, torch, sklearn)

---

## Contact & Updates

When you discover issues or improvements:
1. Update relevant memory file in `/memory/`
2. Update CLAUDE.md if principles change
3. Commit both with clear explanations
4. Add to plan files if systematic changes needed

This keeps knowledge accumulating and prevents regressions.
