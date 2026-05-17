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
