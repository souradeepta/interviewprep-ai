# Contributing to ML Ops Learning Materials

Thank you for your interest in contributing to this ML Ops learning resource! This document provides guidelines for adding concepts, interview questions, and case studies.

## Concept Template

Each ML Ops concept has two main components:

### 1. Markdown File (`concepts/{NN}-{name}.md`)

**Filename convention:** `{NN}-{kebab-case-name}.md`
- Example: `01-data-pipelines.md`, `15-model-registry.md`

**Required sections:**
1. **Title** - Clear concept name
2. **Comprehensive Overview** (4-5 paragraphs, 300-400 words)
   - Industry context and why this matters
   - Key challenges and design decisions
   - Production impact and scale implications
   - Real-world relevance

3. **How It Works** (Architecture + Technical Details)
   - Detailed explanation with ASCII diagrams or Mermaid
   - Step-by-step flow
   - Key decision points

4. **Tool Comparisons** (Multi-tool approach with trade-off tables)
   - 2-3 tools/frameworks side-by-side
   - Decision framework: "When to choose X vs Y"
   - Trade-off table: features, cost, ease of use, scalability
   - Example: MLflow vs W&B vs Neptune for experiment tracking

5. **Interview Q&A** (5-8 judgment-focused questions)
   - NOT: "Define data drift"
   - YES: "How would you detect data drift? What signals? False positives?"
   - Focus on: when to use, why, trade-offs, real-world challenges
   - Provide answer outlines (not full answers)

6. **Best Practices** (5+ production patterns)
   - What experienced engineers do
   - Common pitfalls and how to avoid them
   - Production considerations
   - Cost and performance tips

7. **Real-World Case Studies** (2-3 FAANG examples)
   - "How Netflix designed their feature store"
   - "How Uber detects model drift at scale"
   - Lessons learned and trade-offs made
   - Quantifiable impact (e.g., "reduced latency from 500ms to 50ms")

8. **Sample Interview Questions** (3-5 FAANG questions)
   - Actual questions from Google, Meta, Netflix, Uber, Amazon
   - Difficulty level (Medium/Hard)
   - Answer outline included

9. **Interview Case Study** (Full scenario walkthrough)
   - Setup: context, constraints, requirements
   - Problem: specific challenge to solve
   - Expected solution structure
   - Follow-ups: interviewer questions
   - Strong vs weak answer patterns
   - What interviewers are evaluating

10. **Common Answer Patterns** (Strong vs Weak responses)
    - Weak example with explanation of what's missing
    - Strong example with structured approach
    - What interviewers listen for
    - How to improve weak answers

**Quality Standards:**
- ~700-1000 words total
- Clear, concise writing
- Real examples (not hypothetical)
- Production-focused language
- Interview-prep mindset throughout

### 2. Jupyter Notebook (`notebooks/{NN}-{name}.ipynb`)

**Filename convention:** `{NN}-{kebab-case-name}.ipynb`
- Example: `01-data-pipelines.ipynb`, `15-model-registry.ipynb`

**Required cells:**
1. **Introduction & Learning Objectives**
   - What you'll learn
   - Which interview questions this covers
   - Prerequisites
   - Key concepts to understand

2. **Basic Implementation** (20-40 lines)
   - Core concept in isolation
   - Minimal dependencies
   - Synthetic data
   - Focuses on understanding the core idea

3. **Advanced Implementation** (60-100 lines)
   - Production patterns
   - Error handling
   - Logging and monitoring
   - Real-world complexity

4. **Real-World Example 1** (40-60 lines)
   - Common industry use case
   - Realistic constraints
   - Example: "Implement basic experiment tracking"
   - What a real system looks like

5. **Real-World Example 2** (40-60 lines)
   - Optimization/scaling scenario
   - Example: "Scale experiment tracking for 100K runs/day"
   - Production considerations
   - Cost/performance trade-offs

6. **Real-World Example 3** (40-60 lines)
   - Integration with other concepts
   - Example: "Integrate experiment tracking with model registry"
   - How systems work together

7. **Interview Scenario** (Annotated code)
   - Real interview question as a coding exercise
   - Step-by-step solution
   - Comments: "This is what the interviewer is looking for"
   - Follow-up questions and answers
   - Why this approach beats alternatives

8. **Key Takeaways**
   - Interview-focused summary
   - What you need to know
   - Common follow-up questions
   - How to explain in 2 min vs 10 min

**Code Quality Standards:**
- ✅ All imports are real (MLflow, Airflow, Feast, FastAPI, etc.)
- ✅ Production patterns throughout (error handling, logging, monitoring)
- ✅ Code is tested and runnable
- ✅ Multi-tool approach: show 2-3 implementations where relevant
- ✅ No pseudo-code or generic templates
- ✅ Comments explain "why" not "what"
- ✅ Realistic data and scenarios

## Adding Interview Questions

**File:** `interview-questions/questions.json`

**Format:**
```json
{
  "id": "dp-001",
  "concept": "01-data-pipelines",
  "difficulty": "medium",
  "company": ["Google", "Meta"],
  "question": "Design a data pipeline for a recommendation system processing 1M events/second.",
  "answer_outline": "Separate batch and streaming. Batch for training...",
  "follow_ups": ["How would you handle failures?", "What about cost?"]
}
```

**Guidelines:**
- One question per JSON object
- Difficulty: "easy", "medium", "hard"
- Company: array of companies where this was asked
- Answer outline: 1-2 sentences on approach
- Follow-ups: 2-3 likely interviewer questions
- 3-5 questions per concept

## Adding Case Studies

**File:** `case-studies/{company/problem}-{concept}.md`

**Examples:**
- `stripe-fraud-detection.md`
- `netflix-feature-pipeline.md`
- `uber-monitoring-at-scale.md`

**Structure:**
1. **Scenario** - Setup and context
2. **Constraints** - What makes it hard
3. **Problem Statement** - Specific challenge
4. **Expected Solution Components** - Architecture and approach
5. **Strong Answer Pattern** - What a good answer looks like
6. **Weak Answer Pattern** - Common mistakes
7. **Interviewer Follow-ups** - Expected next questions
8. **What Interviewers Evaluate** - Criteria for success
9. **Real-World Lessons** - What actually happened at company

## PR Process

1. **Create a branch:** `feature/mlops-concept-{name}`
2. **Add concept files:**
   - `mlops/concepts/{NN}-{name}.md`
   - `mlops/notebooks/{NN}-{name}.ipynb`
3. **Add supporting materials:**
   - Interview questions to `interview-questions/questions.json`
   - Case study (if major concept) to `case-studies/`
4. **Test:**
   - Verify notebooks run without errors
   - Check markdown formatting
   - Validate JSON syntax for questions
5. **Commit message:**
   ```
   feat: add concept {NN}: {name}
   
   Add comprehensive concept with:
   - Markdown theory and best practices
   - Jupyter notebook with production code
   - {N} interview questions
   - Real-world case study
   ```
6. **Open PR** with description of what's new

## Testing

Run tests to validate your contributions:

```bash
# Test markdown files
pytest tests/test_mlops_concepts.py -v

# Test notebooks
pytest tests/test_mlops_notebooks.py -v

# Test interview questions
pytest tests/test_mlops_interviews.py -v
```

## Tips for Quality Content

### Markdown Tips
- Use real examples (not hypothetical)
- Include quantifiable metrics (e.g., "2-4x faster")
- Link related concepts
- Keep sentences short and clear
- Use tables for comparisons

### Notebook Tips
- Test code before submitting
- Use meaningful variable names
- Include error handling
- Add timing information (how long does each step take?)
- Show output of code runs
- Explain design choices in comments

### Interview Question Tips
- Base on real questions when possible
- Vary difficulty levels
- Include follow-ups
- Show answer structure, not full answer
- Relate to production systems

## Style Guide

- **Tone:** Professional, practical, interview-focused
- **Tense:** Present tense for concepts, past for case studies
- **Voice:** Active voice preferred
- **Length:** Markdown ~800 words, Notebooks ~300 lines code
- **Examples:** Always use real tools/frameworks

## Resources

- [Original LLM Concepts](../llm/concepts/) - Reference for structure
- [LLM Notebooks](../llm/notebooks/) - Reference for code style
- [CLAUDE.md](../CLAUDE.md) - Project guidelines

## Questions?

If you have questions about contributing:
1. Check existing concepts for examples
2. Review the MLOPS_ROADMAP.md for context
3. Look at similar concepts for structure

Thank you for contributing to this learning resource!
