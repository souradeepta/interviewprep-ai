# Generation Scripts - Curriculum Creation Tools

This directory contains all Python scripts used to generate and enhance the AI/ML/LLM/Agentic-AI curriculum content.

## Overview

These scripts automate the creation and enhancement of:
- 148 concept markdown files (8-section structure)
- 149 Jupyter notebooks (12-cell implementation structure)
- Glossary and reference materials

## Script Categories

### Core Concept Generation

- **create_new_concepts.py** - Creates skeleton markdown files for new concepts with title and description
- **create_ai_concepts.py** - Generates AI fundamentals concept files with metadata
- **batch_concepts.py** - Batch processing for multiple concepts at once
- **complete_all_concepts.py** - Fills in complete concept definitions for all 148 concepts

### Content Enhancement

- **enhance_all_ai_fundamentals.py** - Enhances Detailed Explanation and Core Intuition for all 40 AI fundamentals (01-40)
- **enhance_ai_29_40.py** - Enhances advanced AI concepts (29-40) with comprehensive explanations
- **enhance_architecture_sections.py** - Adds Architecture/Trade-offs sections with Mermaid diagrams
- **enhance_architecture_ai.py** - Initial architecture enhancements for AI concepts
- **enhance_all_architecture.py** - Generates architecture templates for remaining concepts
- **enhance_remaining_architectures.py** - Completes architecture sections for AI concepts (06-40)

### Notebook Generation

- **create_ai_notebooks.py** - Generates 12-cell Jupyter notebooks for AI fundamentals
- **auto_nb.py** - Automatic notebook structure generation
- **create_rag_notebook.py** - Creates RAG (Retrieval-Augmented Generation) specific notebook
- **create_memory_notebook.py** - Creates agent memory management notebook
- **create_reflection_notebook.py** - Creates agent reflection/self-improvement notebook
- **create_simulation_notebook.py** - Creates simulation-based learning notebook

### Content Sections

- **add_qa_sections.py** - Adds Interview Q&A sections to concepts
- **add_real_code_examples.py** - Includes production-grade code examples
- **add_remaining_code_examples.py** - Fills in missing code implementations
- **add_systemdesign_qa.py** - Adds system design focused questions

## Usage

### Single Concept Enhancement

```bash
# Enhance Detailed Explanation and Core Intuition
python3 enhance_all_ai_fundamentals.py

# Add Architecture sections
python3 enhance_remaining_architectures.py

# Generate notebooks
python3 create_ai_notebooks.py
```

### Batch Operations

```bash
# Create multiple concepts
python3 batch_concepts.py

# Complete all concept definitions
python3 complete_all_concepts.py
```

### Interactive Sections

```bash
# Add Q&A sections
python3 add_qa_sections.py

# Add code examples
python3 add_real_code_examples.py
```

## Script Features

### Common Patterns

All scripts follow these patterns:

1. **Directory Structure Awareness**
   - Know where ai/, llm/, agentic-ai/ directories are
   - Handle relative and absolute paths correctly

2. **Regex-Based Content Replacement**
   - Find specific markdown sections (## Headers)
   - Replace placeholder content with detailed content
   - Preserve file structure and formatting

3. **Batch Processing**
   - Process multiple files efficiently
   - Report success/failure for each file
   - Provide summary statistics

4. **Data-Driven Approach**
   - Content stored in Python dictionaries
   - Maps concept slugs to detailed definitions
   - Enables easy customization and updates

### Key Libraries Used

```python
import os          # File operations
import re          # Regular expressions for pattern matching
import json        # JSON serialization for notebooks
import nbformat    # Jupyter notebook manipulation
```

## Content Structure Conventions

### Markdown Files (8 Sections)

All concept markdown files follow this structure:

1. **Detailed Explanation** (150-250 words)
   - What the concept is
   - Why it matters
   - Real-world applications
   - Challenges and limitations

2. **Core Intuition** (2-3 sentences)
   - Simple analogy from everyday life
   - Key mechanism explained simply
   - Why it matters for the field

3. **How It Works** (Numbered steps + Mermaid diagram)
   - Step-by-step explanation
   - Visual representation of mechanism
   - Key components and relationships

4. **Architecture / Trade-offs** (Comparison tables + design patterns)
   - Trade-off analysis matrices
   - Strategy selection frameworks
   - Implementation patterns

5. **Interview Q&A** (5-8 judgment-focused questions)
   - When/why/how questions
   - Real-world problem-solving focus
   - No memorization questions

6. **Best Practices** (7-10 production tips)
   - Real-world deployment considerations
   - Common optimization strategies
   - Integration patterns

7. **Common Pitfalls** (4-6 real mistakes)
   - Actual problems practitioners face
   - How to avoid them
   - Early warning signs

8. **Code Examples** (3 implementations)
   - Example 1: Basic implementation
   - Example 2: Production patterns
   - Example 3: Advanced/optimization
   - Real, working code (not pseudo-code)

9. **Related Concepts** (Cross-curriculum links)
   - Prerequisites
   - Dependent concepts
   - Advanced extensions

### Jupyter Notebooks (12 Cells)

All concept notebooks follow this structure:

1. **Title & Learning Objectives** (Markdown)
2. **Level 1: Basic Understanding** (Markdown + Code)
3. **Level 2: Production Implementation** (Markdown + Code)
4. **Real-World Example 1** (Markdown + Code)
5. **Real-World Example 2** (Markdown + Code)
6. **Real-World Example 3** (Markdown + Code)
7. **Key Takeaways** (Markdown)

## Maintenance and Updates

### Running All Enhancement Scripts

To refresh/regenerate all content:

```bash
# Run enhancements in order
python3 enhance_all_ai_fundamentals.py
python3 enhance_ai_29_40.py
python3 enhance_remaining_architectures.py
python3 add_qa_sections.py
python3 add_real_code_examples.py
```

### Adding New Concepts

1. Define concept metadata (title, slug, description)
2. Create skeleton with `create_new_concepts.py`
3. Enhance with detailed explanations
4. Add architecture/trade-offs
5. Generate notebooks
6. Add code examples

### Modifying Existing Concepts

Edit the data dictionaries in the scripts:

```python
EXPANSIONS = {
    "concept-slug": {
        "detailed": "150-250 word explanation...",
        "intuition": "2-3 sentence intuition..."
    }
}
```

Then re-run the appropriate enhancement script to apply changes.

## Content Quality Standards

All generated content meets these standards:

✓ **Detailed Explanation**: 150-250 words minimum
✓ **Core Intuition**: 2-3 sentences with memorable analogy
✓ **Code Examples**: Real, working code (not pseudo-code)
✓ **No Templates**: All content is specific to the concept
✓ **Production-Ready**: Includes error handling, best practices
✓ **Cross-Linked**: Related concepts explicitly connected
✓ **Verified**: All markdown and notebooks validated

## Documentation

For more information:
- See CLAUDE.md for project guidelines
- See AI-ML-GLOSSARY.md for terminology reference
- See individual concept files for complete content

---

*Last Updated: 2026-05-18*
*Scripts Version: 2.0 (Session 9 - Comprehensive Enhancement)*
