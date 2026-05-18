# Generation Scripts Index

Quick reference for all Python scripts used in curriculum generation.

## Alphabetical Index

### A

**add_qa_sections.py**
- Purpose: Add Interview Q&A sections to concept markdown files
- Input: Concept files with placeholder Q&A sections
- Output: Enhanced concepts with 5-8 judgment-focused interview questions
- Status: Used in content creation pipeline

**add_real_code_examples.py**
- Purpose: Add production-grade code examples to concepts
- Input: Concept markdown files
- Output: Three working code examples per concept
- Status: Critical for code-over-theory approach

**add_remaining_code_examples.py**
- Purpose: Fill in missing code implementations for incomplete concepts
- Input: Concepts with partial or no code examples
- Output: Complete code examples (Level 1, 2, and real-world)
- Status: Used for completeness pass

**add_systemdesign_qa.py**
- Purpose: Add system design-focused interview questions
- Input: Concept Q&A sections
- Output: Enhanced Q&A with architectural design questions
- Status: Optional enhancement for infrastructure concepts

**auto_nb.py**
- Purpose: Automatically generate notebook structure from concept metadata
- Input: Concept definitions and metadata
- Output: Skeleton Jupyter notebooks with proper cell structure
- Status: Foundation for notebook generation

### B

**batch_concepts.py**
- Purpose: Batch process multiple concepts at once
- Input: List of concept definitions
- Output: Multiple markdown files and/or notebooks
- Status: Efficiency tool for bulk operations

### C

**complete_all_concepts.py**
- Purpose: Fill in complete concept definitions for all 148 concepts
- Input: Concept slugs and titles
- Output: Complete markdown files with full 8-section structure
- Status: Core generation script

**create_ai_concepts.py**
- Purpose: Create AI fundamentals concept files (01-40)
- Input: Concept metadata (title, slug, description)
- Output: Markdown files for ai/concepts/ directory
- Status: Used in AI fundamentals creation

**create_ai_notebooks.py**
- Purpose: Generate 12-cell Jupyter notebooks for AI concepts
- Input: AI concept definitions
- Output: Notebooks in ai/notebooks/ directory
- Status: Critical for notebook generation

**create_memory_notebook.py**
- Purpose: Create agent memory management concept notebook
- Input: Memory management patterns and examples
- Output: Complete 12-cell notebook
- Status: Specialized notebook for agentic-ai section

**create_new_concepts.py**
- Purpose: Create skeleton markdown files for new concepts
- Input: Concept title and description
- Output: Markdown file with basic structure (no content)
- Status: Foundation script for new content

**create_rag_notebook.py**
- Purpose: Create Retrieval-Augmented Generation notebook
- Input: RAG concept and examples
- Output: Complete 12-cell notebook with RAG patterns
- Status: Specialized notebook for agentic-ai section

**create_reflection_notebook.py**
- Purpose: Create agent reflection/self-improvement notebook
- Input: Reflection patterns and examples
- Output: Complete 12-cell notebook
- Status: Specialized notebook for agentic-ai section

**create_simulation_notebook.py**
- Purpose: Create simulation-based learning notebook
- Input: Simulation examples and patterns
- Output: Complete 12-cell notebook
- Status: Specialized notebook for agentic-ai section

### E

**enhance_ai_29_40.py**
- Purpose: Enhance Detailed Explanation and Core Intuition for AI concepts 29-40
- Input: AI concept markdown files (29-40)
- Output: Markdown files with comprehensive explanations and intuitions
- Status: Session 9 completion script

**enhance_ai_concepts.py**
- Purpose: Enhance multiple AI concept sections
- Input: AI concept definitions
- Output: Enhanced markdown files
- Status: General enhancement tool

**enhance_all_ai_fundamentals.py**
- Purpose: Enhance all 40 AI fundamentals with comprehensive explanations
- Input: All AI concept files (01-40)
- Output: Enhanced Detailed Explanation and Core Intuition sections
- Status: Session 9 main enhancement script

**enhance_all_architecture.py**
- Purpose: Generate comprehensive Architecture sections for all concepts
- Input: Concept files with placeholder architecture sections
- Output: Detailed architecture content with tables and diagrams
- Status: Incomplete (template structure)

**enhance_architecture_ai.py**
- Purpose: Enhance architecture sections for AI concepts with Mermaid diagrams
- Input: AI concept files
- Output: Architecture/Trade-offs sections with comparison tables and diagrams
- Status: Early version (replaced by more comprehensive scripts)

**enhance_architecture_sections.py**
- Purpose: Add detailed Architecture/Trade-offs sections with Mermaid diagrams
- Input: Concept markdown files
- Output: Enhanced architecture sections with visual diagrams
- Status: Core enhancement script

**enhance_remaining_architectures.py**
- Purpose: Complete Architecture/Trade-offs sections for AI concepts 06-40
- Input: AI concept markdown files (06-40)
- Output: Detailed architecture content with comparison tables
- Status: Session 9 completion script

## Usage by Purpose

### Concept Creation (Initial)

1. **create_new_concepts.py** - Create skeleton files
2. **complete_all_concepts.py** - Fill in basic content
3. **auto_nb.py** - Generate notebook structure

### Content Enhancement (Iterative)

1. **enhance_all_ai_fundamentals.py** - Add explanations/intuitions
2. **enhance_remaining_architectures.py** - Add architecture sections
3. **add_qa_sections.py** - Add interview questions
4. **add_real_code_examples.py** - Add code implementations

### Notebook Generation

1. **create_ai_notebooks.py** - For AI concepts
2. **create_rag_notebook.py** - For RAG-specific content
3. **create_memory_notebook.py** - For memory management
4. **create_reflection_notebook.py** - For agent reflection
5. **create_simulation_notebook.py** - For simulation patterns

### Specialized Enhancements

- **add_systemdesign_qa.py** - For architecture-focused concepts
- **batch_concepts.py** - For bulk operations

## Typical Generation Pipeline

```
1. Create skeleton files
   └─ create_new_concepts.py

2. Fill in basic definitions
   └─ complete_all_concepts.py

3. Generate notebook structure
   └─ auto_nb.py → create_ai_notebooks.py

4. Enhance explanations
   └─ enhance_all_ai_fundamentals.py

5. Add architecture content
   └─ enhance_remaining_architectures.py

6. Add interview Q&A
   └─ add_qa_sections.py

7. Add code examples
   └─ add_real_code_examples.py

8. Final polish & validation
   └─ Manual review + commit
```

## Script Dependencies

### Core Dependencies (All Scripts)

```python
import os          # File operations
import re          # Pattern matching
```

### Optional Dependencies

```python
import json        # Notebook creation
import nbformat    # Jupyter notebook format
```

## Configuration

### Paths (Hardcoded in Scripts)

All scripts use:
```python
BASE = "/home/sbisw/github/interviewprep-ml"
```

Change this variable to adapt scripts for different repository locations.

### Data Dictionaries

Content is stored in Python dictionaries:
- `EXPANSIONS`: Detailed explanation + intuition
- `ARCHITECTURES`: Architecture/Trade-offs content
- `QA_CONTENT`: Interview questions
- `CODE_EXAMPLES`: Implementation examples

Modify these dictionaries to update content.

## Performance Notes

- **Fastest**: Create skeleton files (~1 second per concept)
- **Moderate**: Enhance existing content (~2-5 seconds per concept)
- **Slowest**: Generate complete notebooks with code (~5-10 seconds per concept)

### Batch Processing Times

- 40 concepts: ~2-5 minutes
- 148 concepts: ~10-20 minutes
- Full pipeline: ~45-60 minutes

## Version History

### Version 2.0 (Session 9 - Current)
- Comprehensive enhancement of all 148 concepts
- Architecture/Trade-offs for all concepts
- Detailed Explanation and Core Intuition for all 40 AI concepts
- Glossary integration

### Version 1.0 (Earlier Sessions)
- Initial concept creation
- Basic notebook generation
- Simple enhancement scripts

## Future Improvements

Potential enhancements to scripts:
- [ ] Parallel processing for faster batch operations
- [ ] Database/YAML for content instead of Python dicts
- [ ] Interactive CLI for script selection
- [ ] Automated testing of generated content
- [ ] Content validation (word count, structure checks)
- [ ] Diff preview before applying changes

---

*Generated: 2026-05-18*
*Curriculum Version: Complete (148 concepts + 149 notebooks + glossary)*
