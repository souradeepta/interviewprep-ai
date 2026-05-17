"""
Generate Jupyter notebooks for LLM concepts using nbformat.
Reads markdown files and concepts mapping, produces standardized notebooks.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import nbformat as nbf

def read_markdown_file(filepath: str) -> str:
    """Read markdown file and return content."""
    with open(filepath, 'r') as f:
        return f.read()

def extract_tldr(content: str) -> str:
    """Extract TL;DR section from markdown."""
    if "## TL;DR" in content:
        start = content.find("## TL;DR") + len("## TL;DR")
        end = content.find("##", start)
        return content[start:end].strip()
    return ""

def extract_section(content: str, section_name: str) -> str:
    """Extract a section by name from markdown."""
    start_marker = f"## {section_name}"
    if start_marker not in content:
        return f"<!-- {section_name} not found in source -->"

    start = content.find(start_marker) + len(start_marker)
    # Find next ## or end of content
    next_section = content.find("##", start)
    if next_section == -1:
        return content[start:].strip()
    return content[start:next_section].strip()

def create_workflow_flowchart(concept_name: str) -> str:
    """Generate a basic workflow flowchart for a concept."""
    # Basic template - should be customized per concept
    concept_slug = concept_name.lower().replace(" ", "-")
    return f"""```mermaid
graph LR
    A["Input"] --> B["{concept_name} Process"]
    B --> C["Output"]

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#e8f5e9
```

**Note:** This is a basic workflow template. Review and customize based on specific concept."""

def create_relationship_flowchart(concept_key: str, concepts_data: Dict) -> str:
    """Generate a relationship flowchart showing how concept connects to others."""
    concept = concepts_data.get(concept_key, {})
    related = concept.get("related", {})

    mermaid_lines = ["graph TD"]
    mermaid_lines.append(f'    A["{concept.get("title", concept_key)}"]')

    # Add prerequisite relationships
    for prereq_key in concept.get("prerequisites", []):
        prereq_title = concepts_data.get(prereq_key, {}).get("title", prereq_key)
        mermaid_lines.append(f'    B["{prereq_title}"] -->|prerequisite| A')

    # Add dependent relationships
    for dep in related.get("prerequisite_for", []):
        dep_title = concepts_data.get(dep, {}).get("title", dep)
        mermaid_lines.append(f'    A -->|prerequisite for| C["{dep_title}"]')

    # Add "used with" relationships
    for related_concept in related.get("used_with", [])[:3]:  # Limit to 3 for clarity
        related_title = concepts_data.get(related_concept, {}).get("title", related_concept)
        mermaid_lines.append(f'    A -->|used with| D["{related_title}"]')

    mermaid_lines.append("    \n    style A fill:#fff3e0")

    return "```mermaid\n" + "\n".join(mermaid_lines) + "\n```"

def create_code_section(concept_key: str, concept_title: str) -> str:
    """Generate basic code section template."""
    return f"""### Code Implementation

```python
# Key imports for {concept_title}
import numpy as np
import torch
from typing import Any

# {concept_title} example implementation
class {concept_title.replace(' ', '').replace('-', '')}:
    \"\"\"
    {concept_title} implementation.
    This is a template - customize with actual code.
    \"\"\"
    def __init__(self):
        pass

    def process(self, input_data: Any) -> Any:
        # Interview tip: Explain the core insight here
        return input_data
```

### Mathematical Formulation

Include LaTeX equations relevant to this concept.

**Example:**
$$\\text{{Output}} = f(\\text{{Input}})$$"""

def create_qa_section(concept_title: str) -> str:
    """Generate interview Q&A section template."""
    return f"""### Common Interview Questions

**Q: What is {concept_title} used for?**
A: [Add concise answer about practical application]

**Q: What are the main trade-offs of {concept_title}?**
A: [Discuss pros/cons and when to use vs alternatives]

**Q: How does {concept_title} compare to [related concept]?**
A: [Explain key differences and when to use each]

**Q: What are common mistakes when using {concept_title}?**
A: [List 1-2 common pitfalls and how to avoid them]

**Q: Can you explain the intuition behind {concept_title}?**
A: [Provide a simple analogy or explanation]"""

def generate_notebook(concept_key: str, concept_data: Dict, all_concepts: Dict,
                     source_markdown: str) -> nbf.NotebookNode:
    """Generate a single notebook for a concept."""

    nb = nbf.v4.new_notebook()

    title = concept_data.get("title", concept_key)
    tags = concept_data.get("tags", [])
    prerequisites = concept_data.get("prerequisites", [])
    related = concept_data.get("related", {})

    # Cell 1: Metadata
    metadata_content = f"""# {title}

**Tags:** {", ".join(tags)}
**Prerequisites:** {", ".join([all_concepts.get(p, {}).get('title', p) for p in prerequisites]) or "None"}
**Related Concepts:** See flowchart below
**Source:** {concept_data.get('source_file', 'N/A')}"""

    nb.cells.append(nbf.v4.new_markdown_cell(metadata_content))

    # Cell 2: TL;DR
    tldr = extract_tldr(source_markdown)
    nb.cells.append(nbf.v4.new_markdown_cell(f"## TL;DR\n\n{tldr if tldr else '[Add TL;DR from source markdown]'}"))

    # Cell 3: Core Intuition
    intuition = extract_section(source_markdown, "Core Intuition")
    nb.cells.append(nbf.v4.new_markdown_cell(f"## Core Intuition\n\n{intuition}"))

    # Cell 4: How It Works + Workflow Flowchart
    workflow = extract_section(source_markdown, "How It Works")
    flowchart = create_workflow_flowchart(title)
    nb.cells.append(nbf.v4.new_markdown_cell(f"## How It Works\n\n{workflow}\n\n### Workflow Diagram\n\n{flowchart}"))

    # Cell 5: Key Properties
    properties = extract_section(source_markdown, "Key Properties")
    nb.cells.append(nbf.v4.new_markdown_cell(f"## Key Properties & Trade-offs\n\n{properties}"))

    # Cell 6: Code Implementation
    code_section = create_code_section(concept_key, title)
    nb.cells.append(nbf.v4.new_markdown_cell(code_section))

    # Cell 7: Related Concepts Flowchart
    rel_flowchart = create_relationship_flowchart(concept_key, all_concepts)
    nb.cells.append(nbf.v4.new_markdown_cell(f"## Related Concepts\n\n{rel_flowchart}"))

    # Cell 8: Interview Q&A
    qa_section = create_qa_section(title)
    nb.cells.append(nbf.v4.new_markdown_cell(qa_section))

    # Cell 9: References
    references = f"""## References

- **Source Document:** `{concept_data.get('source_file', 'N/A')}`
- **Related Papers:** [Add relevant papers]
- **Implementations:**
  - HuggingFace: [Add links]
  - GitHub: [Add links]"""

    nb.cells.append(nbf.v4.new_markdown_cell(references))

    return nb

def main():
    """Main execution: generate all notebooks."""

    # Load configuration
    mapping_path = Path("data/concepts_mapping.json")
    notebooks_dir = Path("llm/notebooks")
    concepts_dir = Path("llm/concepts")

    if not mapping_path.exists():
        print(f"Error: {mapping_path} not found")
        return

    with open(mapping_path) as f:
        mapping = json.load(f)

    all_concepts = mapping.get("concepts", {})
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    # Sort by order
    sorted_concepts = sorted(
        all_concepts.items(),
        key=lambda x: x[1].get("order", 999)
    )

    for concept_key, concept_data in sorted_concepts:
        order = concept_data.get("order", 0)
        source_file = concept_data.get("source_file")

        # Read source markdown
        markdown_path = concepts_dir / source_file.split("/")[-1]
        if not markdown_path.exists():
            print(f"Warning: Source file not found: {markdown_path}")
            source_content = f"<!-- Source file not found: {source_file} -->"
        else:
            source_content = read_markdown_file(str(markdown_path))

        # Generate notebook
        print(f"Generating notebook for: {concept_data.get('title')} ({order}/33)...")
        notebook = generate_notebook(concept_key, concept_data, all_concepts, source_content)

        # Save notebook
        notebook_name = f"{order:02d}-{concept_key}.ipynb"
        notebook_path = notebooks_dir / notebook_name

        with open(notebook_path, 'w') as f:
            nbf.write(notebook, f)

        print(f"  ✓ Saved: {notebook_path}")

    print(f"\n✓ Generated {len(sorted_concepts)} notebooks in {notebooks_dir}")

if __name__ == "__main__":
    main()
