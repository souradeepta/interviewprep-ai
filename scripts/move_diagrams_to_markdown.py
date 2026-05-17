"""
Extract mermaid diagrams from notebooks and add them to markdown concept files.
This ensures diagrams render on GitHub (which doesn't render mermaid in .ipynb).
"""

import json
import re
from pathlib import Path

NOTEBOOKS_DIR = Path("llm/notebooks")
CONCEPTS_DIR = Path("llm/concepts")
MAPPING_FILE = Path("data/concepts_mapping.json")

def extract_mermaid_blocks(text):
    """Extract all mermaid code blocks from text."""
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def get_mermaid_from_notebook(notebook_path):
    """Extract workflow and relationship diagrams from notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)

    workflow_diagram = None
    relationship_diagram = None

    # Cell 3 typically has workflow flowchart (How It Works)
    if len(nb['cells']) > 3:
        cell_text = ''.join(nb['cells'][3].get('source', []))
        diagrams = extract_mermaid_blocks(cell_text)
        if diagrams:
            workflow_diagram = diagrams[0]

    # Cell 6 typically has relationship flowchart (Related Concepts)
    if len(nb['cells']) > 6:
        cell_text = ''.join(nb['cells'][6].get('source', []))
        diagrams = extract_mermaid_blocks(cell_text)
        if diagrams:
            relationship_diagram = diagrams[0]

    return workflow_diagram, relationship_diagram

def add_diagrams_to_markdown(markdown_path, workflow_diagram, relationship_diagram):
    """Add mermaid diagrams to markdown file."""
    with open(markdown_path) as f:
        content = f.read()

    # Check if diagrams already exist
    if 'mermaid' in content:
        return False  # Already has mermaid diagrams

    # Add workflow diagram after "How It Works" section
    if workflow_diagram and "## How It Works" in content:
        workflow_section = f"\n### Workflow Flowchart\n\n```mermaid\n{workflow_diagram}\n```\n"
        # Insert after How It Works section
        how_it_works_pos = content.find("## How It Works")
        next_section = content.find("\n##", how_it_works_pos + 1)
        if next_section != -1:
            content = content[:next_section] + workflow_section + content[next_section:]
        else:
            content += workflow_section

    # Add relationship diagram at end or in a new section
    if relationship_diagram:
        relationship_section = f"\n## Concept Relationships\n\n```mermaid\n{relationship_diagram}\n```\n"
        # Add at the end before any existing References section
        if "## References" in content:
            ref_pos = content.find("## References")
            content = content[:ref_pos] + relationship_section + "\n" + content[ref_pos:]
        else:
            content += relationship_section

    # Write updated content
    with open(markdown_path, 'w') as f:
        f.write(content)

    return True

def main():
    """Process all notebooks and add diagrams to markdown files."""
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    concepts = mapping.get("concepts", {})
    updated_count = 0

    for concept_key, concept_data in concepts.items():
        order = concept_data.get("order", 0)
        source_file = concept_data.get("source_file")

        # Find notebook
        notebook_path = NOTEBOOKS_DIR / f"{order:02d}-{concept_key}.ipynb"
        markdown_path = CONCEPTS_DIR / source_file.split("/")[-1]

        if not notebook_path.exists() or not markdown_path.exists():
            continue

        # Extract diagrams from notebook
        workflow_diagram, relationship_diagram = get_mermaid_from_notebook(notebook_path)

        if workflow_diagram or relationship_diagram:
            print(f"Processing: {concept_key}...")
            if add_diagrams_to_markdown(markdown_path, workflow_diagram, relationship_diagram):
                print(f"  ✓ Added diagrams to {markdown_path.name}")
                updated_count += 1
            else:
                print(f"  - Already has diagrams")

    print(f"\n✓ Updated {updated_count} markdown files with diagrams")

if __name__ == "__main__":
    main()
