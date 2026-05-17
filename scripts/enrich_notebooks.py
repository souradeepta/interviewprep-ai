"""
Enrich generated notebooks with real content from markdown files.
Extracts sections, code examples, and trade-off tables.
"""

import json
import nbformat as nbf
from pathlib import Path
import re

NOTEBOOKS_DIR = Path("llm/notebooks")
CONCEPTS_DIR = Path("llm/concepts")
MAPPING_FILE = Path("data/concepts_mapping.json")

def extract_section(content: str, section: str) -> str:
    """Extract a section from markdown content."""
    start_marker = f"## {section}"
    if start_marker not in content:
        return ""

    start = content.find(start_marker) + len(start_marker)
    next_section = content.find("##", start)
    if next_section == -1:
        return content[start:].strip()
    return content[start:next_section].strip()

def enrich_notebook(notebook_path: Path, source_markdown: str):
    """Enrich a single notebook with content from source markdown."""

    with open(notebook_path) as f:
        nb = nbf.read(f, as_version=4)

    # Update Cell 4 (How It Works) with actual content
    if len(nb.cells) > 3:
        how_it_works = extract_section(source_markdown, "How It Works")
        if how_it_works:
            workflow_content = f"## How It Works\n\n{how_it_works}"
            # Keep the flowchart that's already there
            if "Workflow Diagram" in nb.cells[3].source:
                flowchart_start = nb.cells[3].source.find("### Workflow Diagram")
                if flowchart_start != -1:
                    flowchart = nb.cells[3].source[flowchart_start:]
                    workflow_content += f"\n\n{flowchart}"
            nb.cells[3].source = workflow_content

    # Update Cell 5 (Properties) with actual content
    if len(nb.cells) > 4:
        properties = extract_section(source_markdown, "Key Properties")
        if properties:
            props_content = f"## Key Properties & Trade-offs\n\n{properties}"
            nb.cells[4].source = props_content

    # Update Cell 2 (TL;DR) with actual content
    if len(nb.cells) > 1:
        tldr = extract_section(source_markdown, "TL;DR")
        if tldr:
            tldr_content = f"## TL;DR\n\n{tldr}"
            nb.cells[1].source = tldr_content

    # Update Cell 3 (Core Intuition) with actual content
    if len(nb.cells) > 2:
        intuition = extract_section(source_markdown, "Core Intuition")
        if intuition:
            intuition_content = f"## Core Intuition\n\n{intuition}"
            nb.cells[2].source = intuition_content

    # Save enriched notebook
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

def main():
    """Enrich all notebooks."""
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    concepts = mapping.get("concepts", {})
    enriched_count = 0

    for concept_key, concept_data in concepts.items():
        source_file = concept_data.get("source_file")
        markdown_path = CONCEPTS_DIR / source_file.split("/")[-1]

        if not markdown_path.exists():
            print(f"Warning: Source file not found: {markdown_path}")
            continue

        # Read markdown
        with open(markdown_path) as f:
            content = f.read()

        # Find corresponding notebook
        order = concept_data.get("order", 0)
        notebook_path = NOTEBOOKS_DIR / f"{order:02d}-{concept_key}.ipynb"

        if notebook_path.exists():
            print(f"Enriching: {concept_key}...")
            enrich_notebook(notebook_path, content)
            enriched_count += 1
        else:
            print(f"Warning: Notebook not found: {notebook_path}")

    print(f"✓ Enriched {enriched_count} notebooks")

if __name__ == "__main__":
    main()
