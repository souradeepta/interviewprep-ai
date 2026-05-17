"""
Remove mermaid code blocks from notebooks and replace with text references.
Diagrams are now in the markdown files where they render on GitHub.
"""

import json
import re
from pathlib import Path

NOTEBOOKS_DIR = Path("llm/notebooks")

def clean_mermaid_from_cell(cell_text):
    """Remove mermaid code block and replace with reference."""
    # Remove mermaid code blocks
    cleaned = re.sub(r'\n*```mermaid\n.*?\n```\n*', '', cell_text, flags=re.DOTALL)

    # If cell had a mermaid block, add a reference
    if '```mermaid' in cell_text:
        # Add reference to markdown file
        cleaned += "\n**Note:** Flowchart diagrams are in the source markdown file (`llm/concepts/{concept}.md`) for better rendering on GitHub."

    return cleaned

def process_notebook(notebook_path):
    """Remove mermaid blocks from notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)

    modified = False

    # Process each cell
    for cell in nb['cells']:
        if 'mermaid' in ''.join(cell.get('source', [])):
            cell_text = ''.join(cell.get('source', []))
            cleaned_text = clean_mermaid_from_cell(cell_text)

            if cell_text != cleaned_text:
                cell['source'] = cleaned_text.split('\n')
                # Add newlines back properly
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                modified = True

    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        return True
    return False

def main():
    """Process all notebooks."""
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    updated_count = 0

    for notebook_path in notebooks:
        if process_notebook(notebook_path):
            print(f"✓ Cleaned: {notebook_path.name}")
            updated_count += 1

    print(f"\n✓ Cleaned {updated_count} notebooks")

if __name__ == "__main__":
    main()
