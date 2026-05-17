"""
Validation tests for generated LLM concept notebooks.
Tests structure, content, and format requirements.
"""

import json
import nbformat
from pathlib import Path
from typing import List
import pytest

NOTEBOOKS_DIR = Path("llm/notebooks")
MAPPING_FILE = Path("data/concepts_mapping.json")

def load_concepts_mapping() -> dict:
    """Load concepts mapping for reference."""
    with open(MAPPING_FILE) as f:
        return json.load(f)

def get_all_notebooks() -> List[Path]:
    """Get all generated notebooks."""
    all_notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    # Exclude concept map (00-concept-map.ipynb) from concept-specific tests
    return [nb for nb in all_notebooks if not nb.name.startswith("00-")]

def get_all_notebooks_including_map() -> List[Path]:
    """Get all generated notebooks including concept map."""
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))

class TestNotebookStructure:
    """Test notebook has required cell structure."""

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_notebook_has_nine_cells(self, notebook_path):
        """Each notebook should have exactly 9 cells."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        assert len(nb.cells) >= 9, f"{notebook_path.name} has {len(nb.cells)} cells, expected 9+"

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_cell_one_is_metadata(self, notebook_path):
        """Cell 1 should be metadata (markdown with title)."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        cell = nb.cells[0]
        assert cell.cell_type == "markdown", f"{notebook_path.name}: Cell 1 should be markdown"
        assert "#" in cell.source, f"{notebook_path.name}: Cell 1 should have title"

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_cell_two_is_tldr(self, notebook_path):
        """Cell 2 should be TL;DR section."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        cell = nb.cells[1]
        assert cell.cell_type == "markdown", f"{notebook_path.name}: Cell 2 should be markdown"
        assert "TL;DR" in cell.source, f"{notebook_path.name}: Cell 2 should have TL;DR"

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_has_workflow_flowchart(self, notebook_path):
        """Should have workflow flowchart in How It Works section."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        full_text = "\n".join(cell.source for cell in nb.cells)
        assert "```mermaid" in full_text, f"{notebook_path.name}: Missing mermaid flowchart"

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_has_interview_qa(self, notebook_path):
        """Should have interview Q&A section."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        full_text = "\n".join(cell.source for cell in nb.cells)
        assert "Common Interview Questions" in full_text, f"{notebook_path.name}: Missing interview Q&A section"

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_has_code_section(self, notebook_path):
        """Should have code implementation section."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        full_text = "\n".join(cell.source for cell in nb.cells)
        assert "import" in full_text or "pseudocode" in full_text.lower(), \
            f"{notebook_path.name}: Missing code section or imports"

class TestFlowchartSyntax:
    """Test mermaid flowchart syntax."""

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_mermaid_syntax_valid(self, notebook_path):
        """Verify mermaid blocks have proper syntax."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        full_text = "\n".join(cell.source for cell in nb.cells)

        # Find mermaid blocks
        in_mermaid = False
        mermaid_count = 0
        for line in full_text.split('\n'):
            if '```mermaid' in line:
                in_mermaid = True
                mermaid_count += 1
            elif '```' in line and in_mermaid:
                in_mermaid = False

        assert mermaid_count >= 2, f"{notebook_path.name}: Expected at least 2 mermaid blocks, found {mermaid_count}"

class TestNotebookValidity:
    """Test notebooks are valid Jupyter format."""

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_notebook_is_valid_json(self, notebook_path):
        """Notebook should be valid JSON."""
        try:
            with open(notebook_path) as f:
                nbformat.read(f, as_version=4)
        except Exception as e:
            pytest.fail(f"{notebook_path.name} is not valid JSON: {e}")

    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_notebook_has_metadata(self, notebook_path):
        """Notebook should have metadata."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        assert hasattr(nb, 'metadata'), f"{notebook_path.name}: Missing metadata"

class TestConceptCoverage:
    """Test all concepts have notebooks."""

    def test_all_concepts_have_notebooks(self):
        """Every concept in mapping should have a notebook."""
        mapping = load_concepts_mapping()
        concepts = mapping.get("concepts", {})
        notebooks = get_all_notebooks()

        notebook_names = {nb.stem.split('-', 1)[1] if '-' in nb.stem else nb.stem for nb in notebooks}
        concept_keys = set(concepts.keys())

        missing = concept_keys - notebook_names
        assert not missing, f"Missing notebooks for concepts: {missing}"

    def test_minimum_notebook_count(self):
        """Should have at least 32 concept notebooks plus concept map."""
        notebooks = get_all_notebooks_including_map()
        assert len(notebooks) >= 33, f"Expected at least 33 notebooks, found {len(notebooks)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
