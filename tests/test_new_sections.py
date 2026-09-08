"""Regression tests for the statistics, computer-vision, and NLP sections."""

import ast
import re
from pathlib import Path

import nbformat
import pytest


SECTIONS = {
    "stats": 15,
    "cv": 8,
    "nlp": 8,
}


def section_files(section: str):
    root = Path(section)
    return sorted((root / "concepts").glob("*.md")), sorted(
        (root / "notebooks").glob("*.ipynb")
    )


@pytest.mark.parametrize("section, expected_count", SECTIONS.items())
def test_section_has_matching_concepts_and_notebooks(section, expected_count):
    concepts, notebooks = section_files(section)
    assert len(concepts) == expected_count
    assert len(notebooks) == expected_count
    assert {p.stem for p in concepts} == {p.stem for p in notebooks}


@pytest.mark.parametrize("section", SECTIONS)
def test_concepts_have_eight_sections(section):
    concepts, _ = section_files(section)
    for path in concepts:
        text = path.read_text()
        assert len(text.split()) >= 700, path
        assert len(re.findall(r"^## ", text, flags=re.MULTILINE)) == 8, path


@pytest.mark.parametrize("section", SECTIONS)
def test_notebooks_have_required_structure_and_valid_python(section):
    _, notebooks = section_files(section)
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        assert len(notebook.cells) == 12, path
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert len(code.splitlines()) >= 400, path
        ast.parse(code, filename=str(path))


def test_stats_special_topics_are_present():
    ab_testing = Path("stats/notebooks/08-ab-testing-statistics.ipynb").read_text()
    information = Path("stats/notebooks/10-information-theory.ipynb").read_text()
    assert all(term in ab_testing for term in ("CUPED", "SPRT", "Bayesian"))
    assert all(term in information for term in ("entropy", "KL", "mutual"))
