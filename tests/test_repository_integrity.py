"""Repository-level checks for links and representative notebook execution."""

import importlib.util
import sys
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
SMOKE_NOTEBOOKS = (
    ROOT / "stats/notebooks/08-ab-testing-statistics.ipynb",
    ROOT / "cv/notebooks/05-contrastive-learning-vision.ipynb",
    ROOT / "llm/notebooks/18-rag.ipynb",
)

ML_INTERVIEW_TOPICS = (
    "rejection sampling",
    "topic model",
    "ranking metrics",
)


def load_link_auditor():
    path = ROOT / "scripts/audit_markdown_links.py"
    spec = importlib.util.spec_from_file_location("audit_markdown_links", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_active_markdown_links_are_valid():
    auditor = load_link_auditor()
    assert auditor.audit(ROOT) == []


def test_ml_interview_material_covers_recently_recurring_topics():
    content = (ROOT / "ml/interview-prep/ml-coding-questions.md").read_text().lower()
    assert all(topic in content for topic in ML_INTERVIEW_TOPICS)
    for guide in (
        "prefix-sums.md",
        "intervals-sweep-line.md",
        "matrix-grid.md",
        "simulation-state-machines.md",
    ):
        assert (ROOT / "coding/algorithms" / guide).exists()


@pytest.mark.parametrize("notebook_path", SMOKE_NOTEBOOKS)
def test_representative_notebook_executes_without_mutating_source(notebook_path):
    nbclient = pytest.importorskip("nbclient")
    kernelspec = pytest.importorskip("jupyter_client.kernelspec")
    if "python3" not in kernelspec.find_kernel_specs():
        pytest.skip("python3 Jupyter kernel is not installed in this environment")

    notebook = nbformat.read(notebook_path, as_version=4)
    nbclient.NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    ).execute()

    # Execution is performed on an in-memory notebook, so the checked-in file
    # remains free of transient outputs.
    assert not any(cell.get("outputs") for cell in nbformat.read(notebook_path, as_version=4).cells)
