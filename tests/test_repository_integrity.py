"""Repository-level checks for links and representative notebook execution."""

import importlib.util
import json
import sys
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
TIER_LEDGER = json.loads((ROOT / "data/notebook_execution_tiers.json").read_text())
TIER_ROWS = {row["path"]: row for row in TIER_LEDGER["notebooks"]}
SMOKE_NOTEBOOKS = tuple(ROOT / path for path, row in TIER_ROWS.items() if row["tier"] == 2)
CLASSIFIER_SPEC = importlib.util.spec_from_file_location(
    "classify_notebook_tiers", ROOT / "scripts/classify_notebook_tiers.py"
)
assert CLASSIFIER_SPEC and CLASSIFIER_SPEC.loader
CLASSIFIER = importlib.util.module_from_spec(CLASSIFIER_SPEC)
CLASSIFIER_SPEC.loader.exec_module(CLASSIFIER)

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


def test_execution_tier_ledger_covers_every_tracked_notebook():
    # Git files are authoritative; ignored/generated notebooks do not belong in the matrix.
    import subprocess
    tracked = set(subprocess.check_output(["git", "ls-files", "*.ipynb"], cwd=ROOT, text=True).splitlines())
    assert set(TIER_ROWS) == tracked
    assert all(row["tier"] in {1, 2, 3} for row in TIER_ROWS.values())
    assert all(row.get("reason") for row in TIER_ROWS.values())
    assert all(row.get("alternative_check") for row in TIER_ROWS.values() if row["tier"] == 3)
    assert all(CLASSIFIER.classify(path) == row for path, row in TIER_ROWS.items())
    assert len(TIER_LEDGER["tier3_representatives"]) == 4
    for representative in TIER_LEDGER["tier3_representatives"]:
        row = TIER_ROWS[representative["path"]]
        assert row["tier"] == 3
        assert representative["check"]
        if "evidence_import" in representative:
            assert representative["evidence_import"] in row["detected_imports"]
        if "evidence_call" in representative:
            assert representative["evidence_call"] in row["detected_resource_calls"]
    for relative in tracked:
        notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        nbformat.validate(notebook)


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


def numbered_ids(directory, suffix):
    return {
        int(path.name.split("-", 1)[0])
        for path in (ROOT / directory).glob(f"[0-9][0-9]-*.{suffix}")
    }


def test_modern_ai_numbering_has_one_paired_artifact_per_topic():
    concepts = numbered_ids("modern-ai/concepts", "md")
    notebooks = numbered_ids("modern-ai/notebooks", "ipynb")
    implementations = numbered_ids("modern-ai/implementations", "py")
    assert len(list((ROOT / "modern-ai/concepts").glob("[0-9][0-9]-*.md"))) == len(concepts)
    assert concepts == notebooks == implementations == set(range(1, 56))


def test_mlops_concepts_have_matching_notebooks_excluding_concept_map():
    concepts = numbered_ids("mlops/concepts", "md")
    notebooks = numbered_ids("mlops/notebooks", "ipynb")
    assert concepts == notebooks - {0}


@pytest.mark.parametrize(
    "section, expected_max",
    (
        ("ai", 40),
        ("llm", 44),
        ("agentic-ai", 64),
        ("stats", 21),
        ("cv", 8),
        ("nlp", 8),
        ("rl", 20),
    ),
)
def test_numbered_concept_and_notebook_ids_stay_in_sync(section, expected_max):
    concepts = numbered_ids(f"{section}/concepts", "md")
    notebooks = numbered_ids(f"{section}/notebooks", "ipynb")
    expected = set(range(1, expected_max + 1))
    assert concepts == expected
    assert notebooks - {0} == expected


def test_ml_concept_and_notebook_counts_are_complete():
    assert len(list((ROOT / "ml/concepts").glob("*.md"))) == 40
    assert len(list((ROOT / "ml/notebooks").glob("*.ipynb"))) == 40


def test_active_section_readmes_match_current_inventory():
    ai_readme = (ROOT / "ai/README.md").read_text()
    mlops_readme = (ROOT / "mlops/README.md").read_text()
    assert "**40 concepts**" in ai_readme
    assert "# 40 markdown files" in ai_readme
    assert "# 40 Jupyter notebooks" in ai_readme
    assert "**28 concepts**" not in ai_readme
    assert "(17-20)" not in mlops_readme
    assert "(21-24)" not in mlops_readme
    assert "(25-28)" not in mlops_readme
    assert "(29-32)" not in mlops_readme


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
