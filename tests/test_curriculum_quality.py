"""Manifest schema and quality-audit regression tests."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_curriculum.py"
SPEC = importlib.util.spec_from_file_location("audit_curriculum", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_checked_in_manifest_records_all_audited_tracks():
    report = AUDIT.audit()
    assert report["ok"], report["errors"]
    assert report["lesson_count"] == 248
    assert report["baseline_count"] == 76
    assert report["required_count"] == 172
    data, errors = AUDIT._manifest_data(ROOT / "data/curriculum_manifest.json")
    assert not errors
    from collections import Counter
    assert Counter(lesson["track"] for lesson in data["lessons"]) == {"ml": 40, "modern-ai": 10, "mlops": 8, "stats": 6, "llm": 44, "agentic-ai": 64, "ai": 40, "rl": 20, "cv": 8, "nlp": 8}


def test_checked_in_manifest_prerequisites_are_resolved_and_acyclic():
    data, errors = AUDIT._manifest_data(ROOT / "data/curriculum_manifest.json")
    assert not errors
    result, _ = AUDIT._validate_manifest(data, ROOT, "ml")
    assert not result
    assert len(data["entry_lessons"]) == 80
    assert {"ai/gradient-descent", "rl/markov-decision-processes", "cv/image-classification", "cv/vision-transformers", "nlp/text-preprocessing", "llm/tokenization"} <= set(data["entry_lessons"])
    assert "llm/tokenization" in data["entry_lessons"]


def _valid_concept() -> str:
    sections = [
        "# Fixture Concept\n\nTL;DR: This lesson explains a deterministic fixture topic.",
        "## Prerequisites\nNo earlier lesson is required for this test fixture.",
        "## Learning Outcomes\n- Explain the behavior of the fixture method.\n- Compare it with a baseline.\n- Select it under a stated constraint.",
        "## Core Intuition\nA concrete analogy is a small labeled drawer: each input follows one predictable route.",
        "## How It Works\nFor input x, the rule computes y = 2x. The assumption is a fixed deterministic mapping; its time complexity is O(1).",
        "## Trade-offs\n| Choice | Benefit | Cost |\n|---|---|---|\n| Direct | Simple | Limited |\n| Cached | Faster repeats | Memory |\n| Batched | Throughput | Queue delay |",
        "## Implementation\n```python\ndef double(value):\n    return 2 * value\n\nassert double(3) == 6\n```",
        "## Failure Modes\n- Overflow: inspect numeric range and use a wider type.\n- Stale cache: compare version keys and invalidate.\n- Bad input: validate the type at the boundary.",
        "## Interview Quick Reference\n| Question | Answer | Follow-up |\n|---|---|---|\n| What does it do? | Doubles input. | Debug an overflow. |\n| What is the cost? | Constant time. | Design caching. |\n| When would you avoid it? | When the input contract differs. | Debug validation. |",
        "## Related Concepts\nSee [the practice guide](../practice.md#practice) and the [Python documentation](https://docs.python.org/3/). A second authoritative reference is https://peps.python.org/pep-0008/.",
    ]
    # Expand substantive explanation so the fixture crosses the prose threshold
    # without making word-count behavior depend on headings or code.
    sections[4] += "\n\n" + ("A measured decision uses a held-out sample and compares errors under the same conditions. " * 115)
    return "\n\n".join(sections)


def _valid_notebook() -> dict:
    sources = [
        "# Fixture notebook\nPrerequisites: none. Estimated time: 20 minutes. Objectives: understand the method.",
        "Concept recap: assume deterministic inputs and compare the measured output.",
        "Baseline implementation: def double(value): return 2 * value",
        "assert 2 * 3 == 6",
        "Extension: compare a transparent implementation with the baseline.",
        "Comparison metric: count mismatches across the examples.",
        "Failure cases: test an edge case and a failure mode explicitly.",
        "Exercise: add a negative input and predict the result.",
        "Summary: review results. Next lesson: none. Practice: answer the linked question.",
    ]
    cells = [{"cell_type": "markdown", "metadata": {}, "source": source} for source in sources]
    cells[2]["cell_type"] = "code"
    cells[3]["cell_type"] = "code"
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}


def _fixture_manifest(root: Path, *, required: bool = True) -> Path:
    (root / "concepts").mkdir(parents=True)
    (root / "notebooks").mkdir()
    (root / "concepts/fixture.md").write_text(_valid_concept(), encoding="utf-8")
    (root / "notebooks/fixture.ipynb").write_text(json.dumps(_valid_notebook()), encoding="utf-8")
    (root / "practice.md").write_text(
        "# Practice\n\n## Practice\nPrompt: x. Role/Domain: test. Difficulty: easy. Expected Answer: x. "
        "Evaluation: correctness. Follow-up: why? Common Mistake: guessing.", encoding="utf-8"
    )
    data = {
        "schema_version": 1,
        "entry_lessons": ["test/fixture"],
        "lessons": [{
            "id": "test/fixture", "track": "test", "level": "foundation",
            "concept": "concepts/fixture.md", "notebook": "notebooks/fixture.ipynb",
            "implementation": None, "prerequisites": [],
            "outcomes": ["Explain the fixture.", "Use the fixture."], "practice": ["practice.md"],
            "estimated_minutes": 20, "quality_status": "required" if required else "baseline",
        }],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    return manifest


def test_required_fixture_satisfies_concept_notebook_and_link_contract(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    result = AUDIT.audit(tmp_path, manifest)
    assert result["ok"], result["errors"]
    assert result["required_count"] == 1


@pytest.mark.parametrize("phrase", AUDIT.SCAFFOLD_PHRASES)
def test_scaffold_denylist_is_enforced_for_required_concepts(tmp_path, phrase):
    manifest = _fixture_manifest(tmp_path)
    concept = tmp_path / "concepts/fixture.md"
    concept.write_text(concept.read_text(encoding="utf-8") + f"\n\n{phrase}\n", encoding="utf-8")
    result = AUDIT.audit(tmp_path, manifest)
    assert not result["ok"]
    assert any("scaffold phrase" in issue and phrase in issue for issue in result["errors"])


def test_required_fixture_rejects_short_concept(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    concept = tmp_path / "concepts/fixture.md"
    concept.write_text("# Short\n\nTL;DR: Short.", encoding="utf-8")
    result = AUDIT.audit(tmp_path, manifest)
    assert any("prose words; minimum is 900" in issue for issue in result["errors"])


def test_missing_artifact_and_malformed_notebook_are_errors(tmp_path):
    manifest = _fixture_manifest(tmp_path, required=False)
    (tmp_path / "notebooks/fixture.ipynb").write_text("{invalid", encoding="utf-8")
    result = AUDIT.audit(tmp_path, manifest)
    assert not result["ok"]
    assert any("malformed notebook JSON" in issue for issue in result["errors"])
    (tmp_path / "concepts/fixture.md").unlink()
    result = AUDIT.audit(tmp_path, manifest)
    assert any("missing concept artifact" in issue for issue in result["errors"])


def test_manifest_rejects_duplicate_ids_unknown_prerequisites_and_cycles(tmp_path):
    manifest = _fixture_manifest(tmp_path, required=False)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    lesson = data["lessons"][0]
    lesson["prerequisites"] = ["test/missing"]
    duplicate = dict(lesson)
    duplicate["prerequisites"] = ["test/missing", "test/fixture"]
    data["lessons"].append(duplicate)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    result = AUDIT.audit(tmp_path, manifest)
    assert any("duplicate lesson id" in issue for issue in result["errors"])
    assert any("unresolved prerequisite" in issue for issue in result["errors"])
    assert any("prerequisite cycle" in issue for issue in result["errors"])


def test_cli_exit_codes_and_json_format(tmp_path, capsys):
    good = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json", "--track", "ml"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["lesson_count"] == 40

    manifest = _fixture_manifest(tmp_path, required=False)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["lessons"][0]["concept"] = "missing.md"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert bad.returncode == 1
    assert "missing concept artifact" in bad.stderr


def test_inventory_reports_scope_and_missing_without_claiming_unindexed_coverage(tmp_path):
    manifest = _fixture_manifest(tmp_path, required=False)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    other = dict(data["lessons"][0], id="other/fixture", track="other", quality_status="required")
    data["entry_lessons"].append(other["id"])
    data["lessons"].append(other)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "concepts/fixture.md").unlink()
    result = AUDIT.audit(tmp_path, manifest, "test")
    assert result["inventory_by_track"]["test"] == {"required": 0, "baseline": 1, "excluded": 0, "missing": 1}
    assert result["inventory_by_track"]["other"] == {"required": 0, "baseline": 0, "excluded": 1, "missing": 0}
    assert "Unindexed lessons are not counted" in result["inventory_definition"]


def test_audit_lists_unindexed_concepts_notebooks_and_implementations(tmp_path):
    manifest = _fixture_manifest(tmp_path, required=False)
    (tmp_path / "test/concepts").mkdir(parents=True)
    (tmp_path / "test/notebooks").mkdir()
    (tmp_path / "test/implementations").mkdir()
    (tmp_path / "test/concepts/extra.md").write_text("extra", encoding="utf-8")
    (tmp_path / "test/notebooks/extra.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "test/implementations/extra.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "test/concepts/README.md").write_text("index", encoding="utf-8")
    (tmp_path / "test/concepts/nested").mkdir()
    (tmp_path / "test/concepts/nested/indexed.md").write_text("indexed", encoding="utf-8")

    report = AUDIT.audit(tmp_path, manifest)

    assert report["unindexed_artifacts_by_track"]["test"] == {
        "concepts": ["test/concepts/extra.md", "test/concepts/nested/indexed.md"],
        "notebooks": ["test/notebooks/extra.ipynb"],
        "implementations": ["test/implementations/extra.py"],
    }
    lessons = [
        {"track": "test", "concept": "test/concepts/nested/./indexed.md"},
        {"track": "other", "concept": "test/concepts/extra.md"},
        None,
    ]
    selected = AUDIT._unindexed_artifacts(tmp_path, lessons, "test")
    assert selected == {"test": {"concepts": ["test/concepts/extra.md"], "notebooks": ["test/notebooks/extra.ipynb"], "implementations": ["test/implementations/extra.py"]}}
    assert AUDIT._unindexed_artifacts(tmp_path, [{"track": "../outside"}]) == {}
    assert AUDIT._unindexed_artifacts(tmp_path, [], "../outside") == {
        "../outside": {"concepts": [], "notebooks": [], "implementations": []}
    }


def test_generic_repetition_is_reported_as_warning_not_proof_of_quality(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    result = AUDIT.audit(tmp_path, manifest)
    assert result["ok"], result["errors"]
    assert any("generic filler" in warning and "concepts/fixture.md" in warning for warning in result["warnings"])
    assert "human review" in result["human_review"]


def test_relative_and_same_page_links_and_duplicate_anchors(tmp_path):
    source = tmp_path / "page.md"
    content = "# Start\n## Repeated\n## Repeated\n[ok](#repeated-1) [bad](#absent) [missing](gone.md)\n"
    source.write_text(content, encoding="utf-8")
    errors = AUDIT._check_links("test/fixture", source, content, tmp_path)
    assert any("broken local anchor" in error and "#absent" in error for error in errors)
    assert any("broken local link" in error and "gone.md" in error for error in errors)
    assert not any("#repeated-1" in error for error in errors)


@pytest.mark.parametrize("mutation, expected", [
    (lambda nb: nb.update(nbformat=3), "valid v4 JSON"),
    (lambda nb: nb["cells"].append({"cell_type": "code", "source": [1], "metadata": {}, "outputs": [], "execution_count": None}), "invalid source"),
    (lambda nb: nb.update(cells=[]), "minimum is 9"),
])
def test_malformed_notebooks_have_actionable_errors(tmp_path, mutation, expected):
    manifest = _fixture_manifest(tmp_path)
    notebook = _valid_notebook()
    mutation(notebook)
    (tmp_path / "notebooks/fixture.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    result = AUDIT.audit(tmp_path, manifest)
    assert any(expected in error and "notebooks/fixture.ipynb" in error for error in result["errors"])


def test_assertion_keyword_in_comment_or_string_does_not_satisfy_notebook():
    notebook = _valid_notebook()
    notebook["cells"][3]["source"] = '# assert a result\nmessage = "assert"'
    assert not AUDIT._has_assertion_token(notebook)


def test_misleading_assertion_proxy_is_reported_without_breaking_compatibility(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    notebook = _valid_notebook()
    notebook["cells"][3]["source"] = '# assert a result\nmessage = "assert"'
    (tmp_path / "notebooks/fixture.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    result = AUDIT.audit(tmp_path, manifest)
    assert result["ok"], result["errors"]
    assert any("no detectable Python assertion token" in warning for warning in result["warnings"])


def test_multi_lesson_cycle_and_invalid_prerequisite_type_are_reported(tmp_path):
    manifest = _fixture_manifest(tmp_path, required=False)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    first = data["lessons"][0]
    second = dict(first, id="test/second", prerequisites=["test/fixture"])
    first["prerequisites"] = ["test/second", {"invalid": True}]
    data["lessons"].append(second)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    result = AUDIT.audit(tmp_path, manifest)
    assert any("prerequisite cycle" in error and "test/second" in error for error in result["errors"])
    assert any("unresolved prerequisite" in error for error in result["errors"])
