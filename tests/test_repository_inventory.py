"""Regression checks for the generated repository inventory report."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_repository_inventory.py"
SPEC = importlib.util.spec_from_file_location("report_repository_inventory", SCRIPT)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_track_counts_exclude_folder_readmes_and_count_nested_artifacts(tmp_path, monkeypatch):
    track = tmp_path / "example" / "concepts"
    (track / "nested").mkdir(parents=True)
    (track / "README.md").write_text("index", encoding="utf-8")
    (track / "lesson.md").write_text("lesson", encoding="utf-8")
    (track / "nested" / "another.md").write_text("lesson", encoding="utf-8")
    monkeypatch.setattr(REPORT, "ROOT", tmp_path)

    assert REPORT.count_files("example", "concepts", "*.md") == 2
